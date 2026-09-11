#!/usr/bin/env python3
"""Galaxy10 morphology retention for sequential and joint image representations."""

from __future__ import annotations
import argparse
import sys
import json
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from Evals.cross_modal_backbone_eval.evaluate_frozen_backbones import (
    amp_context, load_crossmodal_image, torch_load_checkpoint,
)
from Evals.galaxy10_checkpoint_evolution.galaxy10_probe_evolution import (
    evaluate_split_seed, read_labels, summarize_repeats,
)
from models.cross_modal_posttrained import CrossModalPosttrainedModel

CACHE = ROOT / "artifacts" / "galaxy10"
RESULTS = ROOT / "results"
H5 = ROOT.parent / "Evals/DeCals_linearProbing/galaxy10/Galaxy10_DECals.h5"
SEQ = ROOT.parent / "checkpoints/CrossModalPostTrain_DESI307K_AstroCLIPPool_LeJEPA_SIGReg/last.pt"
JOINT = ROOT.parent / "checkpoints/CrossModalScratch_DESI307K_AllPairs_CrossOnly_SIGReg_4GPU/last.pt"


def load_sequential(device):
    checkpoint = torch_load_checkpoint(SEQ)
    config = checkpoint["config"]
    model = CrossModalPosttrainedModel(
        image_model_name=config["image_model_name"],
        shared_dim=config["shared_dim"],
        spectra_patch_size=config["spectra_patch_size"],
        spectra_num_patches=config["spectra_num_patches"],
        spectra_embed_dim=config["spectra_embed_dim"],
        spectra_depth=config["spectra_depth"],
        spectra_num_heads=config["spectra_num_heads"],
        spectra_mlp_ratio=config["spectra_mlp_ratio"],
        spectra_dropout=config["spectra_dropout"],
        spectra_pooling=config["spectra_pooling"],
        pool_num_heads=config["pool_num_heads"],
        pool_dropout=config["pool_dropout"],
    )
    sources = checkpoint["source_checkpoints"]
    loaded = model.load_pretrained_backbones(
        sources["image"]["path"], sources["spectrum"]["path"])
    model.source_metadata = loaded
    model.load_alignment_state_dict(checkpoint["alignment_heads"])
    metadata = {"checkpoint": str(SEQ), "source_checkpoints_recorded": sources,
                "source_checkpoints_loaded": loaded}
    return model.to(device).eval(), metadata



@torch.inference_mode()
def extract(regime, device, batch_size=256):
    CACHE.mkdir(parents=True, exist_ok=True)
    if regime == "sequential":
        model, metadata = load_sequential(device)
        names = ("U_I", "S_I")
    else:
        model, metadata = load_crossmodal_image(JOINT, device)
        names = ("J_I_raw", "J_I")
    with h5py.File(H5, "r") as handle:
        images = handle["images"]
        arrays = None
        for start in range(0, len(images), batch_size):
            stop = min(start + batch_size, len(images))
            batch = torch.from_numpy(np.asarray(images[start:stop], dtype=np.uint8))
            batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255)
            batch = F.interpolate(batch, (140, 140), mode="bicubic",
                                  align_corners=False, antialias=True)
            with amp_context(device):
                if regime == "sequential":
                    raw, projected = model._encode_image_tokens(batch[:, None])
                else:
                    raw, projected = model(batch[:, None])
            raw, projected = raw[0].float().cpu(), projected[0].float().cpu()
            if arrays is None:
                arrays = (np.empty((len(images), raw.shape[1]), np.float32),
                          np.empty((len(images), projected.shape[1]), np.float32))
            arrays[0][start:stop], arrays[1][start:stop] = raw.numpy(), projected.numpy()
            if start % (batch_size * 10) == 0:
                print(regime, stop, "/", len(images), flush=True)
    for name, array in zip(names, arrays):
        np.save(CACHE / f"{name}.npy", array)
    (CACHE / f"{regime}_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")


def probe(state, device):
    embedding = torch.from_numpy(np.load(CACHE / f"{state}.npy"))
    labels = read_labels(H5)
    repeats = [evaluate_split_seed(
        embeddings=embedding, labels_np=labels, split_seed=seed,
        l2_values=[1e-6, 1e-4, 1e-3, 1e-2],
        lbfgs_iterations=60, device=device) for seed in (42, 43, 44)]
    payload = {
        "state": state, "dataset": str(H5), "rows": len(labels),
        "preprocessing": "RGB /255, bicubic resize to 140, no channel normalization",
        "protocol": "80/10/10 stratified, class-balanced linear probe, seeds 42-44",
        "repeats": repeats, "summary": summarize_repeats(repeats),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"galaxy10_{state}.json").write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract"); ex.add_argument("--regime", choices=("sequential", "joint")); ex.add_argument("--device")
    pr = sub.add_parser("probe"); pr.add_argument("--state", choices=("U_I", "S_I", "J_I_raw", "J_I")); pr.add_argument("--device")
    args = parser.parse_args()
    if args.cmd == "extract": extract(args.regime, torch.device(args.device))
    else: probe(args.state, torch.device(args.device))


if __name__ == "__main__":
    main()

