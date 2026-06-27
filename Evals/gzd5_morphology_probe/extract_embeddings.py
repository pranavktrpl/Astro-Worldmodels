#!/usr/bin/env python3
"""Extract frozen Astro-Worldmodels embeddings for GZD-5 images."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

if "wandb" not in sys.modules:
    wandb_stub = types.ModuleType("wandb")
    wandb_stub.log = lambda *args, **kwargs: None
    sys.modules["wandb"] = wandb_stub

import timm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"

MODELS = {
    "large": {
        "label": "astro_vit_large_step_52000",
        "checkpoint": REPO_ROOT / "checkpoints/VitLargePatch14_OfficialTrain5_Epoch5_2504/step_52000.pt",
    },
    "small": {
        "label": "astro_vit_small_step_21000",
        "checkpoint": REPO_ROOT / "checkpoints/VitSmallPatch14_2204/step_21000.pt",
    },
}


class ImagePathDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_size: int):
        self.paths = frame["file_loc"].tolist()
        self.ids = frame["id_str"].tolist()
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return {"image": tensor, "id_str": self.ids[idx]}


def torch_load(path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": False}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def load_backbone(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch_load(checkpoint_path)
    cfg = dict(checkpoint.get("cfg") or {})
    model_name = str(cfg["model_name"])
    backbone = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True,
        dynamic_img_pad=True,
    )
    state = {
        key.removeprefix("backbone."): value
        for key, value in checkpoint["model"].items()
        if key.startswith("backbone.")
    }
    backbone.load_state_dict(state, strict=True)
    meta = {
        "checkpoint": str(checkpoint_path.resolve()),
        "global_step": int(checkpoint.get("global_step", -1)),
        "cfg": cfg,
        "model_name": model_name,
        "embedding_dim": int(getattr(backbone, "num_features", -1)),
        "backbone_params": int(sum(v.numel() for v in state.values() if torch.is_tensor(v))),
    }
    del checkpoint, state
    gc.collect()
    backbone.to(device).eval()
    return backbone, meta


@torch.inference_mode()
def extract_split(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    output_path: Path,
    ids_path: Path,
    image_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> None:
    if output_path.exists() and ids_path.exists():
        print(f"cached: {output_path}")
        return
    missing = [p for p in frame["file_loc"].head(100).tolist() if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing extracted images, e.g. {missing[0]}")

    loader = DataLoader(
        ImagePathDataset(frame, image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    features: list[np.ndarray] = []
    ids: list[str] = []
    for batch in tqdm(loader, desc=f"embedding {output_path.stem}"):
        images = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
        embeddings = model(images)
        features.append(embeddings.detach().float().cpu().numpy())
        ids.extend(batch["id_str"])

    array = np.concatenate(features, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, array)
    ids_path.write_text("\n".join(ids) + "\n")
    print(f"saved: {output_path} {array.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), default="large")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=140)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    spec = MODELS[args.model]
    output_dir = RESULTS_DIR / spec["label"]
    train = pd.read_parquet(DATA_DIR / "merged_train.parquet")
    test = pd.read_parquet(DATA_DIR / "merged_test.parquet")

    device = torch.device(args.device)
    backbone, meta = load_backbone(spec["checkpoint"], device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"using DataParallel over {torch.cuda.device_count()} visible GPUs")
        backbone = torch.nn.DataParallel(backbone)

    extract_split(
        backbone,
        train,
        output_dir / "train_embeddings.npy",
        output_dir / "train_ids.txt",
        args.image_size,
        args.batch_size,
        args.workers,
        device,
    )
    extract_split(
        backbone,
        test,
        output_dir / "test_embeddings.npy",
        output_dir / "test_ids.txt",
        args.image_size,
        args.batch_size,
        args.workers,
        device,
    )
    meta.update(
        {
            "image_size": args.image_size,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "embedding_files": {
                "train": str((output_dir / "train_embeddings.npy").resolve()),
                "test": str((output_dir / "test_embeddings.npy").resolve()),
            },
        }
    )
    (output_dir / "embedding_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()

