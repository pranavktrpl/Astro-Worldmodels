#!/usr/bin/env python
"""Redshift probe for the frozen spectra backbone on the AstroCLIP cross-match.

Embeds the cross-match spectra (7781 flux values, the same sample and 80/20
split as the image probes) with a frozen SpectrumTransformerEncoder and runs
the standard ridge / zero-shot kNN / MLP heads, so image and spectra
backbones are measured on identical data with identical protocols.
Reference: AstroCLIP's spectrum encoder reaches test R2 = 0.98 here.

Inference preprocessing matches training exactly: raw flux -> drop the final
value -> 389 ordered patches of 20 -> all-real mask (no crops). The encoder
is rebuilt from the checkpoint's saved cfg, so architecture always matches.

Two modes:

  --scan DIR         ridge-only validation R2 for every step_*.pt in DIR on a
                     subsample (checkpoint selection without touching test)
  --checkpoint PATH  full battery on one checkpoint, results under
                     results/<label>/metrics.json

Run scan first, pick the best validation step, then run --checkpoint once.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
EVALS_DIR = SCRIPT_DIR.parent
REPO_ROOT = EVALS_DIR.parent
DEFAULT_DATA_DIR = EVALS_DIR / "desi_crossmatch" / "data" / "astroclip"
DEFAULT_FAMILY = REPO_ROOT / "checkpoints" / "SPECTRA_run4_bs16_2806_Epoch5_utbd_desi"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


acp = load_module(
    "astroclip_redshift_probe",
    EVALS_DIR / "desi_crossmatch" / "astroclip_redshift_probe.py",
)
rp = acp.rp  # redshift_probe: heads, metrics, atomic_json
train_spectra = load_module("train_spectra", REPO_ROOT / "train-spectra.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument(
        "--scan",
        type=Path,
        nargs="?",
        const=DEFAULT_FAMILY,
        default=None,
        help="Rank every step_*.pt in this directory by ridge validation R2 "
        f"(default dir: {DEFAULT_FAMILY.name}).",
    )
    parser.add_argument("--scan-subsample", type=int, default=20000)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_encoder(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = dict(checkpoint.get("cfg") or {})
    encoder = train_spectra.SpectrumTransformerEncoder(
        proj_dim=int(cfg.get("proj_dim", 64)),
        patch_size=int(cfg.get("spectra_patch_size", 20)),
        num_patches=int(cfg.get("spectra_num_patches", 389)),
        embed_dim=int(cfg.get("spectra_embed_dim", 768)),
        depth=int(cfg.get("spectra_depth", 12)),
        num_heads=int(cfg.get("spectra_num_heads", 12)),
        mlp_ratio=float(cfg.get("spectra_mlp_ratio", 4.0)),
        dropout=float(cfg.get("spectra_dropout", 0.0)),
        pooling=str(cfg.get("spectra_pooling", "cls")),
    )
    encoder.load_state_dict(checkpoint["model"])
    # run5+ checkpoints carry their per-spectrum normalization mode in cfg;
    # attach it so every embedding path preprocesses exactly like training.
    encoder.flux_normalize = str(cfg.get("spectra_normalize", "none"))
    meta = {
        "checkpoint": str(checkpoint_path.resolve()),
        "global_step": int(checkpoint.get("global_step", -1)),
        "model_name": str(cfg.get("model_name", "spectrum_transformer")),
        "embedding_dim": int(encoder.embed_dim),
        "pooling": encoder.pooling,
        "flux_normalize": encoder.flux_normalize,
    }
    encoder.to(device).eval()
    return encoder, meta


def iter_spectrum_batches(files: list[Path], batch_size: int):
    """Yield (B, L) float32 spectra from nested-list spectrum columns."""
    for path in files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(
            batch_size=batch_size, columns=["spectrum"]
        ):
            column = batch.column("spectrum")
            if hasattr(column, "storage"):
                column = column.storage
            flat = column.flatten().flatten().to_numpy(zero_copy_only=False)
            count = len(column)
            yield flat.reshape(count, -1).astype(np.float32)


@torch.inference_mode()
def embed_spectra(
    encoder: torch.nn.Module,
    batches,
    device: torch.device,
    description: str,
    total: int | None = None,
) -> torch.Tensor:
    from contextlib import nullcontext

    patch_size = encoder.patch_size
    num_patches = encoder.num_patches
    usable = num_patches * patch_size
    normalize = getattr(encoder, "flux_normalize", "none")
    chunks = []
    progress = tqdm(desc=description, total=total, leave=False)
    for array in batches:
        # Normalize over the full spectrum before dropping the tail, exactly
        # as the training transform does.
        spectra = train_spectra.normalize_flux(torch.from_numpy(array), normalize)
        crops = spectra[:, :usable].reshape(-1, 1, num_patches, patch_size).to(device)
        masks = torch.ones(
            crops.shape[0], 1, num_patches, dtype=torch.float32, device=device
        )
        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with amp:
            emb, _ = encoder._encode_views(crops, masks)
        chunks.append(emb[0].float().cpu())
        progress.update(crops.shape[0])
    progress.close()
    return torch.cat(chunks)


def cached_embeddings(
    label: str,
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    output_dir = args.output_dir.resolve() / label
    cache = output_dir / "embeddings.npy"
    meta_path = output_dir / "embedding_metadata.json"
    if cache.exists() and meta_path.exists() and not args.overwrite:
        print(f"cached embeddings: {cache}")
        return torch.from_numpy(np.load(cache)), json.loads(meta_path.read_text())
    encoder, meta = load_encoder(checkpoint_path, device)
    train, test = acp.split_files(args.data_dir)
    embeddings = embed_spectra(
        encoder,
        iter_spectrum_batches(train + test, args.batch_size),
        device,
        description=f"{label} spectra embeddings",
    )
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    meta.update(
        {
            "preprocessing": "raw_flux_full_patchify_no_crop",
            "source": str(args.data_dir.resolve()),
            "split": "AstroCLIP train/test splits as shipped in the dataset",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, embeddings.numpy())
    rp.atomic_json(meta_path, meta)
    return embeddings, meta


def scan_family(args: argparse.Namespace, device: torch.device) -> None:
    checkpoints = sorted(
        args.scan.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1])
    )
    if not checkpoints:
        raise SystemExit(f"No step_*.pt under {args.scan}")
    redshifts, _, is_test = acp.read_targets(args.data_dir)
    train_idx = np.flatnonzero(~is_test)
    rng = np.random.RandomState(42)
    subsample = rng.permutation(train_idx)[: args.scan_subsample]
    val_count = max(1, int(0.1 * len(subsample)))
    val_idx, fit_idx = subsample[:val_count], subsample[val_count:]
    train_files, test_files = acp.split_files(args.data_dir)

    # row-aligned read of only the subsampled train spectra
    keep = np.zeros(len(redshifts), dtype=bool)
    keep[subsample] = True

    print(f"scanning {len(checkpoints)} checkpoints on {len(subsample)} train spectra")
    results = []
    for checkpoint_path in checkpoints:
        encoder, meta = load_encoder(checkpoint_path, device)

        def kept_batches():
            offset = 0
            for array in iter_spectrum_batches(train_files, args.batch_size):
                mask = keep[offset : offset + len(array)]
                offset += len(array)
                if mask.any():
                    yield array[mask]

        embeddings = embed_spectra(
            encoder,
            kept_batches(),
            device,
            description=checkpoint_path.stem,
            total=int(keep.sum()),
        )
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # map global indices to positions within the kept subset
        kept_order = np.flatnonzero(keep)
        position = {int(g): i for i, g in enumerate(kept_order)}
        fit_pos = np.array([position[int(g)] for g in fit_idx])
        val_pos = np.array([position[int(g)] for g in val_idx])
        mean = embeddings[torch.from_numpy(fit_pos)].mean(dim=0)
        std = embeddings[torch.from_numpy(fit_pos)].std(dim=0).clamp_min(1e-6)
        fit_x = ((embeddings[torch.from_numpy(fit_pos)] - mean) / std).to(device)
        val_x = ((embeddings[torch.from_numpy(val_pos)] - mean) / std).to(device)
        fit_y = torch.from_numpy(redshifts[fit_idx]).float().to(device)

        best_r2 = -np.inf
        for l2_value in rp.RIDGE_L2_GRID:
            weights, y_mean = rp.fit_ridge(fit_x, fit_y, l2_value)
            r2 = rp.regression_metrics(
                rp.predict_ridge(val_x, weights, y_mean), redshifts[val_idx]
            )["r2"]
            best_r2 = max(best_r2, r2)
        results.append((checkpoint_path.name, meta["global_step"], best_r2))
        print(f"  {checkpoint_path.name}: ridge validation R2 = {best_r2:.4f}")

    results.sort(key=lambda row: row[2], reverse=True)
    print("\nranking (validation R2):")
    for name, step, r2 in results:
        print(f"  {r2:.4f}  {name}")
    best = results[0]
    print(
        f"\nbest: {best[0]} — run the full battery with:\n"
        f"  python {Path(__file__).name} --checkpoint {args.scan / best[0]}"
    )
    rp.atomic_json(
        args.output_dir.resolve() / "scan.json",
        {
            "family": str(args.scan.resolve()),
            "subsample": int(len(subsample)),
            "ranking": [
                {"checkpoint": n, "global_step": s, "validation_r2": r}
                for n, s, r in results
            ],
        },
    )


def evaluate_checkpoint(args: argparse.Namespace, device: torch.device) -> None:
    label = args.label or "_".join(
        [args.checkpoint.resolve().parent.name, args.checkpoint.stem]
    )
    started = time.time()
    embeddings, embed_meta = cached_embeddings(label, args.checkpoint, args, device)
    redshifts, _, is_test = acp.read_targets(args.data_dir)
    if len(redshifts) != len(embeddings):
        raise RuntimeError(f"{len(redshifts)} targets, {len(embeddings)} embeddings")

    train_idx = np.flatnonzero(~is_test)
    test_idx = np.flatnonzero(is_test)
    mean = embeddings[torch.from_numpy(train_idx)].mean(dim=0)
    std = embeddings[torch.from_numpy(train_idx)].std(dim=0).clamp_min(1e-6)
    train_x = ((embeddings[torch.from_numpy(train_idx)] - mean) / std).to(device)
    test_x = ((embeddings[torch.from_numpy(test_idx)] - mean) / std).to(device)

    repeats = [
        acp.evaluate_seed(
            train_x,
            redshifts[train_idx],
            test_x,
            redshifts[test_idx],
            seed,
            args.mlp_epochs,
            device,
        )
        for seed in args.seeds
    ]
    result = {
        "label": label,
        "modality": "spectra",
        "sample": "AstroCLIP DESI-LS x DESI EDR cross-match",
        "embedding_metadata": embed_meta,
        "published_reference": {"AstroCLIP Spectrum, few-shot MLP": 0.98},
        "seeds": args.seeds,
        "repeats": repeats,
        "summary": acp.summarize(repeats),
        "elapsed_seconds": time.time() - started,
    }
    rp.atomic_json(args.output_dir.resolve() / label / "metrics.json", result)
    for head in ("ridge", "knn", "mlp"):
        stats = result["summary"][head]["test"]["r2"]
        print(
            f"{label} {head}: test R2 {stats['mean']:.4f} ± {stats['std']:.4f} "
            "(AstroCLIP spectrum 0.98)"
        )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.scan is not None:
        scan_family(args, device)
    elif args.checkpoint is not None:
        evaluate_checkpoint(args, device)
    else:
        raise SystemExit("Pass --scan [family-dir] or --checkpoint <path>.")


if __name__ == "__main__":
    main()
