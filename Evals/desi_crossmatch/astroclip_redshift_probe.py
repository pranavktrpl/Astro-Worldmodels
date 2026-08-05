#!/usr/bin/env python
"""Redshift probe on AstroCLIP's DESI-LS x DESI EDR cross-matched sample.

This is the apples-to-apples version of Evals/redshift_regression: the same
frozen backbones and heads, but evaluated on the exact sample and 80/20 split
behind AstroCLIP's published image-encoder numbers (zero-shot kNN R2=0.79,
few-shot MLP R2=0.78), so the R2 denominators are finally comparable.

Data: parquet shards from the mhsotoudeh/astroclip HF mirror (see
download_astroclip_desi.sh) — image (152,152,3) float32 grz fluxes, spectrum,
redshift, targetid, with AstroCLIP's train/test split preserved as the
dataset's train/test splits.

Preprocessing: raw grz fluxes are mapped to RGB in [0,1] with the Legacy
Survey dr2-style arcsinh mapping (legacypipe; also used by Stein et al. and
AstroCLIP for their image-model inputs), then bicubic-resized to the probe
input size. This matches both AstroCLIP's own image pipeline and the RGB-like
domain our backbones were trained on.

The AstroCLIP train/test split is fixed, so seeds vary only the validation
carve-out (10% of train, used for hyperparameter/epoch selection) and MLP
initialization.

Run:  python astroclip_redshift_probe.py --model all
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
import torch.nn.functional as F
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
EVALS_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "astroclip"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rp = load_module(
    "redshift_probe", EVALS_DIR / "redshift_regression" / "redshift_probe.py"
)

# Legacy Survey dr2-style RGB mapping constants (legacypipe / Stein et al.
# ssl-legacysurvey / AstroCLIP `to_rgb`): per-band (output plane, flux scale).
DR2_RGB_SCALES = {"g": (2, 6.0), "r": (1, 3.4), "z": (0, 2.2)}
DR2_RGB_M = 0.03
DR2_RGB_Q = 20.0
BANDS = ("g", "r", "z")


def dr2_rgb_batch(batch: np.ndarray) -> np.ndarray:
    """Vectorized dr2-style arcsinh RGB mapping. batch: (B,H,W,3) grz fluxes."""
    intensity = np.zeros(batch.shape[:3], dtype=np.float64)
    for i, band in enumerate(BANDS):
        _, scale = DR2_RGB_SCALES[band]
        intensity += np.maximum(0.0, batch[..., i].astype(np.float64) * scale + DR2_RGB_M)
    intensity /= len(BANDS)
    stretch = np.arcsinh(DR2_RGB_Q * intensity) / np.sqrt(DR2_RGB_Q)
    intensity = np.where(intensity == 0.0, 1e-6, intensity)
    rgb = np.zeros(batch.shape, dtype=np.float32)
    for i, band in enumerate(BANDS):
        plane, scale = DR2_RGB_SCALES[band]
        rgb[..., plane] = (
            (batch[..., i].astype(np.float64) * scale + DR2_RGB_M)
            * stretch
            / intensity
        )
    return np.clip(rgb, 0.0, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*rp.MODELS, "all"], default="all")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Evaluate this checkpoint instead of the --model presets.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Results directory name for --checkpoint (default: derived from its path).",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--input-size", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_files(data_dir: Path) -> tuple[list[Path], list[Path]]:
    """Ordered (train, test) parquet shards. Global row order everywhere in
    this module is: all train shards in filename order, then all test shards."""
    train = sorted((data_dir / "data").glob("train-*.parquet"))
    test = sorted((data_dir / "data").glob("test-*.parquet"))
    if not train or not test:
        raise RuntimeError(
            f"No parquet shards under {data_dir}/data — run "
            "download_astroclip_desi.sh first."
        )
    return train, test


def read_targets(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global redshift/targetid arrays plus the AstroCLIP test-split mask."""
    train, test = split_files(data_dir)
    redshifts, targetids, is_test = [], [], []
    for flag, files in ((False, train), (True, test)):
        for path in files:
            table = pq.read_table(path, columns=["redshift", "targetid"])
            n = len(table)
            redshifts.append(
                table.column("redshift").to_numpy().astype(np.float64)
            )
            targetids.append(table.column("targetid").to_numpy())
            is_test.append(np.full(n, flag))
    return np.concatenate(redshifts), np.concatenate(targetids), np.concatenate(is_test)


def iter_image_batches(files: list[Path], batch_size: int):
    """Yield (B,H,W,3) float32 flux arrays from nested-list image columns."""
    for path in files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(
            batch_size=batch_size, columns=["image"]
        ):
            column = batch.column("image")
            # Unwrap the HF Array3D extension array if `datasets` is imported
            # in this process (its registered type has no .flatten()).
            if hasattr(column, "storage"):
                column = column.storage
            flat = column.flatten().flatten().flatten().to_numpy(
                zero_copy_only=False
            )
            count = len(column)
            side = int(round((flat.size / (count * 3)) ** 0.5))
            yield flat.reshape(count, side, side, 3).astype(np.float32)


@torch.inference_mode()
def extract_embeddings(
    backbone: torch.nn.Module,
    data_dir: Path,
    input_size: int,
    batch_size: int,
    device: torch.device,
    description: str,
) -> torch.Tensor:
    from contextlib import nullcontext

    train, test = split_files(data_dir)
    chunks = []
    progress = tqdm(desc=description, leave=False)
    for array in iter_image_batches(train + test, batch_size):
        rgb = dr2_rgb_batch(array)
        batch = torch.from_numpy(rgb).permute(0, 3, 1, 2)
        batch = batch.to(device=device, dtype=torch.float32)
        batch = F.interpolate(
            batch,
            size=(input_size, input_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with amp:
            features = backbone(batch)
        chunks.append(features.float().cpu())
        progress.update(len(array))
    progress.close()
    return torch.cat(chunks)


def cached_embeddings(
    spec: dict[str, Any], args: argparse.Namespace, device: torch.device
) -> tuple[torch.Tensor, dict[str, Any]]:
    output_dir = args.output_dir.resolve() / spec["label"]
    cache = output_dir / "embeddings.npy"
    meta_path = output_dir / "embedding_metadata.json"
    if cache.exists() and meta_path.exists() and not args.overwrite:
        print(f"cached embeddings: {cache}")
        return torch.from_numpy(np.load(cache)), json.loads(meta_path.read_text())
    backbone, meta = rp.load_backbone(spec["checkpoint"], device)
    embeddings = extract_embeddings(
        backbone,
        args.data_dir,
        args.input_size,
        args.batch_size,
        device,
        description=f"{spec['label']} embeddings",
    )
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    meta.update(
        {
            "input_size": args.input_size,
            "preprocessing": "dr2_rgb_arcsinh_to_0_1_then_bicubic_resize",
            "dr2_rgb": {"scales": DR2_RGB_SCALES, "m": DR2_RGB_M, "q": DR2_RGB_Q},
            "source": "mhsotoudeh/astroclip parquet mirror: "
            + str(args.data_dir.resolve()),
            "split": "AstroCLIP train/test splits as shipped in the dataset",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, embeddings.numpy())
    rp.atomic_json(meta_path, meta)
    return embeddings, meta


def evaluate_seed(
    train_x_all: torch.Tensor,
    train_y_all: np.ndarray,
    test_x: torch.Tensor,
    test_y: np.ndarray,
    seed: int,
    mlp_epochs: int,
    device: torch.device,
) -> dict[str, Any]:
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(train_y_all))
    val_count = max(1, int(0.1 * len(order)))
    val_idx, fit_idx = order[:val_count], order[val_count:]
    fit_x = train_x_all[torch.from_numpy(fit_idx)]
    val_x = train_x_all[torch.from_numpy(val_idx)]
    fit_y_np, val_y = train_y_all[fit_idx], train_y_all[val_idx]
    fit_y = torch.from_numpy(fit_y_np).float().to(device)

    heads: dict[str, Any] = {}

    ridge_candidates = []
    best = None
    for l2_value in rp.RIDGE_L2_GRID:
        weights, y_mean = rp.fit_ridge(fit_x, fit_y, l2_value)
        val_r2 = rp.regression_metrics(
            rp.predict_ridge(val_x, weights, y_mean), val_y
        )["r2"]
        ridge_candidates.append({"l2": l2_value, "validation_r2": val_r2})
        if best is None or val_r2 > best[1]:
            best = (l2_value, val_r2, weights, y_mean)
    heads["ridge"] = {
        "selected_l2": best[0],
        "candidates": ridge_candidates,
        "validation": rp.regression_metrics(
            rp.predict_ridge(val_x, best[2], best[3]), val_y
        ),
        "test": rp.regression_metrics(
            rp.predict_ridge(test_x, best[2], best[3]), test_y
        ),
    }

    knn_candidates = []
    best_k = None
    for k in rp.KNN_K_GRID:
        val_r2 = rp.regression_metrics(
            rp.predict_knn(fit_x, fit_y, val_x, k), val_y
        )["r2"]
        knn_candidates.append({"k": k, "validation_r2": val_r2})
        if best_k is None or val_r2 > best_k[1]:
            best_k = (k, val_r2)
    heads["knn"] = {
        "selected_k": best_k[0],
        "candidates": knn_candidates,
        "validation": rp.regression_metrics(
            rp.predict_knn(fit_x, fit_y, val_x, best_k[0]), val_y
        ),
        "test": rp.regression_metrics(
            rp.predict_knn(fit_x, fit_y, test_x, best_k[0]), test_y
        ),
    }

    mlp = rp.fit_mlp(fit_x, fit_y, val_x, val_y, mlp_epochs, seed, device)
    heads["mlp"] = {
        "validation": rp.regression_metrics(rp.predict_mlp(mlp, val_x), val_y),
        "test": rp.regression_metrics(rp.predict_mlp(mlp, test_x), test_y),
    }

    return {"seed": seed, "heads": heads}


def summarize(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for head in ("ridge", "knn", "mlp"):
        summary[head] = {}
        for split in ("validation", "test"):
            summary[head][split] = {}
            for metric in ("r2", "mae", "rmse", "nmad", "outlier_fraction"):
                values = [r["heads"][head][split][metric] for r in repeats]
                summary[head][split][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values,
                }
    return summary


def evaluate_model(
    spec: dict[str, Any], args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    started = time.time()
    output_dir = args.output_dir.resolve() / spec["label"]
    embeddings, embed_meta = cached_embeddings(spec, args, device)
    redshifts, targetids, is_test = read_targets(args.data_dir)
    if len(redshifts) != len(embeddings):
        raise RuntimeError(
            f"{len(redshifts)} targets but {len(embeddings)} embeddings"
        )

    train_idx = np.flatnonzero(~is_test)
    test_idx = np.flatnonzero(is_test)
    mean = embeddings[torch.from_numpy(train_idx)].mean(dim=0)
    std = embeddings[torch.from_numpy(train_idx)].std(dim=0).clamp_min(1e-6)
    train_x = ((embeddings[torch.from_numpy(train_idx)] - mean) / std).to(device)
    test_x = ((embeddings[torch.from_numpy(test_idx)] - mean) / std).to(device)
    train_y, test_y = redshifts[train_idx], redshifts[test_idx]

    repeats = []
    for seed in args.seeds:
        repeat = evaluate_seed(
            train_x, train_y, test_x, test_y, seed, args.mlp_epochs, device
        )
        repeats.append(repeat)
        line = ", ".join(
            f"{head} test R2={repeat['heads'][head]['test']['r2']:.4f}"
            for head in ("ridge", "knn", "mlp")
        )
        print(f"{spec['label']} seed={seed}: {line}")

    result = {
        "label": spec["label"],
        "sample": "AstroCLIP DESI-LS x DESI EDR cross-match",
        "embedding_metadata": embed_meta,
        "target_stats": {
            "num_train": int(len(train_y)),
            "num_test": int(len(test_y)),
            "redshift_min": float(redshifts.min()),
            "redshift_max": float(redshifts.max()),
            "redshift_median": float(np.median(redshifts)),
        },
        "published_reference": {
            "AstroCLIP Image, zero-shot kNN": 0.79,
            "AstroCLIP Image, few-shot MLP": 0.78,
        },
        "seeds": args.seeds,
        "ridge_l2_grid": rp.RIDGE_L2_GRID,
        "knn_k_grid": rp.KNN_K_GRID,
        "repeats": repeats,
        "summary": summarize(repeats),
        "elapsed_seconds": time.time() - started,
    }
    rp.atomic_json(output_dir / "metrics.json", result)
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.checkpoint is not None:
        label = args.label or "_".join(
            [args.checkpoint.resolve().parent.name, args.checkpoint.stem]
        )
        specs = [{"label": label, "checkpoint": args.checkpoint}]
    else:
        keys = list(rp.MODELS) if args.model == "all" else [args.model]
        specs = [rp.MODELS[key] for key in keys]
    for spec in specs:
        result = evaluate_model(spec, args, device)
        for head in ("ridge", "knn", "mlp"):
            stats = result["summary"][head]["test"]["r2"]
            print(
                f"{result['label']} {head}: test R2 "
                f"{stats['mean']:.4f} ± {stats['std']:.4f} "
                f"(AstroCLIP image kNN 0.79 / MLP 0.78)"
            )


if __name__ == "__main__":
    main()
