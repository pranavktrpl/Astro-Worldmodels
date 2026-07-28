#!/usr/bin/env python3
"""Redshift regression probes on frozen Astro-Worldmodels image backbones.

Targets come from the `redshift` column of the local Galaxy10 DECaLS HDF5.
Three heads are evaluated per checkpoint on the same frozen embeddings:

- ridge:  deterministic closed-form linear ridge regression;
- knn:    k-nearest-neighbour regression (AstroCLIP zero-shot style);
- mlp:    a small 2-hidden-layer MLP head (AION-style).

Protocol mirrors Evals/galaxy10_checkpoint_evolution: bicubic resize to
140 x 140, RGB floats in [0, 1] with no ImageNet normalization, frozen
backbone, quantile-stratified 80/10/10 splits repeated over seeds 42/43/44,
feature standardization from train statistics only, hyperparameters selected
on validation R^2 only, test metrics reported after selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import timm  # noqa: E402

MODELS = {
    "large": {
        "label": "astro_vit_large_step_52000",
        "checkpoint": REPO_ROOT
        / "checkpoints/VitLargePatch14_OfficialTrain5_Epoch5_2504/step_52000.pt",
    },
    "small": {
        "label": "astro_vit_small_step_21000",
        "checkpoint": REPO_ROOT / "checkpoints/VitSmallPatch14_2204/step_21000.pt",
    },
}
DEFAULT_H5 = (
    REPO_ROOT / "Evals" / "DeCals_linearProbing" / "galaxy10" / "Galaxy10_DECals.h5"
)
SPLIT_SEEDS = [42, 43, 44]
STRATIFY_BINS = 10
RIDGE_L2_GRID = [1e-4, 1e-2, 1.0, 1e2, 1e4]
KNN_K_GRID = [4, 16, 64]
OUTLIER_THRESHOLD = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--galaxy10-h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--input-size", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def torch_load(path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": False}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def load_backbone(
    checkpoint_path: Path, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
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
        "model_name": model_name,
        "embedding_dim": int(getattr(backbone, "num_features", -1)),
    }
    del checkpoint, state
    backbone.to(device).eval()
    return backbone, meta


def load_targets(h5_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with h5py.File(h5_path, "r") as handle:
        if "redshift" not in handle:
            raise KeyError(
                f"{h5_path} has no 'redshift' dataset; keys: {sorted(handle.keys())}"
            )
        redshift = np.asarray(handle["redshift"][:], dtype=np.float64)
    keep = np.isfinite(redshift) & (redshift > 0.0)
    stats = {
        "total_examples": int(len(redshift)),
        "kept_examples": int(keep.sum()),
        "dropped_examples": int((~keep).sum()),
        "keep_rule": "finite and redshift > 0",
        "redshift_min": float(redshift[keep].min()),
        "redshift_max": float(redshift[keep].max()),
        "redshift_median": float(np.median(redshift[keep])),
    }
    return redshift, keep, stats


@torch.inference_mode()
def extract_embeddings(
    backbone: nn.Module,
    h5_path: Path,
    input_size: int,
    batch_size: int,
    device: torch.device,
    description: str,
) -> torch.Tensor:
    embeddings: torch.Tensor | None = None
    with h5py.File(h5_path, "r") as handle:
        images = handle["images"]
        starts = range(0, len(images), batch_size)
        for start in tqdm(
            starts,
            total=(len(images) + batch_size - 1) // batch_size,
            desc=description,
            leave=False,
        ):
            stop = min(start + batch_size, len(images))
            array = np.asarray(images[start:stop], dtype=np.uint8)
            batch = torch.from_numpy(array).permute(0, 3, 1, 2)
            batch = batch.to(device=device, dtype=torch.float32).div_(255.0)
            batch = F.interpolate(
                batch,
                size=(input_size, input_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            amp_context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                batch_embeddings = backbone(batch)
            batch_embeddings = batch_embeddings.float().cpu()
            if embeddings is None:
                embeddings = torch.empty(
                    (len(images), batch_embeddings.shape[1]), dtype=torch.float32
                )
            embeddings[start:stop] = batch_embeddings
    if embeddings is None:
        raise RuntimeError("Embedding extraction produced no examples.")
    return embeddings


def cached_embeddings(
    spec: dict[str, Any], args: argparse.Namespace, device: torch.device
) -> tuple[torch.Tensor, dict[str, Any]]:
    output_dir = args.output_dir.resolve() / spec["label"]
    cache = output_dir / "embeddings.npy"
    meta_path = output_dir / "embedding_metadata.json"
    if cache.exists() and meta_path.exists() and not args.overwrite:
        print(f"cached embeddings: {cache}")
        return (
            torch.from_numpy(np.load(cache)),
            json.loads(meta_path.read_text()),
        )
    backbone, meta = load_backbone(spec["checkpoint"], device)
    embeddings = extract_embeddings(
        backbone=backbone,
        h5_path=args.galaxy10_h5,
        input_size=args.input_size,
        batch_size=args.batch_size,
        device=device,
        description=f"{spec['label']} embeddings",
    )
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    meta.update(
        {
            "input_size": args.input_size,
            "preprocessing": "bicubic_resize_to_tensor_0_1_no_normalization",
            "galaxy10_h5": str(args.galaxy10_h5.resolve()),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, embeddings.numpy())
    atomic_json(meta_path, meta)
    return embeddings, meta


def quantile_stratified_split(
    targets: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """80/10/10 split stratified over redshift quantile bins."""
    edges = np.quantile(targets, np.linspace(0, 1, STRATIFY_BINS + 1)[1:-1])
    bins = np.searchsorted(edges, targets)
    rng = np.random.RandomState(seed)
    train_parts, val_parts, test_parts = [], [], []
    for bin_id in np.unique(bins):
        indices = np.flatnonzero(bins == bin_id)
        rng.shuffle(indices)
        train_end = int(np.floor(0.8 * len(indices)))
        val_end = train_end + int(np.floor(0.1 * len(indices)))
        train_parts.append(indices[:train_end])
        val_parts.append(indices[train_end:val_end])
        test_parts.append(indices[val_end:])
    train = np.concatenate(train_parts)
    validation = np.concatenate(val_parts)
    test = np.concatenate(test_parts)
    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test


def regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    residuals = predictions - targets
    scaled = residuals / (1.0 + targets)
    total_variance = float(((targets - targets.mean()) ** 2).sum())
    return {
        "r2": float(1.0 - (residuals**2).sum() / max(total_variance, 1e-12)),
        "mae": float(np.abs(residuals).mean()),
        "rmse": float(np.sqrt((residuals**2).mean())),
        "nmad": float(1.4826 * np.median(np.abs(scaled - np.median(scaled)))),
        "outlier_fraction": float((np.abs(scaled) > OUTLIER_THRESHOLD).mean()),
        "num_examples": int(len(targets)),
    }


def fit_ridge(
    train_x: torch.Tensor, train_y: torch.Tensor, l2_value: float
) -> tuple[torch.Tensor, float]:
    """Closed-form ridge on standardized features with centered targets."""
    y_mean = train_y.mean()
    centered = (train_y - y_mean).double()
    x = train_x.double()
    gram = x.T @ x + l2_value * torch.eye(x.shape[1], dtype=torch.float64, device=x.device)
    weights = torch.linalg.solve(gram, x.T @ centered)
    return weights, float(y_mean)


def predict_ridge(features: torch.Tensor, weights: torch.Tensor, y_mean: float) -> np.ndarray:
    return (features.double() @ weights + y_mean).cpu().numpy()


def predict_knn(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    query_x: torch.Tensor,
    k: int,
    chunk: int = 2048,
) -> np.ndarray:
    predictions = []
    for part in query_x.split(chunk):
        distances = torch.cdist(part, train_x)
        _, indices = distances.topk(k, dim=1, largest=False)
        predictions.append(train_y[indices].mean(dim=1).cpu())
    return torch.cat(predictions).numpy()


def fit_mlp(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: np.ndarray,
    epochs: int,
    seed: int,
    device: torch.device,
) -> nn.Module:
    set_seed(seed)
    model = nn.Sequential(
        nn.Linear(train_x.shape[1], 256),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(256, 256),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(256, 1),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    best_state = None
    best_val_r2 = -np.inf
    batch = 512
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), batch):
            index = order[start : start + batch].to(train_x.device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(train_x[index]).squeeze(-1), train_y[index])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_pred = model(val_x).squeeze(-1).cpu().numpy()
        val_r2 = regression_metrics(val_pred, val_y)["r2"]
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.inference_mode()
def predict_mlp(model: nn.Module, features: torch.Tensor) -> np.ndarray:
    return model(features).squeeze(-1).cpu().numpy()


def evaluate_split_seed(
    embeddings: torch.Tensor,
    targets: np.ndarray,
    split_seed: int,
    mlp_epochs: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train_idx, val_idx, test_idx = quantile_stratified_split(targets, split_seed)
    train_features = embeddings[torch.from_numpy(train_idx)]
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0).clamp_min(1e-6)

    def normalized(indices: np.ndarray) -> torch.Tensor:
        return ((embeddings[torch.from_numpy(indices)] - mean) / std).to(device)

    train_x, val_x, test_x = normalized(train_idx), normalized(val_idx), normalized(test_idx)
    train_y = torch.from_numpy(targets[train_idx]).float().to(device)
    val_y, test_y = targets[val_idx], targets[test_idx]

    heads: dict[str, Any] = {}
    test_predictions: dict[str, np.ndarray] = {"targets": test_y}

    ridge_candidates = []
    best = None
    for l2_value in RIDGE_L2_GRID:
        weights, y_mean = fit_ridge(train_x, train_y, l2_value)
        val_r2 = regression_metrics(predict_ridge(val_x, weights, y_mean), val_y)["r2"]
        ridge_candidates.append({"l2": l2_value, "validation_r2": val_r2})
        if best is None or val_r2 > best[0]:
            best = (val_r2, l2_value, weights, y_mean)
    heads["ridge"] = {
        "selected_l2": best[1],
        "candidates": ridge_candidates,
        "validation": regression_metrics(predict_ridge(val_x, best[2], best[3]), val_y),
        "test": regression_metrics(predict_ridge(test_x, best[2], best[3]), test_y),
    }
    test_predictions["ridge"] = predict_ridge(test_x, best[2], best[3])

    knn_candidates = []
    best_k = None
    for k in KNN_K_GRID:
        val_r2 = regression_metrics(predict_knn(train_x, train_y, val_x, k), val_y)["r2"]
        knn_candidates.append({"k": k, "validation_r2": val_r2})
        if best_k is None or val_r2 > best_k[0]:
            best_k = (val_r2, k)
    heads["knn"] = {
        "selected_k": best_k[1],
        "candidates": knn_candidates,
        "validation": regression_metrics(
            predict_knn(train_x, train_y, val_x, best_k[1]), val_y
        ),
        "test": regression_metrics(
            predict_knn(train_x, train_y, test_x, best_k[1]), test_y
        ),
    }
    test_predictions["knn"] = predict_knn(train_x, train_y, test_x, best_k[1])

    mlp = fit_mlp(train_x, train_y, val_x, val_y, mlp_epochs, split_seed, device)
    heads["mlp"] = {
        "epochs": mlp_epochs,
        "validation": regression_metrics(predict_mlp(mlp, val_x), val_y),
        "test": regression_metrics(predict_mlp(mlp, test_x), test_y),
    }
    test_predictions["mlp"] = predict_mlp(mlp, test_x)

    result = {
        "split_seed": split_seed,
        "split_sizes": {
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        "heads": heads,
    }
    return result, test_predictions


def summarize_repeats(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for head in ("ridge", "knn", "mlp"):
        summary[head] = {}
        for split in ("validation", "test"):
            summary[head][split] = {}
            for metric in ("r2", "mae", "rmse", "nmad", "outlier_fraction"):
                values = np.asarray(
                    [repeat["heads"][head][split][metric] for repeat in repeats]
                )
                summary[head][split][metric] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "values": values.tolist(),
                }
    return summary


def plot_predictions(
    output_dir: Path, label: str, predictions: dict[str, np.ndarray]
) -> None:
    heads = [head for head in ("ridge", "knn", "mlp") if head in predictions]
    targets = predictions["targets"]
    fig, axes = plt.subplots(1, len(heads), figsize=(5.4 * len(heads), 5), squeeze=False)
    limit = float(np.quantile(targets, 0.995)) * 1.05
    for ax, head in zip(axes[0], heads):
        ax.hexbin(
            targets,
            predictions[head],
            gridsize=60,
            extent=(0, limit, 0, limit),
            cmap="viridis",
            mincnt=1,
        )
        ax.plot([0, limit], [0, limit], color="#d62728", linewidth=1, linestyle="--")
        metrics = regression_metrics(predictions[head], targets)
        ax.set_title(
            f"{head}: R2={metrics['r2']:.3f}  NMAD={metrics['nmad']:.4f}"
        )
        ax.set_xlabel("Spectroscopic redshift")
        ax.set_ylabel("Predicted redshift")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
    fig.suptitle(f"Galaxy10 redshift regression, frozen {label} (seed 42 test split)")
    fig.tight_layout()
    fig.savefig(output_dir / "predicted_vs_true.png", dpi=200)
    plt.close(fig)


def write_summary_csv(output_dir: Path, all_results: list[dict[str, Any]]) -> None:
    rows = []
    for result in all_results:
        for head in ("ridge", "knn", "mlp"):
            row = {"model": result["label"], "head": head}
            for metric in ("r2", "mae", "rmse", "nmad", "outlier_fraction"):
                stats = result["summary"][head]["test"][metric]
                row[f"test_{metric}_mean"] = stats["mean"]
                row[f"test_{metric}_std"] = stats["std"]
            rows.append(row)
    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_model(
    key: str, args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    spec = MODELS[key]
    started = time.time()
    output_dir = args.output_dir.resolve() / spec["label"]
    embeddings, embed_meta = cached_embeddings(spec, args, device)
    redshift, keep, target_stats = load_targets(args.galaxy10_h5)
    embeddings = embeddings[torch.from_numpy(np.flatnonzero(keep))]
    targets = redshift[keep]

    repeats = []
    seed42_predictions: dict[str, np.ndarray] | None = None
    for split_seed in SPLIT_SEEDS:
        repeat, predictions = evaluate_split_seed(
            embeddings=embeddings,
            targets=targets,
            split_seed=split_seed,
            mlp_epochs=args.mlp_epochs,
            device=device,
        )
        repeats.append(repeat)
        if split_seed == SPLIT_SEEDS[0]:
            seed42_predictions = predictions
        summary_line = ", ".join(
            f"{head} test R2={repeat['heads'][head]['test']['r2']:.4f}"
            for head in ("ridge", "knn", "mlp")
        )
        print(f"{spec['label']} seed={split_seed}: {summary_line}")

    result = {
        "label": spec["label"],
        "embedding_metadata": embed_meta,
        "target_stats": target_stats,
        "split_seeds": SPLIT_SEEDS,
        "stratify_bins": STRATIFY_BINS,
        "ridge_l2_grid": RIDGE_L2_GRID,
        "knn_k_grid": KNN_K_GRID,
        "outlier_threshold": OUTLIER_THRESHOLD,
        "repeats": repeats,
        "summary": summarize_repeats(repeats),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(output_dir / "metrics.json", result)
    plot_predictions(output_dir, spec["label"], seed42_predictions)
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    set_seed(SPLIT_SEEDS[0])
    keys = list(MODELS) if args.model == "all" else [args.model]
    all_results = [evaluate_model(key, args, device) for key in keys]
    write_summary_csv(args.output_dir.resolve(), all_results)
    for result in all_results:
        for head in ("ridge", "knn", "mlp"):
            test = result["summary"][head]["test"]
            print(
                f"{result['label']} {head}: "
                f"test R2={test['r2']['mean']:.4f} +/- {test['r2']['std']:.4f}, "
                f"NMAD={test['nmad']['mean']:.4f}, "
                f"outliers={test['outlier_fraction']['mean']:.4f}"
            )
    print(f"Results written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
