#!/usr/bin/env python3
"""Controlled Galaxy10 linear probes across LeJEPA checkpoint evolution."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

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
from models.resnet9 import Resnet9  # noqa: E402


CLASS_NAMES = [
    "disturbed",
    "merging",
    "round_smooth",
    "in_between_round_smooth",
    "cigar_shaped_smooth",
    "barred_spiral",
    "unbarred_tight_spiral",
    "unbarred_loose_spiral",
    "edge_on_without_bulge",
    "edge_on_with_bulge",
]
IGNORED_FAMILIES = {"FirstTrain_VitSmallPatch14_2104"}
DISPLAY_NAMES = {
    "VitLargePatch14_OfficialTrain5_Epoch5_2504": "ViT-L/14",
    "VitSmallPatch14_2204": "ViT-S/14",
    "checkpoints-test": "ResNet9",
}
STEP_RE = re.compile(r"^step_(\d+)\.pt$")


@dataclass(frozen=True)
class CheckpointRef:
    family: str
    path: Path
    step: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create checkpoint manifest.")
    add_common_paths(prepare)

    worker = subparsers.add_parser("worker", help="Evaluate a manifest shard.")
    add_common_paths(worker)
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--worker-count", type=int, required=True)
    worker.add_argument("--device", default="cuda")
    worker.add_argument("--embedding-batch-size", type=int, default=256)
    worker.add_argument("--input-size", type=int, default=140)
    worker.add_argument("--split-seeds", type=int, nargs="+", default=[42, 43, 44])
    worker.add_argument(
        "--l2-values",
        type=float,
        nargs="+",
        default=[1e-6, 1e-4, 1e-3, 1e-2],
    )
    worker.add_argument("--lbfgs-iterations", type=int, default=60)
    worker.add_argument("--overwrite", action="store_true")

    plot = subparsers.add_parser("plot", help="Aggregate metrics and make plots.")
    add_common_paths(plot)

    return parser.parse_args()


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=REPO_ROOT / "checkpoints",
    )
    parser.add_argument(
        "--galaxy10-h5",
        type=Path,
        default=REPO_ROOT
        / "Evals"
        / "DeCals_linearProbing"
        / "galaxy10"
        / "Galaxy10_DECals.h5",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load(path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": False}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def checkpoint_priority(path: Path, step: int) -> tuple[int, str]:
    if path.name == f"step_{step}.pt":
        return (0, path.name)
    if path.name == "complete.pt":
        return (1, path.name)
    if path.name.startswith("last_epoch_"):
        return (2, path.name)
    return (3, path.name)


def discover_checkpoints(root: Path) -> list[CheckpointRef]:
    refs: list[CheckpointRef] = []
    for family_dir in sorted(root.resolve().iterdir()):
        if (
            not family_dir.is_dir()
            or family_dir.name in IGNORED_FAMILIES
            or not any(family_dir.glob("*.pt"))
        ):
            continue
        family_refs: list[CheckpointRef] = []
        for path in sorted(family_dir.glob("*.pt")):
            match = STEP_RE.match(path.name)
            if match:
                step = int(match.group(1))
            else:
                try:
                    checkpoint = torch_load(path)
                    step = int(checkpoint.get("global_step", -1))
                    del checkpoint
                except Exception as exc:
                    print(f"Skipping unreadable {path}: {exc}", file=sys.stderr)
                    continue
            if step >= 0:
                family_refs.append(
                    CheckpointRef(family=family_dir.name, path=path, step=step)
                )
        by_step: dict[int, CheckpointRef] = {}
        for ref in family_refs:
            current = by_step.get(ref.step)
            if current is None or checkpoint_priority(
                ref.path, ref.step
            ) < checkpoint_priority(current.path, current.step):
                by_step[ref.step] = ref
        refs.extend(by_step.values())
    return sorted(refs, key=lambda ref: (ref.family, ref.step))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    refs = discover_checkpoints(args.checkpoint_root)
    manifest = [
        {"family": ref.family, "path": str(ref.path), "step": ref.step}
        for ref in refs
    ]
    with h5py.File(args.galaxy10_h5, "r") as handle:
        labels = np.asarray(handle["ans"][:], dtype=np.int64)
        image_shape = list(handle["images"].shape)
    metadata = {
        "galaxy10_h5": str(args.galaxy10_h5.resolve()),
        "num_examples": int(len(labels)),
        "image_shape": image_shape,
        "class_names": CLASS_NAMES,
        "class_counts": np.bincount(labels, minlength=len(CLASS_NAMES)).tolist(),
        "ignored_families": sorted(IGNORED_FAMILIES),
        "checkpoint_count": len(manifest),
    }
    atomic_json(output_dir / "manifest.json", manifest)
    atomic_json(output_dir / "dataset_metadata.json", metadata)
    print(f"Prepared {len(manifest)} checkpoints in {output_dir / 'manifest.json'}")


def architecture_name(state: dict[str, torch.Tensor], cfg: dict[str, Any]) -> str:
    if any(key.startswith("backbone.conv.") for key in state):
        return "resnet9"
    if cfg.get("model_name"):
        return "timm"
    raise ValueError("Unable to infer checkpoint architecture.")


def build_backbone(
    checkpoint: dict[str, Any], device: torch.device
) -> tuple[nn.Module, str, str]:
    state = checkpoint["model"]
    cfg = dict(checkpoint.get("cfg") or {})
    architecture = architecture_name(state, cfg)
    if architecture == "resnet9":
        backbone: nn.Module = Resnet9(num_classes=1, num_channels=3)
        model_name = "resnet9"
    else:
        model_name = str(cfg["model_name"])
        backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )
    backbone_state = {
        key.removeprefix("backbone."): value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    backbone.load_state_dict(backbone_state, strict=True)
    backbone.to(device).eval()
    return backbone, architecture, model_name


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


def read_labels(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["ans"][:], dtype=np.int64)


def stratified_split(
    labels: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    train_parts, val_parts, test_parts = [], [], []
    for class_id in range(len(CLASS_NAMES)):
        indices = np.flatnonzero(labels == class_id)
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


def confusion_matrix(
    labels: torch.Tensor, predictions: torch.Tensor, classes: int
) -> torch.Tensor:
    flat = labels.to(torch.int64) * classes + predictions.to(torch.int64)
    return torch.bincount(flat, minlength=classes * classes).reshape(classes, classes)


def metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    predictions = logits.argmax(dim=1)
    cm = confusion_matrix(labels.cpu(), predictions.cpu(), len(CLASS_NAMES))
    cm_float = cm.float()
    support = cm_float.sum(dim=1)
    predicted = cm_float.sum(dim=0)
    true_positive = cm_float.diag()
    recall = true_positive / support.clamp_min(1)
    precision = true_positive / predicted.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    accuracy = true_positive.sum() / cm_float.sum().clamp_min(1)
    return {
        "accuracy": float(accuracy.item()),
        "macro_f1": float(f1.mean().item()),
        "balanced_accuracy": float(recall.mean().item()),
        "per_class_f1": f1.tolist(),
        "support": support.to(torch.int64).tolist(),
        "confusion_matrix": cm.tolist(),
    }


def standardize(
    features: torch.Tensor, train_indices: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    train_features = features[torch.from_numpy(train_indices)]
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0).clamp_min(1e-6)
    return mean, std


def fit_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    l2_value: float,
    iterations: int,
    device: torch.device,
) -> nn.Linear:
    model = nn.Linear(train_x.shape[1], len(CLASS_NAMES)).to(device)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    counts = torch.bincount(train_y, minlength=len(CLASS_NAMES)).float()
    class_weights = train_y.numel() / (
        len(CLASS_NAMES) * counts.clamp_min(1)
    )
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=iterations,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y, weight=class_weights)
        loss = loss + 0.5 * l2_value * model.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return model


@torch.inference_mode()
def predict(model: nn.Module, features: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    return torch.cat([model(part) for part in features.split(chunk)], dim=0).cpu()


def evaluate_split_seed(
    embeddings: torch.Tensor,
    labels_np: np.ndarray,
    split_seed: int,
    l2_values: list[float],
    lbfgs_iterations: int,
    device: torch.device,
) -> dict[str, Any]:
    train_indices, val_indices, test_indices = stratified_split(labels_np, split_seed)
    mean, std = standardize(embeddings, train_indices)
    labels = torch.from_numpy(labels_np).long()

    def normalized(indices: np.ndarray) -> torch.Tensor:
        index_tensor = torch.from_numpy(indices)
        return ((embeddings[index_tensor] - mean) / std).to(device)

    train_x = normalized(train_indices)
    val_x = normalized(val_indices)
    test_x = normalized(test_indices)
    train_y = labels[torch.from_numpy(train_indices)].to(device)
    val_y = labels[torch.from_numpy(val_indices)]
    test_y = labels[torch.from_numpy(test_indices)]

    candidates: list[dict[str, Any]] = []
    best_model: nn.Linear | None = None
    best_key: tuple[float, float] | None = None
    best_l2 = 0.0
    for l2_value in l2_values:
        model = fit_probe(
            train_x=train_x,
            train_y=train_y,
            l2_value=l2_value,
            iterations=lbfgs_iterations,
            device=device,
        )
        val_metrics = metrics_from_logits(predict(model, val_x), val_y)
        candidates.append(
            {
                "l2": l2_value,
                "validation_accuracy": val_metrics["accuracy"],
                "validation_macro_f1": val_metrics["macro_f1"],
            }
        )
        key = (val_metrics["macro_f1"], val_metrics["accuracy"])
        if best_key is None or key > best_key:
            best_key = key
            best_l2 = l2_value
            best_model = model
    if best_model is None:
        raise RuntimeError("No linear-probe candidate was fitted.")

    train_metrics = metrics_from_logits(predict(best_model, train_x), train_y.cpu())
    val_metrics = metrics_from_logits(predict(best_model, val_x), val_y)
    test_metrics = metrics_from_logits(predict(best_model, test_x), test_y)
    return {
        "split_seed": split_seed,
        "split_sizes": {
            "train": len(train_indices),
            "validation": len(val_indices),
            "test": len(test_indices),
        },
        "selected_l2": best_l2,
        "candidates": candidates,
        "train": train_metrics,
        "validation": val_metrics,
        "test": test_metrics,
    }


def summarize_repeats(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split in ("train", "validation", "test"):
        summary[split] = {}
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            values = np.asarray([repeat[split][metric] for repeat in repeats])
            summary[split][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "values": values.tolist(),
            }
        per_class = np.asarray([repeat[split]["per_class_f1"] for repeat in repeats])
        summary[split]["per_class_f1"] = {
            "mean": per_class.mean(axis=0).tolist(),
            "std": per_class.std(axis=0, ddof=1).tolist()
            if len(repeats) > 1
            else np.zeros(per_class.shape[1]).tolist(),
        }
    return summary


def result_path(output_dir: Path, ref: CheckpointRef) -> Path:
    return output_dir / "per_checkpoint" / ref.family / f"step_{ref.step}.json"


def evaluate_checkpoint(
    ref: CheckpointRef, args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    started = time.time()
    checkpoint = torch_load(ref.path)
    backbone, architecture, model_name = build_backbone(checkpoint, device)
    checkpoint_step = int(checkpoint.get("global_step", ref.step))
    del checkpoint
    embeddings = extract_embeddings(
        backbone=backbone,
        h5_path=args.galaxy10_h5,
        input_size=args.input_size,
        batch_size=args.embedding_batch_size,
        device=device,
        description=f"{ref.family} step={ref.step} embeddings",
    )
    embedding_dim = embeddings.shape[1]
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    labels = read_labels(args.galaxy10_h5)
    repeats = [
        evaluate_split_seed(
            embeddings=embeddings,
            labels_np=labels,
            split_seed=split_seed,
            l2_values=list(args.l2_values),
            lbfgs_iterations=args.lbfgs_iterations,
            device=device,
        )
        for split_seed in args.split_seeds
    ]
    return {
        "family": ref.family,
        "checkpoint": ref.path.name,
        "checkpoint_path": str(ref.path),
        "checkpoint_size": ref.path.stat().st_size,
        "global_step": checkpoint_step,
        "architecture": architecture,
        "model_name": model_name,
        "input_size": args.input_size,
        "preprocessing": "bicubic_resize_to_tensor_0_1_no_normalization",
        "embedding_dim": int(embedding_dim),
        "split_seeds": list(args.split_seeds),
        "l2_values": list(args.l2_values),
        "lbfgs_iterations": args.lbfgs_iterations,
        "class_balanced_cross_entropy": True,
        "repeats": repeats,
        "summary": summarize_repeats(repeats),
        "elapsed_seconds": time.time() - started,
    }


def worker(args: argparse.Namespace) -> None:
    if args.worker_index < 0 or args.worker_index >= args.worker_count:
        raise ValueError("worker-index must be within [0, worker-count).")
    manifest_path = args.output_dir.resolve() / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    refs = [
        CheckpointRef(
            family=item["family"], path=Path(item["path"]), step=int(item["step"])
        )
        for item in manifest
    ]
    assigned = refs[args.worker_index :: args.worker_count]
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_seed(10_000 + args.worker_index)
    print(
        f"Worker {args.worker_index}/{args.worker_count}: "
        f"{len(assigned)} checkpoints on {device}"
    )
    errors: list[dict[str, str]] = []
    for ref in assigned:
        path = result_path(args.output_dir.resolve(), ref)
        if path.exists() and not args.overwrite:
            print(f"Cached: {ref.family} step={ref.step}")
            continue
        try:
            result = evaluate_checkpoint(ref, args, device)
            atomic_json(path, result)
            val = result["summary"]["validation"]
            print(
                f"Done: {ref.family} step={ref.step} "
                f"val_macro_f1={val['macro_f1']['mean']:.4f} "
                f"val_acc={val['accuracy']['mean']:.4f}"
            )
        except Exception as exc:
            errors.append(
                {
                    "family": ref.family,
                    "checkpoint": str(ref.path),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"ERROR {ref.family} step={ref.step}: {exc}", file=sys.stderr)
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if errors:
        atomic_json(
            args.output_dir.resolve()
            / "errors"
            / f"worker_{args.worker_index}.json",
            errors,
        )
        raise SystemExit(1)


def load_results(output_dir: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted((output_dir / "per_checkpoint").glob("*/*.json")):
        results.append(json.loads(path.read_text()))
    return sorted(results, key=lambda item: (item["family"], item["global_step"]))


def metric_value(
    result: dict[str, Any], split: str, metric: str, statistic: str = "mean"
) -> float:
    return float(result["summary"][split][metric][statistic])


def metric_ylim(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    lower = max(0.0, min(values) - 0.05)
    upper = min(1.0, max(values) + 0.05)
    if upper - lower < 0.15:
        center = (upper + lower) / 2
        lower = max(0.0, center - 0.075)
        upper = min(1.0, center + 0.075)
    return lower, upper


def write_summary_tables(output_dir: Path, results: list[dict[str, Any]]) -> None:
    rows = []
    for result in results:
        row = {
            "family": result["family"],
            "global_step": result["global_step"],
            "checkpoint": result["checkpoint"],
            "model_name": result["model_name"],
            "embedding_dim": result["embedding_dim"],
        }
        for split in ("train", "validation", "test"):
            for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
                row[f"{split}_{metric}_mean"] = metric_value(
                    result, split, metric, "mean"
                )
                row[f"{split}_{metric}_std"] = metric_value(
                    result, split, metric, "std"
                )
        rows.append(row)
    atomic_json(output_dir / "summary.json", rows)
    if rows:
        with (output_dir / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def best_by_family(results: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    families = sorted({result["family"] for result in results})
    for family in families:
        candidates = [result for result in results if result["family"] == family]
        selected = max(
            candidates,
            key=lambda result: (
                metric_value(result, "validation", "macro_f1"),
                metric_value(result, "validation", "accuracy"),
            ),
        )
        best[family] = {
            "selection_rule": (
                "highest mean validation macro-F1; validation accuracy tie-break"
            ),
            "global_step": selected["global_step"],
            "checkpoint": selected["checkpoint"],
            "checkpoint_path": selected["checkpoint_path"],
            "model_name": selected["model_name"],
            "validation_macro_f1": selected["summary"]["validation"]["macro_f1"],
            "validation_accuracy": selected["summary"]["validation"]["accuracy"],
            "test_macro_f1": selected["summary"]["test"]["macro_f1"],
            "test_accuracy": selected["summary"]["test"]["accuracy"],
        }
    return best


def plot_family_metric(
    output_dir: Path,
    family: str,
    family_results: list[dict[str, Any]],
    metric: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = {"train": "#1f77b4", "validation": "#ff7f0e", "test": "#2ca02c"}
    steps = np.asarray([result["global_step"] for result in family_results])
    plotted_values: list[float] = []
    for split in ("train", "validation", "test"):
        means = np.asarray(
            [metric_value(result, split, metric) for result in family_results]
        )
        stds = np.asarray(
            [metric_value(result, split, metric, "std") for result in family_results]
        )
        style = "--" if split == "test" else "-"
        ax.plot(
            steps,
            means,
            marker="o",
            linewidth=2,
            linestyle=style,
            color=colors[split],
            label=split,
        )
        ax.fill_between(
            steps, means - stds, means + stds, color=colors[split], alpha=0.14
        )
        plotted_values.extend((means - stds).tolist())
        plotted_values.extend((means + stds).tolist())
    best = max(
        family_results,
        key=lambda result: (
            metric_value(result, "validation", "macro_f1"),
            metric_value(result, "validation", "accuracy"),
        ),
    )
    best_step = best["global_step"]
    best_value = metric_value(best, "validation", metric)
    ax.scatter(
        [best_step],
        [best_value],
        marker="*",
        s=220,
        color="#d62728",
        zorder=5,
        label=f"selected step {best_step}",
    )
    ax.axvline(best_step, color="#d62728", alpha=0.25, linewidth=1)
    label = "Macro-F1" if metric == "macro_f1" else "Top-1 accuracy"
    ax.set_title(f"Galaxy10 linear probe: {DISPLAY_NAMES.get(family, family)}")
    ax.set_xlabel("Pretraining checkpoint (global step)")
    ax.set_ylabel(label)
    ax.grid(alpha=0.25)
    ax.legend()
    ax.set_ylim(*metric_ylim(plotted_values))
    fig.tight_layout()
    stem = output_dir / f"{family}_{metric}_evolution"
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_model_comparison(
    output_dir: Path, results: list[dict[str, Any]], metric: str
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    plotted_values: list[float] = []
    for family in sorted({result["family"] for result in results}):
        family_results = [result for result in results if result["family"] == family]
        if len(family_results) < 2:
            continue
        steps = np.asarray([result["global_step"] for result in family_results])
        means = np.asarray(
            [
                metric_value(result, "validation", metric)
                for result in family_results
            ]
        )
        stds = np.asarray(
            [
                metric_value(result, "validation", metric, "std")
                for result in family_results
            ]
        )
        ax.plot(
            steps,
            means,
            marker="o",
            linewidth=2,
            label=DISPLAY_NAMES.get(family, family),
        )
        ax.fill_between(steps, means - stds, means + stds, alpha=0.12)
        plotted_values.extend((means - stds).tolist())
        plotted_values.extend((means + stds).tolist())
    label = "Validation macro-F1" if metric == "macro_f1" else "Validation accuracy"
    ax.set_title("Galaxy10 downstream quality across pretrained checkpoints")
    ax.set_xlabel("Pretraining checkpoint (global step)")
    ax.set_ylabel(label)
    ax.set_ylim(*metric_ylim(plotted_values))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    stem = output_dir / f"all_models_validation_{metric}"
    fig.savefig(stem.with_suffix(".png"), dpi=200)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def plot_best_per_class(
    output_dir: Path, results: list[dict[str, Any]], best: dict[str, Any]
) -> None:
    rows, labels = [], []
    for family, selection in best.items():
        selected = next(
            result
            for result in results
            if result["family"] == family
            and result["global_step"] == selection["global_step"]
        )
        rows.append(
            selected["summary"]["validation"]["per_class_f1"]["mean"]
        )
        labels.append(
            f"{DISPLAY_NAMES.get(family, family)}\nstep {selection['global_step']}"
        )
    matrix = np.asarray(rows)
    fig, ax = plt.subplots(figsize=(15, max(3.5, 1.25 * len(rows))))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=40, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Validation per-class F1 at each family's selected checkpoint")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] < 0.45 else "black"
            ax.text(
                column,
                row,
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, label="F1")
    fig.tight_layout()
    fig.savefig(output_dir / "best_checkpoints_per_class_f1.png", dpi=200)
    fig.savefig(output_dir / "best_checkpoints_per_class_f1.pdf")
    plt.close(fig)


def plot(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    results = load_results(output_dir)
    if not results:
        raise RuntimeError("No per-checkpoint results found.")
    manifest = json.loads((output_dir / "manifest.json").read_text())
    expected = {(item["family"], int(item["step"])) for item in manifest}
    actual = {(item["family"], int(item["global_step"])) for item in results}
    missing = sorted(expected - actual)
    if missing:
        print(f"Warning: {len(missing)} checkpoint results are missing.", file=sys.stderr)
    write_summary_tables(output_dir, results)
    best = best_by_family(results)
    atomic_json(output_dir / "best_checkpoints.json", best)
    for family in sorted({result["family"] for result in results}):
        family_results = [result for result in results if result["family"] == family]
        plot_family_metric(output_dir, family, family_results, "macro_f1")
        plot_family_metric(output_dir, family, family_results, "accuracy")
    plot_model_comparison(output_dir, results, "macro_f1")
    plot_model_comparison(output_dir, results, "accuracy")
    plot_best_per_class(output_dir, results, best)
    print(f"Generated plots and summaries in {output_dir}")


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "worker":
        worker(args)
    elif args.command == "plot":
        plot(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
