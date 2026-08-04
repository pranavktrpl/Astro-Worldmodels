#!/usr/bin/env python
"""Galaxy10 classification with exact reproductions of competitor probe heads.

The published numbers we compare against use higher-capacity heads than our
linear probe, which confounds the comparison (see Evals/README.md). This eval
trains, on the same frozen embeddings and the same 80/10/10 class-stratified
splits (seeds 42/43/44) as galaxy10_checkpoint_evolution, two head
reproductions:

  aion_mlp      AION-1's Galaxy Zoo 10 head: two-layer MLP, hidden size 256,
                GELU, dropout 0.1, cross-entropy on logits (arXiv:2510.17960).
  astroclip_mlp AstroCLIP's morphology head: 4 linear layers, hidden 256,
                ReLU, dropout 0.2, softmax applied before cross-entropy —
                mirroring Evals/gzd5_morphology_probe/train_gzd5_mlp.py, which
                reimplements the AstroCLIP code including that quirk.

Head architectures and losses are exact. AION-1 does not publish the probe's
optimizer, schedule, or batch size, so training follows this repo's convention
(Adam 1e-3, fixed epochs, best-validation-loss checkpoint selection); the
metadata block records which settings come from the papers and which are
assumed. Embeddings are shared with Evals/redshift_regression (identical
140 px bicubic 0-1 preprocessing): an existing cache there is reused, and a
fresh extraction is cached under this suite's results directory.

Run (needs Galaxy10_DECals.h5 and the recommended checkpoints):

    python matched_heads_probe.py --model all
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
EVALS_DIR = SCRIPT_DIR.parent
REDSHIFT_RESULTS = EVALS_DIR / "redshift_regression" / "results"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


redshift_probe = load_module(
    "redshift_probe", EVALS_DIR / "redshift_regression" / "redshift_probe.py"
)
evolution_probe = load_module(
    "galaxy10_probe_evolution",
    EVALS_DIR / "galaxy10_checkpoint_evolution" / "galaxy10_probe_evolution.py",
)

MODELS = redshift_probe.MODELS
CLASS_NAMES = evolution_probe.CLASS_NAMES
SCALAR_METRICS = ("accuracy", "macro_f1", "balanced_accuracy")

HEADS_METADATA = {
    "aion_mlp": {
        "architecture": "Linear(dim, 256) -> GELU -> Dropout(0.1) -> Linear(256, 10)",
        "loss": "CrossEntropyLoss on logits",
        "from_paper": (
            "AION-1 (arXiv:2510.17960): 'two-layer MLP head (hidden size = 256, "
            "GELU, dropout = 0.1)' on frozen mean-pooled embeddings; "
            "class-stratified split"
        ),
        "assumed": (
            "Adam lr 1e-3, batch 256, fixed epochs with best-validation-loss "
            "selection; standardized features; our 80/10/10 split (AION uses "
            "80/20 with no validation set); our eval set is Galaxy10 DECaLS, "
            "not AION's GZ10 x Legacy Survey DR10 cross-match (~8k galaxies)"
        ),
    },
    "astroclip_mlp": {
        "architecture": (
            "Linear(dim, 256) -> Dropout(0.2) -> ReLU, x3, -> Linear(256, 10), "
            "softmax inside forward"
        ),
        "loss": "CrossEntropyLoss applied to softmax outputs, matching AstroCLIP code",
        "from_paper": (
            "AstroCLIP morphology MLP as reimplemented in "
            "Evals/gzd5_morphology_probe/train_gzd5_mlp.py (hidden 256, dropout "
            "0.2, Adam 1e-3, batch 256)"
        ),
        "assumed": (
            "applied to hard Galaxy10 labels instead of GZD-5 debiased soft "
            "labels; our 80/10/10 split; epochs with best-validation-loss "
            "selection"
        ),
    },
}


class AionMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AstroclipMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.layers(x))


HEAD_CLASSES = {"aion_mlp": AionMLP, "astroclip_mlp": AstroclipMLP}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--galaxy10-h5", type=Path, default=redshift_probe.DEFAULT_H5)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--input-size", type=int, default=140)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_embeddings(
    spec: dict[str, Any], args: argparse.Namespace, device: torch.device
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Reuse the redshift probe's cache when present, else extract into ours."""
    for cache_root in (REDSHIFT_RESULTS, args.output_dir.resolve()):
        cache = cache_root / spec["label"] / "embeddings.npy"
        meta_path = cache_root / spec["label"] / "embedding_metadata.json"
        if cache.exists() and meta_path.exists() and not args.overwrite:
            print(f"cached embeddings: {cache}")
            return (
                torch.from_numpy(np.load(cache)),
                json.loads(meta_path.read_text()),
            )
    extraction_args = SimpleNamespace(
        output_dir=args.output_dir.resolve(),
        galaxy10_h5=args.galaxy10_h5,
        input_size=args.input_size,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
    return redshift_probe.cached_embeddings(spec, extraction_args, device)


def read_labels(h5_path: Path) -> np.ndarray:
    import h5py

    with h5py.File(h5_path, "r") as handle:
        return np.asarray(handle["ans"][:], dtype=np.int64)


def train_head(
    head: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, float]]]:
    redshift_probe.set_seed(seed)
    head = head.to(device)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    for epoch in range(epochs):
        head.train()
        train_loss = 0.0
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(features), labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
        train_loss /= max(1, len(loader))

        head.eval()
        with torch.no_grad():
            val_loss = float(criterion(head(val_x), val_y).item())
        history.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                key: value.detach().clone() for key, value in head.state_dict().items()
            }
    assert best_state is not None
    head.load_state_dict(best_state)
    head.eval()
    return head, history


def evaluate_split_seed(
    embeddings: torch.Tensor,
    labels_np: np.ndarray,
    split_seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    train_idx, val_idx, test_idx = evolution_probe.stratified_split(
        labels_np, split_seed
    )
    mean, std = evolution_probe.standardize(embeddings, train_idx)
    labels = torch.from_numpy(labels_np).long()

    def normalized(indices: np.ndarray) -> torch.Tensor:
        index_tensor = torch.from_numpy(indices)
        return ((embeddings[index_tensor] - mean) / std).to(device)

    train_x, val_x, test_x = (normalized(i) for i in (train_idx, val_idx, test_idx))
    train_y = labels[torch.from_numpy(train_idx)].to(device)
    val_y = labels[torch.from_numpy(val_idx)].to(device)
    test_y = labels[torch.from_numpy(test_idx)]

    heads: dict[str, Any] = {}
    for head_name, head_class in HEAD_CLASSES.items():
        head, history = train_head(
            head_class(embeddings.shape[1], len(CLASS_NAMES)),
            train_x,
            train_y,
            val_x,
            val_y,
            epochs=args.epochs,
            batch_size=args.head_batch_size,
            lr=args.lr,
            seed=split_seed,
            device=device,
        )
        with torch.no_grad():
            heads[head_name] = {
                "best_val_loss": min(entry["val_loss"] for entry in history),
                "epochs_trained": len(history),
                "train": evolution_probe.metrics_from_logits(
                    evolution_probe.predict(head, train_x), train_y.cpu()
                ),
                "validation": evolution_probe.metrics_from_logits(
                    evolution_probe.predict(head, val_x), val_y.cpu()
                ),
                "test": evolution_probe.metrics_from_logits(
                    evolution_probe.predict(head, test_x), test_y
                ),
            }
    return {
        "split_seed": split_seed,
        "split_sizes": {
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        "heads": heads,
    }


def summarize_repeats(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for head in HEAD_CLASSES:
        summary[head] = {}
        for split in ("train", "validation", "test"):
            summary[head][split] = {}
            for metric in SCALAR_METRICS:
                values = [
                    repeat["heads"][head][split][metric] for repeat in repeats
                ]
                summary[head][split][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values,
                }
    return summary


def evaluate_model(
    key: str, args: argparse.Namespace, device: torch.device
) -> dict[str, Any]:
    spec = MODELS[key]
    started = time.time()
    output_dir = args.output_dir.resolve() / spec["label"]
    embeddings, embed_meta = load_embeddings(spec, args, device)
    labels = read_labels(args.galaxy10_h5)
    if len(labels) != len(embeddings):
        raise RuntimeError(
            f"{len(labels)} labels but {len(embeddings)} embeddings; the "
            "embedding cache does not match this HDF5."
        )

    repeats = []
    for split_seed in args.split_seeds:
        repeat = evaluate_split_seed(embeddings, labels, split_seed, args, device)
        repeats.append(repeat)
        line = ", ".join(
            f"{head} test acc={repeat['heads'][head]['test']['accuracy']:.4f}"
            for head in HEAD_CLASSES
        )
        print(f"{spec['label']} seed={split_seed}: {line}")

    result = {
        "label": spec["label"],
        "embedding_metadata": embed_meta,
        "split_seeds": args.split_seeds,
        "heads_metadata": HEADS_METADATA,
        "training": {
            "optimizer": "Adam",
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.head_batch_size,
            "selection": "best validation loss",
            "features": "standardized by train-split mean/std",
        },
        "class_names": CLASS_NAMES,
        "repeats": repeats,
        "summary": summarize_repeats(repeats),
        "elapsed_seconds": time.time() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    redshift_probe.atomic_json(output_dir / "metrics.json", result)
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    keys = list(MODELS) if args.model == "all" else [args.model]
    for key in keys:
        result = evaluate_model(key, args, device)
        for head in HEAD_CLASSES:
            stats = result["summary"][head]["test"]
            print(
                f"{result['label']} {head}: "
                f"test acc {stats['accuracy']['mean']:.4f} "
                f"± {stats['accuracy']['std']:.4f}, "
                f"macro-F1 {stats['macro_f1']['mean']:.4f} "
                f"± {stats['macro_f1']['std']:.4f}"
            )


if __name__ == "__main__":
    main()
