#!/usr/bin/env python3
"""Train AstroCLIP-style morphology MLPs on frozen GZD-5 embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
RESULTS_DIR = SCRIPT_DIR / "results"

MODELS = {
    "large": "astro_vit_large_step_52000",
    "small": "astro_vit_small_step_21000",
}

QUESTIONS = {
    "smooth": ("smooth-or-featured", ["smooth", "featured-or-disk", "artifact"]),
    "disk-edge-on": ("disk-edge-on", ["yes", "no"]),
    "spiral-arms": ("has-spiral-arms", ["yes", "no"]),
    "bar": ("bar", ["strong", "weak", "no"]),
    "bulge-size": ("bulge-size", ["dominant", "large", "moderate", "small", "none"]),
    "how-rounded": ("how-rounded", ["round", "in-between", "cigar-shaped"]),
    "edge-on-bulge": ("edge-on-bulge", ["boxy", "none", "rounded"]),
    "spiral-winding": ("spiral-winding", ["tight", "medium", "loose"]),
    "spiral-arm-count": ("spiral-arm-count", ["1", "2", "3", "4", "more-than-4", "cant-tell"]),
    "merging": ("merging", ["none", "minor-disturbance", "major-disturbance", "merger"]),
}


class MLP(nn.Module):
    """AstroCLIP morphology MLP: 3 hidden layers plus output layer.

    apply_softmax=True reproduces the AstroCLIP code (softmax inside forward,
    then CrossEntropyLoss re-softmaxes — the double-softmax collapses training
    to majority-class prediction on imbalanced questions). apply_softmax=False
    is the fixed-head protocol: raw logits into the loss.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        dropout_rate: float,
        apply_softmax: bool = True,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.apply_softmax = apply_softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.layers(x)
        return self.softmax(logits) if self.apply_softmax else logits


def normalize_targets(y: torch.Tensor) -> torch.Tensor:
    y = y.float()
    y = y / y.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return y


def valid_target_mask(y: torch.Tensor) -> torch.Tensor:
    return (~torch.isnan(y).any(dim=1)) & (y.sum(dim=1) > 0)


def train_eval_on_question(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    hidden_dim: int,
    batch_size: int,
    lr: float,
    epochs: int,
    dropout: float,
    seed: int,
    device: torch.device,
    fixed_head: bool = False,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    if fixed_head:
        # inverse-frequency class weights from the soft targets, mean 1
        class_frequency = y_train.mean(dim=0).clamp_min(1e-3)
        class_weights = (1.0 / class_frequency)
        class_weights = (class_weights / class_weights.mean()).to(device)
    else:
        class_weights = None

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.1,
        random_state=42,
    )
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    samples_weight = y_train.max(dim=1).values.double().clamp_min(1e-6)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=WeightedRandomSampler(samples_weight, len(samples_weight), generator=generator),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    mlp = MLP(
        x_train.shape[1],
        y_train.shape[1],
        hidden_dim,
        dropout,
        apply_softmax=not fixed_head,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    best_val_loss = float("inf")
    best_model = None
    history = []

    for epoch in range(epochs):
        mlp.train()
        train_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = mlp(data.to(device))
            loss = criterion(output, target.to(device))
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
        train_loss /= max(1, len(train_loader))

        mlp.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                output = mlp(data.to(device))
                loss = criterion(output, target.to(device))
                val_loss += float(loss.item())
        val_loss /= max(1, len(val_loader))
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = {k: v.detach().cpu().clone() for k, v in mlp.state_dict().items()}

    assert best_model is not None
    mlp.load_state_dict(best_model)
    mlp.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(x_test), batch_size * 8):
            preds.append(mlp(x_test[start : start + batch_size * 8].to(device)).detach().cpu())
    y_pred = torch.cat(preds, dim=0)

    pred_class = y_pred.argmax(dim=1).numpy()
    true_class = y_test.argmax(dim=1).numpy()
    pred_onehot = np.eye(y_test.shape[1], dtype=np.int64)[pred_class]
    true_onehot = np.eye(y_test.shape[1], dtype=np.int64)[true_class]
    accuracy = accuracy_score(true_onehot, pred_onehot)
    f1 = precision_recall_fscore_support(
        true_onehot,
        pred_onehot,
        average="weighted",
        zero_division=0,
    )[2]
    return {
        "Accuracy": float(accuracy),
        "F1 Score": float(f1),
        "best_val_loss": float(best_val_loss),
        "history": history,
    }


def question_targets(frame: pd.DataFrame, question: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    prefix, answers = QUESTIONS[question]
    target_cols = [f"{prefix}_{answer}_debiased" for answer in answers]
    count_col = f"{prefix}_total-votes"
    return frame[target_cols].to_numpy(np.float32), frame[count_col].to_numpy(np.float32), target_cols


def make_radar(metrics: dict[str, Any], key: str, output: Path) -> None:
    names = list(QUESTIONS.keys())
    values = [metrics[name][key] for name in names]
    angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
    values = values + values[:1]
    angles = angles + angles[:1]
    fig = plt.figure(figsize=(8, 8), dpi=180)
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title(key)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def write_csv(path: Path, metrics: dict[str, Any]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "question",
                "num_classes",
                "train_samples",
                "test_samples",
                "accuracy",
                "f1",
                "best_val_loss",
            ],
        )
        writer.writeheader()
        for question, record in metrics.items():
            if question == "mean":
                continue
            writer.writerow(
                {
                    "question": question,
                    "num_classes": record["num_classes"],
                    "train_samples": record["train_samples"],
                    "test_samples": record["test_samples"],
                    "accuracy": record["Accuracy"],
                    "f1": record["F1 Score"],
                    "best_val_loss": record["best_val_loss"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), default="large")
    parser.add_argument(
        "--label",
        default=None,
        help="Results directory name under results/ to train on (overrides "
        "--model; must contain embeddings from extract_embeddings.py).",
    )
    parser.add_argument(
        "--fixed-head",
        action="store_true",
        help="CE on logits with inverse-frequency class weights instead of "
        "the AstroCLIP double-softmax; writes *_fixedhead output files.",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    label = args.label or MODELS[args.model]
    output_dir = RESULTS_DIR / label
    if not (output_dir / "train_embeddings.npy").exists():
        raise SystemExit(
            f"No embeddings under {output_dir} — run extract_embeddings.py "
            "with the matching --label first."
        )
    x_train = torch.from_numpy(np.load(output_dir / "train_embeddings.npy")).float()
    x_test = torch.from_numpy(np.load(output_dir / "test_embeddings.npy")).float()
    train = pd.read_parquet(DATA_DIR / "merged_train.parquet")
    test = pd.read_parquet(DATA_DIR / "merged_test.parquet")
    if len(train) != len(x_train) or len(test) != len(x_test):
        raise RuntimeError("Embedding rows do not match merged catalog rows.")

    device = torch.device(args.device)
    metrics: dict[str, Any] = {}
    for question in tqdm(list(QUESTIONS.keys()), desc="questions"):
        y_train_np, _train_counts, target_cols = question_targets(train, question)
        y_test_np, test_counts, _ = question_targets(test, question)
        y_train = torch.from_numpy(y_train_np)
        y_test = torch.from_numpy(y_test_np)

        train_mask = valid_target_mask(y_train)
        test_mask = valid_target_mask(y_test) & torch.from_numpy(test_counts > 34)
        y_train_q = normalize_targets(y_train[train_mask])
        y_test_q = normalize_targets(y_test[test_mask])
        x_train_q = x_train[train_mask]
        x_test_q = x_test[test_mask]
        result = train_eval_on_question(
            x_train_q,
            x_test_q,
            y_train_q,
            y_test_q,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            dropout=args.dropout,
            seed=args.seed,
            device=device,
            fixed_head=args.fixed_head,
        )
        result.update(
            {
                "question": question,
                "target_columns": target_cols,
                "num_classes": len(target_cols),
                "train_samples": int(len(x_train_q)),
                "test_samples": int(len(x_test_q)),
            }
        )
        metrics[question] = result
        print(
            f"{question}: acc={result['Accuracy']:.4f} "
            f"f1={result['F1 Score']:.4f} "
            f"train={len(x_train_q)} test={len(x_test_q)}"
        )

    metrics["mean"] = {
        "Accuracy": float(np.mean([metrics[q]["Accuracy"] for q in QUESTIONS])),
        "F1 Score": float(np.mean([metrics[q]["F1 Score"] for q in QUESTIONS])),
    }
    metadata = {
        "model": label,
        "protocol": (
            "fixed head: CE on logits with inverse-frequency class weights"
            if args.fixed_head
            else "AstroCLIP morphology MLP reimplementation"
        ),
        "mlp": {
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "lr": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "loss": (
                "CrossEntropyLoss on logits with inverse-frequency class "
                "weights (fixes the AstroCLIP double-softmax collapse)"
                if args.fixed_head
                else "CrossEntropyLoss on soft debiased targets after model softmax, matching AstroCLIP code"
            ),
        },
        "filters": {
            "global": "smooth-or-featured_total-votes >= 3 applied in prepare_gzd5.py",
            "train": "valid non-NaN target rows",
            "test": "valid non-NaN target rows and question_total-votes > 34",
        },
    }
    suffix = "_fixedhead" if args.fixed_head else ""
    payload = {"metadata": metadata, "metrics": metrics}
    (output_dir / f"metrics{suffix}.json").write_text(json.dumps(payload, indent=2) + "\n")
    write_csv(output_dir / f"metrics{suffix}.csv", metrics)
    make_radar(metrics, "Accuracy", output_dir / f"radar_accuracy{suffix}.png")
    make_radar(metrics, "F1 Score", output_dir / f"radar_f1{suffix}.png")
    print(json.dumps(metrics["mean"], indent=2))


if __name__ == "__main__":
    main()
