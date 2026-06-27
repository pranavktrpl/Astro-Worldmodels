#!/usr/bin/env python3
"""Plot per-class Galaxy10 test recall and F1 for the best checkpoint."""

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


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = SCRIPT_DIR / "results"

SHORT_LABELS = [
    "Disturbed",
    "Merging",
    "Round\nsmooth",
    "Intermediate\nsmooth",
    "Cigar\nsmooth",
    "Barred\nspiral",
    "Tight\nspiral",
    "Loose\nspiral",
    "Edge-on\nno bulge",
    "Edge-on\n+ bulge",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="Plot held-out test metrics by default.",
    )
    return parser.parse_args()


def selected_result(results_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    selections = json.loads((results_dir / "best_checkpoints.json").read_text())
    family, selection = max(
        selections.items(),
        key=lambda item: (
            float(item[1]["validation_macro_f1"]["mean"]),
            float(item[1]["validation_accuracy"]["mean"]),
        ),
    )
    result_path = (
        results_dir
        / "per_checkpoint"
        / family
        / f"step_{int(selection['global_step'])}.json"
    )
    return selection, json.loads(result_path.read_text())


def per_class_metrics(
    result: dict[str, Any], split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    recalls, f1_scores = [], []
    for repeat in result["repeats"]:
        metrics = repeat[split]
        confusion = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
        support = confusion.sum(axis=1)
        recalls.append(
            np.divide(
                np.diag(confusion),
                support,
                out=np.zeros_like(support),
                where=support > 0,
            )
        )
        f1_scores.append(np.asarray(metrics["per_class_f1"], dtype=np.float64))
    recall_values = np.asarray(recalls)
    f1_values = np.asarray(f1_scores)
    return (
        recall_values.mean(axis=0),
        recall_values.std(axis=0, ddof=1),
        f1_values.mean(axis=0),
        f1_values.std(axis=0, ddof=1),
    )


def close(values: np.ndarray) -> np.ndarray:
    return np.concatenate([values, values[:1]])


def radar(
    ax: plt.Axes,
    angles: np.ndarray,
    values: np.ndarray,
    errors: np.ndarray,
    title: str,
    color: str,
    average: float,
) -> None:
    closed_angles = close(angles)
    closed_values = close(values)
    lower = close(np.clip(values - errors, 0, 1))
    upper = close(np.clip(values + errors, 0, 1))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(closed_angles, closed_values, color=color, linewidth=2.6, marker="o")
    ax.fill(closed_angles, closed_values, color=color, alpha=0.17)
    ax.fill_between(closed_angles, lower, upper, color=color, alpha=0.12)
    ax.plot(
        closed_angles,
        np.full_like(closed_angles, average),
        color="#424b54",
        linestyle="--",
        linewidth=1.4,
        label=f"class mean: {average:.1%}",
    )

    ax.set_xticks(angles)
    ax.set_xticklabels(SHORT_LABELS, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8)
    ax.set_rlabel_position(18)
    ax.grid(alpha=0.3)
    ax.set_title(title, pad=32, fontsize=16, fontweight="bold")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.19), frameon=False)

    for angle, value in zip(angles, values):
        ax.annotate(
            f"{value:.0%}",
            xy=(angle, value),
            xytext=(0, 8 if value < 0.88 else -15),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=color,
        )


def write_metrics(
    results_dir: Path,
    selection: dict[str, Any],
    result: dict[str, Any],
    split: str,
    recall_mean: np.ndarray,
    recall_std: np.ndarray,
    f1_mean: np.ndarray,
    f1_std: np.ndarray,
) -> None:
    class_names = json.loads((results_dir / "dataset_metadata.json").read_text())[
        "class_names"
    ]
    rows = [
        {
            "class": class_name,
            "class_accuracy_recall_mean": float(recall_mean[index]),
            "class_accuracy_recall_std": float(recall_std[index]),
            "f1_mean": float(f1_mean[index]),
            "f1_std": float(f1_std[index]),
        }
        for index, class_name in enumerate(class_names)
    ]
    csv_path = results_dir / "best_model_galaxy10_per_class_metrics.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = result["summary"][split]
    payload = {
        "family": result["family"],
        "checkpoint": selection["checkpoint"],
        "global_step": int(selection["global_step"]),
        "selection_rule": selection["selection_rule"],
        "reported_split": split,
        "overall_accuracy": summary["accuracy"],
        "mean_class_accuracy_balanced_accuracy": summary["balanced_accuracy"],
        "macro_f1": summary["macro_f1"],
        "per_class": rows,
    }
    (results_dir / "best_model_galaxy10_per_class_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    selection, result = selected_result(results_dir)
    recall_mean, recall_std, f1_mean, f1_std = per_class_metrics(
        result, args.split
    )
    summary = result["summary"][args.split]

    angles = np.linspace(0, 2 * np.pi, len(SHORT_LABELS), endpoint=False)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18, 9.5),
        subplot_kw={"projection": "polar"},
    )
    radar(
        axes[0],
        angles,
        recall_mean,
        recall_std,
        "Per-class accuracy (recall)",
        "#2878b5",
        float(summary["balanced_accuracy"]["mean"]),
    )
    radar(
        axes[1],
        angles,
        f1_mean,
        f1_std,
        "Per-class F1 score",
        "#e15759",
        float(summary["macro_f1"]["mean"]),
    )
    fig.suptitle(
        "Galaxy10 DECaLS: best Astro-Worldmodels linear probe",
        fontsize=22,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.945,
        (
            f"ViT-L/14, step {selection['global_step']:,} | "
            f"held-out {args.split} mean +/- SD over "
            f"{len(result['repeats'])} stratified splits | "
            f"overall accuracy {summary['accuracy']['mean']:.1%}"
        ),
        ha="center",
        fontsize=12,
        color="#5f6b76",
    )
    fig.subplots_adjust(top=0.86, bottom=0.13, wspace=0.25)

    stem = results_dir / "best_model_galaxy10_per_class_radar"
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    write_metrics(
        results_dir=results_dir,
        selection=selection,
        result=result,
        split=args.split,
        recall_mean=recall_mean,
        recall_std=recall_std,
        f1_mean=f1_mean,
        f1_std=f1_std,
    )
    print(f"Generated {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
