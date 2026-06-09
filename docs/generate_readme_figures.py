#!/usr/bin/env python3
"""Generate the figures embedded by the repository-level README."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
RESULTS = ROOT / "Evals" / "galaxy10_checkpoint_evolution" / "results"

COLORS = {
    "navy": "#16324f",
    "blue": "#2878b5",
    "cyan": "#5ab1bb",
    "orange": "#f28e2b",
    "gold": "#edc948",
    "green": "#59a14f",
    "red": "#e15759",
    "purple": "#8f63b8",
    "gray": "#5f6b76",
    "light": "#f4f7fa",
}

DISPLAY_NAMES = {
    "VitLargePatch14_OfficialTrain5_Epoch5_2504": "ViT-L/14",
    "VitSmallPatch14_2204": "ViT-S/14",
    "checkpoints-test": "ResNet9",
}


def save(fig: plt.Figure, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(ASSETS / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    color: str,
    title_size: float = 12,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.014,rounding_size=0.025",
        linewidth=1.8,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=color,
    )
    ax.text(
        x + width / 2,
        y + height * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=9.2,
        color=COLORS["gray"],
        linespacing=1.25,
    )


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = COLORS["gray"],
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.7,
            color=color,
            connectionstyle=connectionstyle,
        )
    )


def method_overview() -> None:
    fig, ax = plt.subplots(figsize=(16, 8.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.96,
        "Astro-Worldmodels: astronomy-adapted LeJEPA pretraining",
        ha="center",
        va="center",
        fontsize=22,
        fontweight="bold",
        color=COLORS["navy"],
    )
    ax.text(
        0.5,
        0.915,
        "Labels are never used during pretraining; every crop passes through one shared encoder and projector.",
        ha="center",
        va="center",
        fontsize=11,
        color=COLORS["gray"],
    )

    box(
        ax,
        (0.03, 0.40),
        0.15,
        0.25,
        "HF galaxy stream",
        "Smith42/galaxies\n256 px image_crop\n8,474,566 train objects",
        COLORS["navy"],
    )
    box(
        ax,
        (0.23, 0.61),
        0.17,
        0.19,
        "2 global views",
        "140 x 140\nlarge field of view\nasymmetric blur + noise",
        COLORS["blue"],
    )
    box(
        ax,
        (0.23, 0.25),
        0.17,
        0.19,
        "8 local views",
        "56 x 56\nfine morphology\n50% blur + noise",
        COLORS["cyan"],
    )
    box(
        ax,
        (0.46, 0.40),
        0.16,
        0.25,
        "Shared encoder",
        "ViT-S/14 or ViT-L/14\nrandom initialization\npooled representation h",
        COLORS["purple"],
    )
    box(
        ax,
        (0.67, 0.40),
        0.13,
        0.25,
        "MLP projector",
        "2 hidden layers\nBatchNorm + ReLU\nprojection z",
        COLORS["orange"],
    )
    box(
        ax,
        (0.84, 0.62),
        0.13,
        0.20,
        "Invariance",
        "all views match\nthe two-global-view\nsample center",
        COLORS["green"],
    )
    box(
        ax,
        (0.84, 0.23),
        0.13,
        0.22,
        "SIGReg",
        "1,024 random slices\nEpps-Pulley test\ntoward N(0, 1)",
        COLORS["red"],
    )

    arrow(ax, (0.18, 0.53), (0.23, 0.70), COLORS["blue"])
    arrow(ax, (0.18, 0.49), (0.23, 0.35), COLORS["cyan"])
    arrow(ax, (0.40, 0.70), (0.46, 0.57), COLORS["blue"])
    arrow(ax, (0.40, 0.35), (0.46, 0.48), COLORS["cyan"])
    arrow(ax, (0.62, 0.525), (0.67, 0.525), COLORS["purple"])
    arrow(ax, (0.80, 0.56), (0.84, 0.70), COLORS["green"])
    arrow(ax, (0.80, 0.49), (0.84, 0.34), COLORS["red"])

    ax.text(
        0.75,
        0.075,
        r"$\mathcal{L}=(1-\lambda)\,\mathcal{L}_{inv}"
        r"+\lambda\,\mathcal{L}_{SIGReg},\quad \lambda=0.05$",
        ha="center",
        va="center",
        fontsize=17,
        color=COLORS["navy"],
        bbox=dict(
            boxstyle="round,pad=0.55",
            facecolor=COLORS["light"],
            edgecolor=COLORS["navy"],
        ),
    )
    arrow(
        ax,
        (0.905, 0.23),
        (0.81, 0.135),
        COLORS["red"],
        connectionstyle="arc3,rad=-0.15",
    )
    arrow(
        ax,
        (0.905, 0.62),
        (0.79, 0.135),
        COLORS["green"],
        connectionstyle="arc3,rad=0.15",
    )
    ax.text(
        0.28,
        0.08,
        "No decoder  |  No labels  |  No EMA teacher\n"
        "No stop-gradient  |  No separate predictor",
        ha="center",
        va="center",
        fontsize=10.5,
        color=COLORS["gray"],
    )
    save(fig, "method_overview")


def galaxy10_distribution() -> None:
    metadata = json.loads((RESULTS / "dataset_metadata.json").read_text())
    names = [name.replace("_", " ") for name in metadata["class_names"]]
    counts = np.asarray(metadata["class_counts"])
    order = np.argsort(counts)

    fig, ax = plt.subplots(figsize=(11, 6.8))
    bars = ax.barh(
        np.asarray(names)[order],
        counts[order],
        color=COLORS["blue"],
        edgecolor="white",
    )
    total = counts.sum()
    for bar, count in zip(bars, counts[order]):
        ax.text(
            bar.get_width() + 35,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({100 * count / total:.1f}%)",
            va="center",
            fontsize=9.5,
            color=COLORS["navy"],
        )
    ax.set_title(
        "Galaxy10 DECaLS evaluation set: 17,736 labeled galaxies",
        fontsize=17,
        fontweight="bold",
        color=COLORS["navy"],
    )
    ax.set_xlabel("Images")
    ax.set_xlim(0, counts.max() * 1.24)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "galaxy10_class_distribution")


def read_loss(path: Path) -> dict[str, list[dict[str, str]]]:
    rows = list(csv.DictReader(path.open()))
    result: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "validation", "test"):
        result[split] = sorted(
            [row for row in rows if row["split"] == split],
            key=lambda row: int(row["global_step"]),
        )
    return result


def selection_dashboard() -> None:
    large_loss = read_loss(
        ROOT
        / "checkpoints"
        / "VitLargePatch14_OfficialTrain5_Epoch5_2504"
        / "loss_evolution_metrics.csv"
    )
    small_loss = read_loss(
        ROOT / "checkpoints" / "VitSmallPatch14_2204" / "loss_evolution_metrics.csv"
    )
    probe_rows = list(csv.DictReader((RESULTS / "summary.csv").open()))
    best = json.loads((RESULTS / "best_checkpoints.json").read_text())

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "Checkpoint selection: self-supervised objective and Galaxy10 transfer",
        fontsize=21,
        fontweight="bold",
        color=COLORS["navy"],
    )

    ax = axes[0, 0]
    for label, records, color, selected in (
        ("ViT-L/14", large_loss["validation"], COLORS["blue"], 52000),
        ("ViT-S/14", small_loss["validation"], COLORS["orange"], 20000),
    ):
        steps = np.asarray([int(row["global_step"]) for row in records])
        values = np.asarray([float(row["loss"]) for row in records])
        errors = np.asarray([float(row["loss_standard_error"]) for row in records])
        ax.plot(steps, values, marker="o", linewidth=2.2, color=color, label=label)
        ax.fill_between(steps, values - errors, values + errors, color=color, alpha=0.14)
        chosen = np.flatnonzero(steps == selected)[0]
        ax.scatter(
            steps[chosen],
            values[chosen],
            marker="*",
            s=180,
            color=COLORS["red"],
            zorder=5,
        )
    ax.set_title("A. Streamed validation LeJEPA loss")
    ax.set_xlabel("Pretraining step")
    ax.set_ylabel("Loss (lower is better)")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    for family, color in (
        ("VitLargePatch14_OfficialTrain5_Epoch5_2504", COLORS["blue"]),
        ("VitSmallPatch14_2204", COLORS["orange"]),
    ):
        rows = sorted(
            [row for row in probe_rows if row["family"] == family],
            key=lambda row: int(row["global_step"]),
        )
        steps = np.asarray([int(row["global_step"]) for row in rows])
        values = np.asarray([float(row["validation_macro_f1_mean"]) for row in rows])
        errors = np.asarray([float(row["validation_macro_f1_std"]) for row in rows])
        ax.plot(
            steps,
            values,
            marker="o",
            linewidth=2.2,
            color=color,
            label=DISPLAY_NAMES[family],
        )
        ax.fill_between(steps, values - errors, values + errors, color=color, alpha=0.14)
        selected = int(best[family]["global_step"])
        chosen = np.flatnonzero(steps == selected)[0]
        ax.scatter(
            steps[chosen],
            values[chosen],
            marker="*",
            s=180,
            color=COLORS["red"],
            zorder=5,
        )
    ax.set_title("B. Galaxy10 validation macro-F1")
    ax.set_xlabel("Pretraining step")
    ax.set_ylabel("Macro-F1 (higher is better)")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    families = [
        "VitLargePatch14_OfficialTrain5_Epoch5_2504",
        "VitSmallPatch14_2204",
        "checkpoints-test",
    ]
    labels = [DISPLAY_NAMES[family] for family in families]
    validation = [best[family]["validation_macro_f1"]["mean"] for family in families]
    validation_std = [best[family]["validation_macro_f1"]["std"] for family in families]
    test = [best[family]["test_macro_f1"]["mean"] for family in families]
    test_std = [best[family]["test_macro_f1"]["std"] for family in families]
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(
        x - width / 2,
        validation,
        width,
        yerr=validation_std,
        capsize=4,
        color=COLORS["orange"],
        label="validation",
    )
    ax.bar(
        x + width / 2,
        test,
        width,
        yerr=test_std,
        capsize=4,
        color=COLORS["green"],
        label="test",
    )
    ax.set_xticks(x, labels)
    ax.set_ylim(0.35, 0.75)
    ax.set_ylabel("Macro-F1")
    ax.set_title("C. Selected-checkpoint downstream quality")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)

    ax = axes[1, 1]
    selected_large = next(
        row
        for row in probe_rows
        if row["family"] == "VitLargePatch14_OfficialTrain5_Epoch5_2504"
        and int(row["global_step"]) == 52000
    )
    del selected_large
    result = json.loads(
        (
            RESULTS
            / "per_checkpoint"
            / "VitLargePatch14_OfficialTrain5_Epoch5_2504"
            / "step_52000.json"
        ).read_text()
    )
    class_names = [
        name.replace("_", " ") for name in json.loads(
            (RESULTS / "dataset_metadata.json").read_text()
        )["class_names"]
    ]
    class_f1 = np.asarray(
        result["summary"]["validation"]["per_class_f1"]["mean"]
    )
    class_order = np.argsort(class_f1)
    ax.barh(
        np.asarray(class_names)[class_order],
        class_f1[class_order],
        color=COLORS["purple"],
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Validation F1")
    ax.set_title("D. ViT-L/14 step 52k by morphology")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "checkpoint_selection_dashboard")


def main() -> None:
    method_overview()
    galaxy10_distribution()
    selection_dashboard()
    print(f"Generated README figures in {ASSETS}")


if __name__ == "__main__":
    main()
