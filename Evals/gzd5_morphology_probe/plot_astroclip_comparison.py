#!/usr/bin/env python3
"""Plot AstroCLIP paper vs local ViT-L GZD-5 morphology radar comparison."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "results/astro_vit_large_step_52000"
METRICS_CSV = RESULT_DIR / "metrics.csv"
OUTPUT = RESULT_DIR / "astroclip_vs_ours_gzd5_radar.png"

QUESTIONS = [
    "smooth",
    "disk-edge-on",
    "spiral-arms",
    "bar",
    "bulge-size",
    "how-rounded",
    "edge-on-bulge",
    "spiral-winding",
    "spiral-arm-count",
    "merging",
]

# AstroCLIP morphology scores reported in the paper/table for GZD-5.
ASTROCLIP = {
    "smooth": {"accuracy": 0.83, "f1": 0.83},
    "disk-edge-on": {"accuracy": 0.97, "f1": 0.97},
    "spiral-arms": {"accuracy": 0.92, "f1": 0.94},
    "bar": {"accuracy": 0.56, "f1": 0.54},
    "bulge-size": {"accuracy": 0.79, "f1": 0.78},
    "how-rounded": {"accuracy": 0.74, "f1": 0.74},
    "edge-on-bulge": {"accuracy": 0.82, "f1": 0.81},
    "spiral-winding": {"accuracy": 0.74, "f1": 0.68},
    "spiral-arm-count": {"accuracy": 0.44, "f1": 0.41},
    "merging": {"accuracy": 0.80, "f1": 0.73},
}


def load_ours() -> dict[str, dict[str, float]]:
    with METRICS_CSV.open() as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["question"]: {
            "accuracy": float(row["accuracy"]),
            "f1": float(row["f1"]),
        }
        for row in rows
    }


def closed(values: list[float]) -> list[float]:
    return values + values[:1]


def plot_panel(ax, metric: str, ours: dict[str, dict[str, float]]) -> None:
    angles = np.linspace(0, 2 * np.pi, len(QUESTIONS), endpoint=False).tolist()
    angles_closed = closed(angles)
    astro_values = closed([ASTROCLIP[q][metric] for q in QUESTIONS])
    ours_values = closed([ours[q][metric] for q in QUESTIONS])

    ax.plot(angles_closed, astro_values, color="#1f77b4", linewidth=2.2, label="AstroCLIP paper")
    ax.fill(angles_closed, astro_values, color="#1f77b4", alpha=0.12)
    ax.plot(angles_closed, ours_values, color="#d62728", linewidth=2.2, label="Our ViT-L/14")
    ax.fill(angles_closed, ours_values, color="#d62728", alpha=0.12)

    ax.set_xticks(angles)
    ax.set_xticklabels(QUESTIONS, fontsize=8)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
    ax.set_title(metric.upper() if metric == "f1" else "Accuracy", fontsize=13, pad=18)
    ax.grid(alpha=0.35)


def main() -> None:
    ours = load_ours()
    fig, axes = plt.subplots(
        1,
        2,
        subplot_kw={"projection": "polar"},
        figsize=(14, 7),
        dpi=220,
        constrained_layout=True,
    )
    plot_panel(axes[0], "accuracy", ours)
    plot_panel(axes[1], "f1", ours)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.suptitle("GZD-5 Morphology: AstroCLIP Paper vs Our Frozen ViT-L/14", fontsize=15)
    fig.savefig(OUTPUT, bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()

