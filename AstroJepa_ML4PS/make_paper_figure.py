"""Generate the compact, paper-specific alignment summary figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    blue, orange = "#2474A6", "#D45A2A"
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 1.82))

    # Bidirectional means on the 29,697-object candidate pool.
    recall_j = np.array([0.0039066, 0.0161801, 0.0297000]) * 100
    recall_c = np.array([0.0127117, 0.0476479, 0.0798229]) * 100
    x = np.arange(3)
    width = 0.34
    axes[0].bar(x - width / 2, recall_j, width, color=blue, label="JEPA+SIGReg")
    axes[0].bar(x + width / 2, recall_c, width, color=orange, label="CLIP")
    axes[0].set_xticks(x, ["R@1", "R@5", "R@10"])
    axes[0].set_ylabel("Bidirectional recall (%)")
    axes[0].set_title("(a) Exact pair retrieval")
    axes[0].set_ylim(0, 9.2)
    axes[0].legend(frameon=False, loc="upper left")

    geometry_j = [0.7705, 0.6073]
    geometry_c = [0.6338, 0.4845]
    x = np.arange(2)
    axes[1].bar(x - width / 2, geometry_j, width, color=blue)
    axes[1].bar(x + width / 2, geometry_c, width, color=orange)
    axes[1].set_xticks(x, ["Linear\nCKA", "Distance\nSpearman"])
    axes[1].set_ylabel("Cross-modal agreement")
    axes[1].set_title("(b) Global geometry")
    axes[1].set_ylim(0, 0.9)

    predict_j = [0.7492, 0.7365]
    predict_c = [0.5507, 0.5491]
    x = np.arange(2)
    axes[2].bar(x - width / 2, predict_j, width, color=blue)
    axes[2].bar(x + width / 2, predict_c, width, color=orange)
    axes[2].set_xticks(x, ["Image$\\to$spec.", "Spec.$\\to$image"])
    axes[2].set_ylabel("Held-out global $R^2$")
    axes[2].set_title("(c) Linear predictability")
    axes[2].set_ylim(0, 0.85)

    for ax in axes:
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5)

    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.23, top=0.87, wspace=0.42)
    out = ROOT / "Figures" / "alignment_tradeoffs.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
