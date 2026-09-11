#!/usr/bin/env python3
"""Generate the two vector figures used in the AstroJEPA-centered paper."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Figures"
OUT.mkdir(exist_ok=True)

NAVY = "#183B56"
BLUE = "#3C78A8"
PALE_BLUE = "#E8F1F7"
ORANGE = "#D97732"
PALE_ORANGE = "#FCEBDD"
GREEN = "#4C956C"
PALE_GREEN = "#E5F3EA"
GRAY = "#58636D"
LIGHT = "#F5F7F9"


def rounded(ax, xy, width, height, text, face, edge, *, size=7.0,
            weight="normal", linestyle="-", text_color=NAVY, zorder=2):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.05, edgecolor=edge, facecolor=face,
        linestyle=linestyle, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=size, color=text_color,
            weight=weight, linespacing=1.08, zorder=zorder + 1)
    return patch


def arrow(ax, start, end, *, color=GRAY, width=1.05, style="-|>",
          mutation=8, connection="arc3"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=mutation,
        linewidth=width, color=color, connectionstyle=connection, zorder=1,
    ))


def architecture_figure():
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.52))
    fig.patch.set_facecolor("white")

    panel_specs = [
        ("A", "Pretrain $\u2192$ align", "Frozen unimodal backbones; train 2.23M pooler parameters", True),
        ("B", "Joint from scratch", "Train both backbones and projectors end to end (392M)", False),
    ]

    for ax, (letter, title, subtitle, sequential) in zip(axes, panel_specs):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch(
            (0.005, 0.01), 0.99, 0.97,
            boxstyle="round,pad=0.01,rounding_size=0.025",
            linewidth=0.9, edgecolor="#CBD3D9", facecolor="white",
        ))
        ax.text(0.035, 0.925, letter, fontsize=10.5, weight="bold", color="white",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.24", facecolor=NAVY, edgecolor=NAVY))
        ax.text(0.095, 0.938, title, fontsize=9.5, weight="bold", color=NAVY,
                ha="left", va="center")
        ax.text(0.095, 0.883, subtitle, fontsize=6.4, color=GRAY,
                ha="left", va="center")

        lane_y = [0.61, 0.31]
        lane_names = ["image\nviews", "spectrum\nviews"]
        enc_names = ["ViT-L/14\n$f_I$", "12-layer Tx\n$f_S$"]
        pool_names = (["query pooler\n$g_I$", "query pooler\n$g_S$"] if sequential
                      else ["MLP projector\n$g_I$", "MLP projector\n$g_S$"])
        colors = [(PALE_BLUE, BLUE), (PALE_GREEN, GREEN)]

        for y, lane, enc, pool, (pale, color) in zip(
                lane_y, lane_names, enc_names, pool_names, colors):
            rounded(ax, (0.035, y), 0.15, 0.14, lane, pale, color, size=6.8, weight="bold")
            if sequential:
                rounded(ax, (0.255, y), 0.19, 0.14, enc, LIGHT, BLUE,
                        size=6.8, weight="bold", linestyle="--")
            else:
                rounded(ax, (0.255, y), 0.19, 0.14, enc, PALE_ORANGE, ORANGE,
                        size=6.8, weight="bold")
            rounded(ax, (0.52, y), 0.20, 0.14, pool, PALE_ORANGE, ORANGE,
                    size=6.6, weight="bold")
            rounded(ax, (0.80, y), 0.14, 0.14,
                    "$z_I$" if "image" in lane else "$z_S$",
                    "#F1ECF8", "#70569A", size=9.0, weight="bold")
            arrow(ax, (0.185, y + 0.07), (0.255, y + 0.07))
            arrow(ax, (0.445, y + 0.07), (0.52, y + 0.07))
            arrow(ax, (0.72, y + 0.07), (0.80, y + 0.07))
            ax.text(0.475, y + 0.115, "$h_I$" if "image" in lane else "$h_S$",
                    fontsize=6.6, color=GRAY, ha="center")

        if sequential:
            ax.text(0.350, 0.805, "unimodal SSL checkpoints", fontsize=6.3,
                    color=BLUE, ha="center", weight="bold")
            ax.text(0.350, 0.575, "frozen", fontsize=5.8, color=BLUE,
                    ha="center", va="top")
            ax.text(0.350, 0.275, "frozen", fontsize=5.8, color=BLUE,
                    ha="center", va="top")
        else:
            ax.text(0.475, 0.805, "random initialization", fontsize=6.3,
                    color=ORANGE, ha="center", weight="bold")
            ax.add_patch(FancyBboxPatch(
                (0.235, 0.285), 0.505, 0.485,
                boxstyle="round,pad=0.008,rounding_size=0.018",
                linewidth=0.8, edgecolor=ORANGE, facecolor="none", linestyle=":",
            ))
            ax.text(0.488, 0.277, "all parameters updated", fontsize=5.8,
                    color=ORANGE, ha="center", va="top")

        arrow(ax, (0.955, 0.675), (0.955, 0.375), color="#70569A",
              width=1.25, style="<->", mutation=8)
        ax.text(0.982, 0.525, "match 4\nview pairs", fontsize=5.6,
                color="#70569A", ha="center", va="center", rotation=90)
        rounded(ax, (0.25, 0.055), 0.55, 0.105,
                r"$0.95\,\mathcal{L}_{\rm inv}+0.05\,\mathcal{R}_{\rm SIG}$",
                "#FAF7FC", "#B9A9CF", size=7.4, weight="bold")

    fig.suptitle(
        "AstroJEPA learns a shared image--spectrum space while retaining modality-specific backbone states",
        x=0.5, y=1.015, fontsize=9.5, weight="bold", color=NAVY,
    )
    fig.subplots_adjust(left=0.008, right=0.992, top=0.90, bottom=0.01, wspace=0.035)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"astrojepa_architecture.{suffix}", dpi=260,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def add_bar_labels(ax, bars, *, percent=False, digits=2):
    for bar in bars:
        value = bar.get_height()
        label = f"{value:.{digits}f}" + ("%" if percent else "")
        ax.annotate(label, (bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 2.2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=5.8, color=NAVY)


def training_comparison_figure():
    seq_img = [0.511772, 0.672491, 0.400697, 0.210183, 0.497528]
    seq_sp = [0.573527, 0.714941, 0.427080, 0.253727, 0.532742]
    joint_img = [0.613104, 0.768012, 0.459308, 0.278290, 0.563879]
    joint_sp = [0.632093, 0.779902, 0.448309, 0.273529, 0.569867]

    fig, axes = plt.subplots(1, 4, figsize=(7.15, 2.18))
    fig.patch.set_facecolor("white")
    width = 0.34

    # Scientific utility of the shared, objective-facing embeddings.
    ax = axes[0]
    x = [0, 1]
    pre = [sum(seq_img) / 5, sum(seq_sp) / 5]
    joint = [sum(joint_img) / 5, sum(joint_sp) / 5]
    b1 = ax.bar([v - width / 2 for v in x], pre, width, color=BLUE, label="Pretrain $\u2192$ align")
    b2 = ax.bar([v + width / 2 for v in x], joint, width, color=ORANGE, label="Joint scratch")
    ax.set_xticks(x, ["Image", "Spectrum"])
    ax.set_ylim(0, 0.64)
    ax.set_ylabel("mean probe $R^2$", fontsize=7)
    ax.set_title("a  Scientific utility", loc="left", fontsize=7.5, weight="bold")
    add_bar_labels(ax, b1, digits=3)
    add_bar_labels(ax, b2, digits=3)

    # Global alignment measures.
    ax = axes[1]
    x = [0, 1]
    pre = [0.720, (0.5976 + 0.6267) / 2]
    joint = [0.770, (0.7492 + 0.7365) / 2]
    b1 = ax.bar([v - width / 2 for v in x], pre, width, color=BLUE)
    b2 = ax.bar([v + width / 2 for v in x], joint, width, color=ORANGE)
    ax.set_xticks(x, ["CKA", "map $R^2$"])
    ax.set_ylim(0, 0.9)
    ax.set_title("b  Global sharing", loc="left", fontsize=7.5, weight="bold")
    add_bar_labels(ax, b1, digits=3)
    add_bar_labels(ax, b2, digits=3)

    # Local and instance alignment.
    ax = axes[2]
    x = [0, 1]
    pre = [3.81, (0.8486 + 0.8923) / 2]
    joint = [10.49, (3.3067 + 2.6333) / 2]
    b1 = ax.bar([v - width / 2 for v in x], pre, width, color=BLUE)
    b2 = ax.bar([v + width / 2 for v in x], joint, width, color=ORANGE)
    ax.set_xticks(x, ["$k$NN@100", "R@10"])
    ax.set_ylim(0, 12.4)
    ax.set_ylabel("percent", fontsize=7)
    ax.set_title("c  Local / pair", loc="left", fontsize=7.5, weight="bold")
    add_bar_labels(ax, b1, percent=True, digits=2)
    add_bar_labels(ax, b2, percent=True, digits=2)

    # Effective rank of shared projections.
    ax = axes[3]
    x = [0, 1]
    pre = [8.8335, 8.6017]
    joint = [20.1938, 17.9436]
    b1 = ax.bar([v - width / 2 for v in x], pre, width, color=BLUE)
    b2 = ax.bar([v + width / 2 for v in x], joint, width, color=ORANGE)
    ax.set_xticks(x, ["Image", "Spectrum"])
    ax.set_ylim(0, 23.5)
    ax.set_ylabel("effective rank / 256", fontsize=7)
    ax.set_title("d  Shared capacity", loc="left", fontsize=7.5, weight="bold")
    add_bar_labels(ax, b1, digits=1)
    add_bar_labels(ax, b2, digits=1)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#9CA7AF")
        ax.tick_params(axis="both", labelsize=6.3, length=2.5, color="#9CA7AF")
        ax.grid(axis="y", color="#DDE2E6", linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE),
    ]
    fig.legend(handles, ["Pretrain $\u2192$ align", "Joint from scratch"],
               ncol=2, loc="lower center", frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.035))
    fig.subplots_adjust(left=0.065, right=0.992, top=0.89, bottom=0.23, wspace=0.46)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"astrojepa_training_comparison.{suffix}", dpi=260,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    architecture_figure()
    training_comparison_figure()
