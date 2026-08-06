#!/usr/bin/env python
"""Compare local eval results with published AION-1 / AstroCLIP numbers.

Reads the published baselines from competitor_baselines.json and the local
results produced by the other eval suites:

  galaxy10_checkpoint_evolution/results/best_checkpoints.json
  gzd5_morphology_probe/results/<model>/metrics.json
  redshift_regression/results/<model>/metrics.json

and writes, to --output-dir (default ./results):

  comparison.json           merged local + published numbers
  comparison_tables.md      one markdown table per task
  galaxy10_accuracy_vs_competitors.png
  gzd5_questions_vs_astroclip.png
  redshift_r2_vs_competitors.png   (only once the redshift probe has run)

Missing local results are reported as pending, never fatal, so this can be
re-run after every new eval to refresh the comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINES_PATH = SCRIPT_DIR / "competitor_baselines.json"

# Chart tokens: light surface, text inks, and a categorical order validated
# with the palette checker (blue/orange/aqua pass all-pairs CVD and
# normal-vision floors on this surface; published numbers stay neutral gray).
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e5e4e0"
COMPETITOR_GRAY = "#8a8983"
OURS_BLUE = "#2a78d6"
OURS_ORANGE = "#eb6834"
OURS_AQUA = "#1baf7a"

FAMILY_LABELS = {
    "VitLargePatch14_OfficialTrain5_Epoch5_2504": "Ours ViT-L/14",
    "VitSmallPatch14_2204": "Ours ViT-S/14",
    "checkpoints-test": "Ours ResNet9",
}
MODEL_LABELS = {
    "astro_vit_large_step_52000": "Ours ViT-L/14 (step 52000)",
    "astro_vit_small_step_21000": "Ours ViT-S/14 (step 21000)",
    "vit_large_adapted_xmatch": "Ours ViT-L/14 adapted (xmatch)",
    "vit_small_adapted_xmatch": "Ours ViT-S/14 adapted (xmatch)",
}
OURS_YELLOW = "#eda100"
MODEL_COLORS = {
    "astro_vit_large_step_52000": OURS_BLUE,
    "astro_vit_small_step_21000": OURS_ORANGE,
    "vit_large_adapted_xmatch": OURS_AQUA,
    "vit_small_adapted_xmatch": OURS_YELLOW,
}
GZD5_QUESTIONS = [
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
REDSHIFT_HEADS = ("ridge", "knn", "mlp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evals-root", type=Path, default=SCRIPT_DIR.parent)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    return parser.parse_args()


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Local result collectors — each returns [] when the eval has not run yet.
# ---------------------------------------------------------------------------


def collect_galaxy10(evals_root: Path) -> list[dict[str, Any]]:
    path = evals_root / "galaxy10_checkpoint_evolution/results/best_checkpoints.json"
    if not path.exists():
        return []
    rows = []
    for family, entry in json.loads(path.read_text()).items():
        label = FAMILY_LABELS.get(family, f"Ours {family}")
        rows.append(
            {
                "name": f"{label} ({entry['checkpoint'].removesuffix('.pt')})",
                "family": family,
                "head": "linear probe (LBFGS) on frozen embeddings",
                "head_short": "linear probe",
                "test_accuracy": entry["test_accuracy"]["mean"],
                "test_accuracy_std": entry["test_accuracy"]["std"],
                "test_macro_f1": entry["test_macro_f1"]["mean"],
                "test_macro_f1_std": entry["test_macro_f1"]["std"],
            }
        )
    rows += collect_galaxy10_matched_heads(evals_root)
    rows.sort(key=lambda row: row["test_accuracy"], reverse=True)
    return rows


MATCHED_HEAD_LABELS = {
    "aion_mlp": ("2-layer MLP (AION head repro)", "2-layer MLP, AION head"),
    "astroclip_mlp": (
        "4-layer MLP (AstroCLIP head repro)",
        "4-layer MLP, AstroCLIP head",
    ),
}


def collect_galaxy10_matched_heads(evals_root: Path) -> list[dict[str, Any]]:
    rows = []
    results_dir = evals_root / "galaxy10_matched_heads/results"
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        data = json.loads(metrics_path.read_text())
        label = metrics_path.parent.name
        for head, (head_name, head_short) in MATCHED_HEAD_LABELS.items():
            if head not in data["summary"]:
                continue
            test = data["summary"][head]["test"]
            rows.append(
                {
                    "name": MODEL_LABELS.get(label, f"Ours {label}"),
                    "family": label,
                    "head": head_name,
                    "head_short": head_short,
                    "test_accuracy": test["accuracy"]["mean"],
                    "test_accuracy_std": test["accuracy"]["std"],
                    "test_macro_f1": test["macro_f1"]["mean"],
                    "test_macro_f1_std": test["macro_f1"]["std"],
                }
            )
    return rows


def collect_gzd5(evals_root: Path) -> list[dict[str, Any]]:
    rows = []
    results_dir = evals_root / "gzd5_morphology_probe/results"
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        label = metrics_path.parent.name
        metrics = json.loads(metrics_path.read_text())["metrics"]
        per_question = {
            question: {
                "accuracy": values["Accuracy"],
                "f1": values["F1 Score"],
            }
            for question, values in metrics.items()
        }
        rows.append(
            {
                "name": MODEL_LABELS.get(label, f"Ours {label}"),
                "label": label,
                "head": "4-layer MLP on frozen embeddings (AstroCLIP protocol)",
                "per_question": per_question,
                "mean_accuracy": float(
                    np.mean([v["accuracy"] for v in per_question.values()])
                ),
                "mean_f1": float(np.mean([v["f1"] for v in per_question.values()])),
            }
        )
    return rows


def collect_redshift(evals_root: Path) -> list[dict[str, Any]]:
    rows = []
    results_dir = evals_root / "redshift_regression/results"
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        data = json.loads(metrics_path.read_text())
        label = metrics_path.parent.name
        for head in REDSHIFT_HEADS:
            row: dict[str, Any] = {
                "name": MODEL_LABELS.get(label, f"Ours {label}"),
                "label": label,
                "head": head,
                "sample": "Galaxy10 DECaLS",
            }
            for split in ("test", "test_clipped"):
                for metric in ("r2", "mae", "rmse", "nmad", "outlier_fraction"):
                    stats = data["summary"][head][split][metric]
                    row[f"{split}_{metric}"] = stats["mean"]
                    row[f"{split}_{metric}_std"] = stats["std"]
            rows.append(row)
    return rows


def collect_redshift_astroclip_sample(evals_root: Path) -> list[dict[str, Any]]:
    rows = []
    results_dir = evals_root / "desi_crossmatch/results"
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        data = json.loads(metrics_path.read_text())
        label = metrics_path.parent.name
        for head in REDSHIFT_HEADS:
            stats = data["summary"][head]["test"]["r2"]
            rows.append(
                {
                    "name": MODEL_LABELS.get(label, f"Ours {label}"),
                    "label": label,
                    "head": head,
                    "sample": "AstroCLIP sample",
                    "test_r2": stats["mean"],
                    "test_r2_std": stats["std"],
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def plot_galaxy10(
    ours: list[dict[str, Any]], baseline: dict[str, Any], output_dir: Path
) -> None:
    bars = [
        {
            "name": f"{c['name']} — 2-layer MLP (published)",
            "value": c["test_accuracy"],
            "std": None,
            "ours": False,
        }
        for c in baseline["competitors"]
    ] + [
        {
            "name": f"{row['name']} — {row['head_short']}",
            "value": row["test_accuracy"],
            "std": row["test_accuracy_std"],
            "ours": True,
        }
        for row in ours
    ]
    bars.sort(key=lambda bar: bar["value"])

    fig, ax = plt.subplots(figsize=(9.2, 0.52 * len(bars) + 1.8))
    fig.set_facecolor(SURFACE)
    style_axis(ax, grid_axis="x")
    positions = np.arange(len(bars))
    for position, bar in zip(positions, bars):
        color = OURS_BLUE if bar["ours"] else COMPETITOR_GRAY
        ax.barh(position, bar["value"], height=0.62, color=color, zorder=2)
        if bar["std"]:
            ax.errorbar(
                bar["value"],
                position,
                xerr=bar["std"],
                fmt="none",
                ecolor=TEXT_SECONDARY,
                elinewidth=1.0,
                capsize=2,
                zorder=3,
            )
        ax.text(
            bar["value"] + 0.012,
            position,
            f"{bar['value']:.3f}",
            va="center",
            fontsize=9,
            color=TEXT,
        )
    ax.set_yticks(positions)
    ax.set_yticklabels([bar["name"] for bar in bars], fontsize=9, color=TEXT)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Test accuracy", color=TEXT_SECONDARY, fontsize=10)
    fig.suptitle(
        "Galaxy10 / Galaxy Zoo 10 morphology — ours vs published\n"
        "(each bar labeled with its head; AION evaluates on a GZ10×DR10 cross-match)",
        x=0.02,
        ha="left",
        fontsize=11,
        color=TEXT,
    )
    ax.legend(
        handles=[
            Patch(color=OURS_BLUE, label="Ours (this repo)"),
            Patch(color=COMPETITOR_GRAY, label="Published"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output_dir / "galaxy10_accuracy_vs_competitors.png", dpi=200)
    plt.close(fig)


def plot_gzd5(
    ours: list[dict[str, Any]], baseline: dict[str, Any], output_dir: Path
) -> None:
    astroclip = baseline["competitors"][0]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
    fig.set_facecolor(SURFACE)
    for ax, metric_key, metric_name in zip(axes, ("accuracy", "f1"), ("Accuracy", "F1")):
        style_axis(ax)
        series = [
            (
                f"{row['name']} (mean {row[f'mean_{metric_key}']:.3f})",
                MODEL_COLORS.get(row["label"], OURS_AQUA),
                [row["per_question"].get(q, {}).get(metric_key, np.nan) for q in GZD5_QUESTIONS],
            )
            for row in ours
        ]
        series.append(
            (
                f"{astroclip['name']} (published, mean "
                f"{astroclip['mean_accuracy' if metric_key == 'accuracy' else 'mean_f1']:.3f})",
                COMPETITOR_GRAY,
                [astroclip["per_question"][q][metric_key] for q in GZD5_QUESTIONS],
            )
        )
        positions = np.arange(len(GZD5_QUESTIONS))
        width = 0.8 / len(series)
        for index, (label, color, values) in enumerate(series):
            offset = (index - (len(series) - 1) / 2) * width
            ax.bar(
                positions + offset,
                values,
                width=width * 0.92,
                color=color,
                label=label,
                zorder=2,
            )
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(metric_name, color=TEXT_SECONDARY, fontsize=10)
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.01),
            ncol=len(series),
            frameon=False,
            fontsize=8.5,
        )
    fig.suptitle(
        "GZD-5 question-wise morphology — ours vs AstroCLIP (same protocol: 4-layer MLP, debiased labels)",
        x=0.02,
        ha="left",
        fontsize=11,
        color=TEXT,
    )
    axes[1].set_xticks(np.arange(len(GZD5_QUESTIONS)))
    axes[1].set_xticklabels(GZD5_QUESTIONS, rotation=30, ha="right", color=TEXT, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "gzd5_questions_vs_astroclip.png", dpi=200)
    plt.close(fig)


def plot_redshift(
    ours: list[dict[str, Any]],
    baseline: dict[str, Any],
    output_dir: Path,
    matched_sample: bool = False,
) -> None:
    models = sorted({row["label"] for row in ours})
    by_model = {
        label: {row["head"]: row for row in ours if row["label"] == label}
        for label in models
    }

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    fig.set_facecolor(SURFACE)
    style_axis(ax)
    positions = np.arange(len(REDSHIFT_HEADS))
    width = 0.8 / max(len(models), 1)
    for index, label in enumerate(models):
        offset = (index - (len(models) - 1) / 2) * width
        values = [by_model[label].get(h, {}).get("test_r2", np.nan) for h in REDSHIFT_HEADS]
        errors = [by_model[label].get(h, {}).get("test_r2_std", 0.0) for h in REDSHIFT_HEADS]
        bar_positions = positions + offset
        ax.bar(
            bar_positions,
            values,
            width=width * 0.92,
            color=MODEL_COLORS.get(label, OURS_AQUA),
            label=MODEL_LABELS.get(label, f"Ours {label}"),
            zorder=2,
        )
        ax.errorbar(
            bar_positions,
            values,
            yerr=errors,
            fmt="none",
            ecolor=TEXT_SECONDARY,
            elinewidth=1.0,
            capsize=2,
            zorder=3,
        )
        for x, value in zip(bar_positions, values):
            if np.isfinite(value):
                ax.text(
                    x,
                    value + 0.02,
                    f"{value:.3f}",
                    ha="center",
                    fontsize=8.5,
                    color=TEXT,
                )

    image_refs = [
        c
        for c in baseline["competitors"]
        if c["input"] == "image" and "AstroCLIP" in c["name"]
    ]
    reference_handles = []
    for reference, linestyle in zip(image_refs, ("--", ":")):
        ax.axhline(
            reference["r2"], color=COMPETITOR_GRAY, linewidth=1.2, linestyle=linestyle
        )
        reference_handles.append(
            plt.Line2D(
                [],
                [],
                color=COMPETITOR_GRAY,
                linewidth=1.2,
                linestyle=linestyle,
                label=f"{reference['name']} (published) {reference['r2']:.2f}",
            )
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([h.upper() if h == "mlp" else h for h in REDSHIFT_HEADS], color=TEXT)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test R² (mean over 10 split seeds)", color=TEXT_SECONDARY, fontsize=10)
    subtitle = (
        "(same eval sample and split as AstroCLIP: DESI-LS × DESI cross-match)"
        if matched_sample
        else "(different eval samples: ours Galaxy10 DECaLS, AstroCLIP DESI cross-match)"
    )
    ax.set_title(
        "Redshift regression on frozen embeddings — ours vs AstroCLIP image encoder\n"
        + subtitle,
        color=TEXT,
        fontsize=11,
        loc="left",
    )
    bar_handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles=bar_handles + reference_handles,
        loc="upper left",
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "redshift_r2_vs_competitors.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def markdown_report(
    baselines: dict[str, Any],
    galaxy10: list[dict[str, Any]],
    gzd5: list[dict[str, Any]],
    redshift: list[dict[str, Any]],
) -> str:
    lines: list[str] = ["# Competitor comparison", ""]
    lines.append(
        "Published numbers are transcribed in `competitor_baselines.json`; "
        "local numbers are read from each eval suite's results. Protocol "
        "caveats per task are listed under each table — none of these are "
        "perfectly apples-to-apples."
    )

    lines += ["", "## Galaxy10 / Galaxy Zoo 10 morphology", ""]
    lines.append("| Model | Head | Test accuracy | Test macro-F1 | Source |")
    lines.append("|---|---|---:|---:|---|")
    for row in galaxy10:
        lines.append(
            f"| {row['name']} | {row['head']} | "
            f"{row['test_accuracy']:.4f} ± {row['test_accuracy_std']:.4f} | "
            f"{row['test_macro_f1']:.4f} ± {row['test_macro_f1_std']:.4f} | local |"
        )
    if not galaxy10:
        lines.append("| _pending_ | | | | local |")
    for c in baselines["galaxy10_morphology"]["competitors"]:
        lines.append(
            f"| {c['name']} | {c['head']} | {c['test_accuracy']:.3f} | n/r | published |"
        )
    lines += ["", f"Caveats: {baselines['galaxy10_morphology']['caveats']}"]

    lines += ["", "## GZD-5 question-wise morphology", ""]
    astroclip = baselines["gzd5_morphology"]["competitors"][0]
    header = "| Question | " + " | ".join(
        f"{row['name']} acc / F1" for row in gzd5
    ) + f" | {astroclip['name']} (published) acc / F1 |"
    lines.append(header)
    lines.append("|---|" + "---:|" * (len(gzd5) + 1))
    for question in GZD5_QUESTIONS:
        cells = []
        for row in gzd5:
            values = row["per_question"].get(question)
            cells.append(
                f"{values['accuracy']:.3f} / {values['f1']:.3f}" if values else "n/a"
            )
        published = astroclip["per_question"][question]
        cells.append(f"{published['accuracy']:.2f} / {published['f1']:.2f}")
        lines.append(f"| {question} | " + " | ".join(cells) + " |")
    mean_cells = [f"{row['mean_accuracy']:.3f} / {row['mean_f1']:.3f}" for row in gzd5]
    mean_cells.append(f"{astroclip['mean_accuracy']:.3f} / {astroclip['mean_f1']:.3f}")
    lines.append("| **mean** | " + " | ".join(mean_cells) + " |")
    if not gzd5:
        lines.append("")
        lines.append("_No local GZD-5 results found yet._")
    lines += ["", f"Caveats: {baselines['gzd5_morphology']['caveats']}"]

    lines += ["", "## Redshift regression", ""]
    lines.append(
        "| Model | Head / input | Sample | Test R² | Test R² (z < 0.25) | Source |"
    )
    lines.append("|---|---|---|---:|---:|---|")
    for row in redshift:
        clipped = (
            f"{row['test_clipped_r2']:.4f} ± {row['test_clipped_r2_std']:.4f}"
            if "test_clipped_r2" in row
            else "n/a"
        )
        lines.append(
            f"| {row['name']} | {row['head']} on frozen image embeddings | "
            f"{row['sample']} | "
            f"{row['test_r2']:.4f} ± {row['test_r2_std']:.4f} | {clipped} | local |"
        )
    if not redshift:
        lines.append(
            "| _pending — run `redshift_regression/redshift_probe.py` on the "
            "cluster_ | | | | | local |"
        )
    for c in baselines["redshift_regression"]["competitors"]:
        lines.append(
            f"| {c['name']} | {c['input']} | {c['eval_dataset']} | "
            f"{c['r2']:.2f} | n/r | published |"
        )
    lines += ["", f"Caveats: {baselines['redshift_regression']['caveats']}"]

    lines += ["", "## Physical property regression (reference targets)", ""]
    lines.append(
        "No local eval exists yet (requires PROVABGS labels). Published R² "
        "targets to beat once it does:"
    )
    lines.append("")
    lines.append("| Model | Input | Stellar mass | Age | Metallicity | sSFR-like |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for c in baselines["physical_properties"]["competitors"]:
        lines.append(
            f"| {c['name']} | {c['input']} | {c['stellar_mass']:.2f} | "
            f"{c['age']:.2f} | {c['metallicity']:.2f} | {c['ssfr_like']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    baselines = json.loads(BASELINES_PATH.read_text())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    galaxy10 = collect_galaxy10(args.evals_root)
    gzd5 = collect_gzd5(args.evals_root)
    redshift_galaxy10 = collect_redshift(args.evals_root)
    redshift_astroclip = collect_redshift_astroclip_sample(args.evals_root)
    redshift = redshift_galaxy10 + redshift_astroclip

    for task, rows in (
        ("galaxy10", galaxy10),
        ("gzd5", gzd5),
        ("redshift", redshift),
    ):
        status = f"{len(rows)} local result(s)" if rows else "pending (no local results)"
        print(f"{task}: {status}")

    comparison = {
        "galaxy10_morphology": {
            "ours": galaxy10,
            "published": baselines["galaxy10_morphology"],
        },
        "gzd5_morphology": {"ours": gzd5, "published": baselines["gzd5_morphology"]},
        "redshift_regression": {
            "ours": redshift,
            "published": baselines["redshift_regression"],
        },
        "physical_properties": {
            "ours": [],
            "published": baselines["physical_properties"],
        },
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "comparison_tables.md").write_text(
        markdown_report(baselines, galaxy10, gzd5, redshift)
    )

    if galaxy10:
        plot_galaxy10(galaxy10, baselines["galaxy10_morphology"], output_dir)
    if gzd5:
        plot_gzd5(gzd5, baselines["gzd5_morphology"], output_dir)
    chart_rows = redshift_astroclip or redshift_galaxy10
    if chart_rows:
        plot_redshift(
            chart_rows,
            baselines["redshift_regression"],
            output_dir,
            matched_sample=bool(redshift_astroclip),
        )

    print(f"wrote {output_dir}/comparison.json, comparison_tables.md and charts")


if __name__ == "__main__":
    main()
