#!/usr/bin/env python3
"""Create compact tables and paper figures from the completed study results."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
ORDER = ["J_I", "J_S", "C_I", "C_S"]
DISPLAY = ["JEPA image", "JEPA spectrum", "CLIP image", "CLIP spectrum"]
TARGETS = ["redshift", "stellar_mass", "metallicity", "age", "ssfr"]
TARGET_LABELS = ["Redshift", "Stellar mass", "Metallicity", "Stellar age", "sSFR"]
COLORS = {"JEPA": "#176B87", "CLIP": "#C23B22"}


def load(name):
    return json.loads((RESULTS / name).read_text())


def write_csv(name, rows):
    path = TABLES / name
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def heatmap(matrix, title, path, vmin=0, vmax=1, xlabels=DISPLAY, ylabels=None):
    if ylabels is None:
        ylabels = xlabels
    fig, axis = plt.subplots(figsize=(7.2, 6.1))
    image = axis.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] < (vmin + vmax) / 2 else "black"
            axis.text(column, row, f"{matrix[row, column]:.3f}", ha="center", va="center", color=color, fontsize=9)
    axis.set_xticks(np.arange(len(xlabels)), xlabels)
    axis.set_yticks(np.arange(len(ylabels)), ylabels)
    figure_colorbar = fig.colorbar(image, ax=axis)
    figure_colorbar.set_label("Similarity")
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=25)
    axis.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(FIGURES / path, dpi=200)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    geometry = load("geometry.json")
    retrieval = {model: load(f"retrieval_{model}.json") for model in ("jepa", "clip")}
    mappings = {model: load(f"mappings_{model}.json") for model in ("jepa", "clip")}
    probes = {name: load(f"probes_{name}.json") for name in ORDER}
    transfers = {model: load(f"decoder_transfer_{model}.json") for model in ("jepa", "clip")}
    ranks = {name: load(f"rank_geometry_{name}.json") for name in ORDER}

    heatmap(np.array(geometry["cka"]["projected_full"]), "Projected-space linear CKA (168,280 objects)", "02_cka_projected.png")
    heatmap(np.array(geometry["cka"]["raw_test"]), "Raw-backbone linear CKA (29,697 test objects)", "03_cka_raw.png")
    heatmap(np.array(geometry["distance_spearman_projected"]), "Projected pairwise cosine-geometry Spearman correlation", "04_distance_spearman.png", vmin=0)

    retrieval_rows = []
    for model in ("jepa", "clip"):
        for scope, directions in retrieval[model]["scopes"].items():
            for direction, metrics in directions.items():
                retrieval_rows.append({"model": model, "scope": scope, "direction": direction, **metrics})
    write_csv("retrieval.csv", retrieval_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for axis, scope in zip(axes, ("test_29697", "all_168280")):
        x = np.arange(3)
        width = 0.36
        for offset, model in enumerate(("jepa", "clip")):
            values = []
            for metric in ("recall_at_1", "recall_at_5", "recall_at_10"):
                dirs = retrieval[model]["scopes"][scope]
                values.append(np.mean([entry[metric] for entry in dirs.values()]))
            axis.bar(x + (offset - 0.5) * width, values, width, label=model.upper(), color=COLORS[model.upper()])
        axis.set_xticks(x, ["R@1", "R@5", "R@10"])
        axis.set_title(scope.replace("_", " "))
        axis.set_ylabel("Bidirectional mean recall")
    axes[0].legend(frameon=False)
    fig.suptitle("Exact paired retrieval")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_retrieval.png", dpi=200)
    plt.close(fig)

    overlap_rows = []
    cross_pairs = {"JEPA": (0, 1), "CLIP": (2, 3)}
    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    for model, (i, j) in cross_pairs.items():
        values = []
        for k in (10, 50, 100):
            value = geometry["neighborhood_overlap_test"][str(k)][i][j]
            values.append(value)
            overlap_rows.append({"model": model.lower(), "k": k, "overlap_fraction": value})
        axis.plot((10, 50, 100), values, marker="o", linewidth=2, color=COLORS[model], label=model)
    axis.set_xlabel("Neighborhood size k")
    axis.set_ylabel("Cross-modal neighbor overlap")
    axis.set_title("Local manifold agreement on the test set")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "05_neighborhood_overlap.png", dpi=200)
    plt.close(fig)
    write_csv("neighborhood_overlap.csv", overlap_rows)

    mapping_rows = []
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    labels, jepa_values, clip_values = [], [], []
    for direction in ("image_to_spectrum", "spectrum_to_image"):
        labels.append(direction.replace("_to_", " -> ").replace("image", "I").replace("spectrum", "S"))
        for model, target in (("jepa", jepa_values), ("clip", clip_values)):
            item = mappings[model]["ridge"][direction]
            target.append(item["global_r2"])
            mapping_rows.append({"model": model, "direction": direction, **{k: item[k] for k in ("global_r2", "uniform_mean_dimension_r2", "explained_variance_weighted", "mean_cosine")}})
    x = np.arange(2); width = 0.36
    axis.bar(x - width/2, jepa_values, width, label="JEPA", color=COLORS["JEPA"])
    axis.bar(x + width/2, clip_values, width, label="CLIP", color=COLORS["CLIP"])
    axis.set_xticks(x, labels); axis.set_ylim(0, 0.85)
    axis.set_ylabel("Held-out global R2")
    axis.set_title("Linear cross-modal predictability")
    axis.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "06_linear_predictability.png", dpi=200); plt.close(fig)
    write_csv("linear_predictability.csv", mapping_rows)

    probe_rows = []
    for kind in ("raw", "projected"):
        matrix = np.empty((4, 5))
        for i, name in enumerate(ORDER):
            rep = probes[name]["representations"][kind]
            for j, target in enumerate(TARGETS):
                result = rep["redshift"] if target == "redshift" else rep["properties"][target]
                score = result["summary"]["r2"]["mean"]
                std = result["summary"]["r2"]["std"]
                matrix[i, j] = score
                probe_rows.append({"space": name, "representation": kind, "target": target, "r2_mean": score, "r2_std": std})
        heatmap(matrix, f"Matched downstream information: {kind}", f"07_downstream_{kind}.png", vmin=0.2, vmax=0.9, xlabels=TARGET_LABELS, ylabels=DISPLAY)
    write_csv("downstream_probes.csv", probe_rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for axis, direction in zip(axes, ("image_to_spectrum", "spectrum_to_image")):
        x = np.arange(2); width = 0.34
        for offset, model in enumerate(("jepa", "clip")):
            item = mappings[model]["procrustes"][direction]
            vals = [item["before"]["recall_at_1"], item["after"]["recall_at_1"]]
            axis.bar(x + (offset - .5)*width, vals, width, color=COLORS[model.upper()], label=model.upper())
        axis.set_xticks(x, ["Before", "After rotation"])
        axis.set_title(direction.replace("_", " "))
        axis.set_ylabel("Test R@1")
    axes[0].legend(frameon=False)
    fig.suptitle("Held-out orthogonal Procrustes retrieval")
    fig.tight_layout(); fig.savefig(FIGURES / "08_procrustes_retrieval.png", dpi=200); plt.close(fig)

    transfer_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for axis, decoder in zip(axes, ("image_decoder", "spectrum_decoder")):
        x = np.arange(len(TARGETS)); width = 0.36
        for offset, model in enumerate(("jepa", "clip")):
            values = []
            for target in TARGETS:
                item = transfers[model]["targets"][target][decoder]
                key = "procrustes_spectrum_to_image" if decoder == "image_decoder" else "procrustes_image_to_spectrum"
                own_key = "own_image" if decoder == "image_decoder" else "own_spectrum"
                values.append(item[key]["r2"])
                transfer_rows.append({"model": model, "decoder": decoder, "target": target, "own_r2": item[own_key]["r2"], "direct_transfer_r2": item["direct_spectrum" if decoder == "image_decoder" else "direct_image"]["r2"], "procrustes_transfer_r2": item[key]["r2"]})
            axis.bar(x + (offset-.5)*width, values, width, color=COLORS[model.upper()], label=model.upper())
        axis.set_xticks(x, TARGET_LABELS, rotation=25, ha="right")
        axis.set_title(decoder.replace("_", " "))
        axis.set_ylabel("Procrustes-transfer R2")
    axes[0].legend(frameon=False)
    fig.suptitle("Cross-modal transfer of physical decoding directions")
    fig.tight_layout(); fig.savefig(FIGURES / "09_decoder_transfer.png", dpi=200); plt.close(fig)
    write_csv("decoder_transfer.csv", transfer_rows)

    rank_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, kind in zip(axes, ("raw", "projected")):
        for name, label in zip(ORDER, DISPLAY):
            item = ranks[name]["representations"][kind]
            eig = np.array(item["eigenvalues_ascending"])[::-1]
            axis.plot(np.arange(1, len(eig)+1), np.cumsum(eig)/eig.sum(), label=label)
            rank_rows.append({"space": name, "representation": kind, **{k: item[k] for k in ("feature_dim", "effective_rank", "participation_ratio", "largest_eigenvalue_share")}})
        axis.set_xscale("log"); axis.set_xlabel("Leading dimensions"); axis.set_ylabel("Cumulative variance"); axis.set_title(kind)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Representation eigenspectra")
    fig.tight_layout(); fig.savefig(FIGURES / "10_eigenspectra.png", dpi=200); plt.close(fig)
    write_csv("rank_geometry.csv", rank_rows)

    cka_rows = []
    for scope, matrix in geometry["cka"].items():
        for i, left in enumerate(ORDER):
            for j, right in enumerate(ORDER):
                cka_rows.append({"scope": scope, "left": left, "right": right, "cka": matrix[i][j]})
    write_csv("cka.csv", cka_rows)
    print(f"Wrote figures to {FIGURES} and tables to {TABLES}")


if __name__ == "__main__":
    main()
