#!/usr/bin/env python
"""Collapse anatomy for the spectra backbone: where does rank/information die?

Given a checkpoint, embeds the scan-protocol subsample and measures effective
rank + ridge redshift validation R2 for every representation the encoder
exposes:

  - every transformer layer, CLS token and masked-mean pooled
  - the final (post-norm) embedding, both poolings
  - the 64-d projection head output (the space LeJEPA's SIGReg regularizes)
  - the final embedding under a training-style contiguous global crop
    (PAD-masked window) instead of the full all-real spectrum

plus a PC-vs-covariate analysis of the probe embedding: correlations of the
top principal components with redshift and per-spectrum brightness
(median flux) / variability (flux std). If the leading PCs track brightness,
pretraining on raw un-normalized flux spent the embedding's capacity on
amplitude.

Outputs results/<label>/anatomy.json, layer_anatomy.png, pc_covariates.png
and console verdicts.

Run:  python collapse_anatomy.py --checkpoint checkpoints/spectra.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
EVALS_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = EVALS_DIR / "desi_crossmatch" / "data" / "astroclip"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


eh = load_module("embedding_health", SCRIPT_DIR / "embedding_health.py")
sp = eh.sp
acp = eh.acp
rp = eh.rp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--subsample", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--crop-scale",
        type=float,
        nargs=2,
        default=(0.90, 1.0),
        metavar=("LOW", "HIGH"),
        help="training-style global crop range (fraction of patches kept)",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def masked_mean(hidden: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
    """Mean over real patch tokens (mirrors the encoder's masked_mean pooling)."""
    patch_hidden = hidden[:, 1:]
    weights = token_mask[:, 1:].to(dtype=patch_hidden.dtype).unsqueeze(-1)
    return (patch_hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)


@torch.inference_mode()
def anatomy_pass(
    encoder: torch.nn.Module,
    raw: np.ndarray,
    batch_size: int,
    device: torch.device,
    crop_scale: tuple[float, float] | None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """One pass over the subsample; returns representation name -> (K, D) array.

    crop_scale=None feeds the full all-real spectrum (eval protocol);
    otherwise each spectrum gets a training-style contiguous PAD-masked
    window with a kept fraction drawn from crop_scale.
    """
    patch_size, num_patches = encoder.patch_size, encoder.num_patches
    usable = patch_size * num_patches
    layers = list(encoder.transformer.layers)
    captured: list[torch.Tensor] = []
    hooks = [
        layer.register_forward_hook(lambda m, i, o: captured.append(o))
        for layer in layers
    ]
    rng = np.random.RandomState(seed)
    storage: dict[str, list[torch.Tensor]] = {}

    def store(name: str, value: torch.Tensor) -> None:
        storage.setdefault(name, []).append(value.float().cpu())

    try:
        for start in range(0, len(raw), batch_size):
            array = raw[start : start + batch_size]
            crops = (
                torch.from_numpy(array[:, :usable])
                .reshape(-1, 1, num_patches, patch_size)
                .to(device)
            )
            masks = torch.ones(
                crops.shape[0], 1, num_patches, dtype=torch.float32, device=device
            )
            if crop_scale is not None:
                for row in range(crops.shape[0]):
                    fraction = rng.uniform(*crop_scale)
                    length = max(1, min(num_patches, int(round(num_patches * fraction))))
                    window_start = rng.randint(0, num_patches - length + 1)
                    masks[row] = 0.0
                    masks[row, 0, window_start : window_start + length] = 1.0
            amp = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            captured.clear()
            with amp:
                emb, proj = encoder._encode_views(crops, masks)
            flat_masks = masks.flatten(0, 1).bool()
            cls_mask = torch.ones(
                flat_masks.shape[0], 1, dtype=torch.bool, device=device
            )
            token_mask = torch.cat([cls_mask, flat_masks], dim=1)
            for index, hidden in enumerate(captured):
                store(f"layer{index + 1:02d}_cls", hidden[:, 0])
                store(f"layer{index + 1:02d}_mean", masked_mean(hidden, token_mask))
            final_hidden = encoder.norm(captured[-1])
            store("final_cls", final_hidden[:, 0])
            store("final_mean", masked_mean(final_hidden, token_mask))
            store("proj", proj[0])
    finally:
        for hook in hooks:
            hook.remove()
    return {name: torch.cat(chunks).numpy() for name, chunks in storage.items()}


def representation_report(
    features: np.ndarray,
    targets: np.ndarray,
    fit_rows: np.ndarray,
    val_rows: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    """`targets` must be row-aligned with `features` (subset order)."""
    stats = eh.spectrum_stats(torch.from_numpy(features))
    return {
        "dimension": stats["dimension"],
        "effective_rank": round(stats["effective_rank"], 2),
        "rankme": round(stats["rankme"], 2),
        "ridge_val_r2": round(
            eh.ridge_r2(features, targets, fit_rows, val_rows, device), 4
        ),
    }


def pc_covariates(
    features: np.ndarray, covariates: dict[str, np.ndarray], top: int = 5
) -> dict[str, Any]:
    centered = torch.from_numpy(features - features.mean(axis=0))
    _, _, v = torch.pca_lowrank(centered, q=top, center=False, niter=4)
    scores = (centered @ v).numpy()
    correlations = {
        name: [
            float(np.corrcoef(scores[:, pc], values)[0, 1]) for pc in range(top)
        ]
        for name, values in covariates.items()
    }
    return {"scores": scores, "correlations": correlations}


def plot_layer_anatomy(
    reports: dict[str, dict[str, float]], depth: int, output_dir: Path
) -> None:
    layer_axis = list(range(1, depth + 1)) + [depth + 1]

    def series(pooling: str, key: str) -> list[float]:
        values = [
            reports[f"layer{index:02d}_{pooling}"][key] for index in range(1, depth + 1)
        ]
        return values + [reports[f"final_{pooling}"][key]]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for ax, key, title in zip(
        axes,
        ("effective_rank", "ridge_val_r2"),
        ("effective rank", "ridge validation R²"),
    ):
        for pooling, color in (("cls", "#2a78d6"), ("mean", "#eb6834")):
            ax.plot(
                layer_axis,
                series(pooling, key),
                marker="o",
                ms=3.5,
                lw=1.4,
                color=color,
                label=f"{pooling} pooling",
            )
        ax.scatter(
            [depth + 1],
            [reports["proj"][key]],
            marker="*",
            s=90,
            color="#1baf7a",
            zorder=3,
            label="projection (64-d)",
        )
        ax.set_xticks(layer_axis)
        ax.set_xticklabels([str(i) for i in range(1, depth + 1)] + ["final"])
        ax.set_xlabel("transformer layer")
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(color="#e5e4e0", lw=0.7)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8.5)
    r2_floor = min(
        min(series("cls", "ridge_val_r2")),
        min(series("mean", "ridge_val_r2")),
        reports["proj"]["ridge_val_r2"],
    )
    axes[1].set_ylim(min(0.0, r2_floor - 0.05), 1.0)
    fig.suptitle(
        "Where does rank/information die? — per-layer representations",
        x=0.02,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_dir / "layer_anatomy.png", dpi=200)
    plt.close(fig)


def plot_pc_covariates(
    analysis: dict[str, Any], covariates: dict[str, np.ndarray], output_dir: Path
) -> None:
    scores = analysis["scores"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    for ax, (name, values) in zip(axes, list(covariates.items())[:2]):
        color_values = (
            np.log10(np.clip(values, 1e-3, None)) if "flux" in name else values
        )
        scatter = ax.scatter(
            scores[:, 0], scores[:, 1], c=color_values, s=3, cmap="viridis"
        )
        fig.colorbar(scatter, ax=ax, label=("log10 " if "flux" in name else "") + name)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        correlation = analysis["correlations"][name][0]
        ax.set_title(
            f"embedding PC1/PC2 colored by {name} — corr(PC1, {name}) = "
            f"{correlation:+.2f}",
            fontsize=9.5,
            loc="left",
        )
    fig.tight_layout()
    fig.savefig(output_dir / "pc_covariates.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    label = args.label or "_".join(
        [args.checkpoint.resolve().parent.name, args.checkpoint.stem]
    )
    output_dir = args.output_dir.resolve() / label
    output_dir.mkdir(parents=True, exist_ok=True)

    redshifts, _, is_test = acp.read_targets(args.data_dir)
    train_idx = np.flatnonzero(~is_test)
    rng = np.random.RandomState(42)
    subsample = rng.permutation(train_idx)[: args.subsample]
    val_count = max(1, int(0.1 * len(subsample)))
    val_idx, fit_idx = subsample[:val_count], subsample[val_count:]
    keep = np.zeros(len(redshifts), dtype=bool)
    keep[subsample] = True
    kept_order = np.flatnonzero(keep)
    position = {int(g): i for i, g in enumerate(kept_order)}
    fit_rows = np.array([position[int(g)] for g in fit_idx])
    val_rows = np.array([position[int(g)] for g in val_idx])

    train_files, _ = acp.split_files(args.data_dir)
    print(f"reading {len(subsample)} raw train spectra ...")
    raw = eh.read_subsampled_spectra(train_files, keep, args.batch_size)

    encoder, meta = sp.load_encoder(args.checkpoint, device)
    depth = len(encoder.transformer.layers)

    print("full-spectrum pass (eval protocol) ...")
    representations = anatomy_pass(
        encoder, raw, args.batch_size, device, crop_scale=None
    )
    print("training-style crop pass ...")
    cropped = anatomy_pass(
        encoder, raw, args.batch_size, device, crop_scale=tuple(args.crop_scale)
    )
    representations["crop_final_cls"] = cropped["final_cls"]
    representations["crop_final_mean"] = cropped["final_mean"]
    del cropped

    sub_redshifts = redshifts[kept_order]
    reports = {}
    for name, features in representations.items():
        reports[name] = representation_report(
            features, sub_redshifts, fit_rows, val_rows, device
        )
        print(
            f"{name:16s} rank {reports[name]['effective_rank']:7.1f} / "
            f"{reports[name]['dimension']:4d} | ridge val R2 "
            f"{reports[name]['ridge_val_r2']:.4f}"
        )
    plot_layer_anatomy(reports, depth, output_dir)

    covariates = {
        "median_flux": np.median(raw, axis=1),
        "flux_std": raw.std(axis=1),
        "redshift": redshifts[kept_order],
    }
    analysis = pc_covariates(representations["final_cls"], covariates)
    plot_pc_covariates(analysis, covariates, output_dir)

    verdicts = []
    if (
        reports["final_mean"]["ridge_val_r2"]
        > reports["final_cls"]["ridge_val_r2"] + 0.05
    ):
        verdicts.append(
            "masked-mean pooling beats CLS — patch tokens are healthier than "
            "the CLS token; re-probe with masked_mean for a free gain."
        )
    if reports["proj"]["effective_rank"] > 0.5 * reports["proj"]["dimension"] > 0 and (
        reports["final_cls"]["effective_rank"]
        < 0.1 * reports["final_cls"]["dimension"]
    ):
        verdicts.append(
            "SIGReg kept the 64-d projection healthy while the 768-d backbone "
            "embedding collapsed — regularization never reached the probed space."
        )
    brightness = max(
        abs(c) for c in analysis["correlations"]["median_flux"][:2]
    )
    if brightness > 0.6:
        verdicts.append(
            f"leading PCs track brightness (|corr| {brightness:.2f}) — raw-flux "
            "amplitude dominates the embedding; normalize spectra in pretraining."
        )
    crop_gain = (
        reports["crop_final_cls"]["ridge_val_r2"]
        - reports["final_cls"]["ridge_val_r2"]
    )
    if crop_gain > 0.05:
        verdicts.append(
            f"training-style crops probe {crop_gain:+.3f} better than the full "
            "spectrum — the all-real eval input is out-of-distribution for the "
            "encoder."
        )
    if not verdicts:
        verdicts.append(
            "no single culprit stands out — compare layer_anatomy.png curves "
            "for where rank decays."
        )
    for verdict in verdicts:
        print(f"VERDICT: {verdict}")

    rp.atomic_json(
        output_dir / "anatomy.json",
        {
            "label": label,
            "checkpoint_metadata": meta,
            "subsample": int(len(subsample)),
            "crop_scale": list(args.crop_scale),
            "representations": reports,
            "pc_covariate_correlations": analysis["correlations"],
            "verdicts": verdicts,
        },
    )
    print(f"wrote {output_dir}/anatomy.json, layer_anatomy.png, pc_covariates.png")


if __name__ == "__main__":
    main()
