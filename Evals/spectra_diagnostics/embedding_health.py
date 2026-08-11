#!/usr/bin/env python
"""Embedding-health diagnostics for the frozen spectra backbone.

Answers two questions about a low probe R2 before touching pretraining:

1. Did the embedding collapse? Spectrum statistics of the cached embeddings —
   effective rank (eigenvalue entropy), RankMe (singular-value entropy),
   participation ratio, dead-dimension count, and how much variance the top
   PCs hoard. A healthy 768-d embedding has effective rank well into the
   hundreds; tens indicate (partial) collapse.

2. Does the encoder destroy information? Ridge redshift R2 on the SAME train
   subsample and validation carve-out for three feature sets:
     raw flux (7780 features), PCA of raw flux at the embedding
     dimensionality, and the embedding itself.
   If raw flux or its PCA beats the embedding, the encoder is losing
   information the probes needed — a pretraining problem, not a probe
   problem.

Also writes a preprocessing sanity plot (first spectra with the patch grid
and the exact tensor the encoder sees) to rule out a silent eval-side
mismatch.

Requires the cached embeddings from a spectra_redshift run for --label, plus
the cross-match parquet shards.

Run:  python embedding_health.py --label checkpoints_spectra
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
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
EMBEDDINGS_DIR = EVALS_DIR / "spectra_redshift" / "results"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sp = load_module(
    "spectra_redshift_probe", EVALS_DIR / "spectra_redshift" / "spectra_probe.py"
)
acp = sp.acp
rp = sp.rp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        default="checkpoints_spectra",
        help="spectra_redshift results dir holding the cached embeddings.npy",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--embeddings-dir", type=Path, default=EMBEDDINGS_DIR)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--subsample", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 1. Embedding spectrum statistics
# ---------------------------------------------------------------------------


def spectrum_stats(embeddings: torch.Tensor) -> dict[str, Any]:
    x = embeddings.double()
    x = x - x.mean(dim=0)
    covariance = (x.T @ x) / (len(x) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0).flip(0)
    total = float(eigenvalues.sum())
    shares = (eigenvalues / total).numpy()

    # effective rank: exp of eigenvalue-share entropy (Roy & Vetterli)
    nonzero = shares[shares > 0]
    effective_rank = float(np.exp(-(nonzero * np.log(nonzero)).sum()))
    # RankMe uses singular-value shares (sqrt of eigenvalues)
    singular = np.sqrt(eigenvalues.numpy())
    singular_shares = singular / singular.sum()
    nonzero_s = singular_shares[singular_shares > 0]
    rankme = float(np.exp(-(nonzero_s * np.log(nonzero_s)).sum()))
    participation = float(eigenvalues.sum() ** 2 / (eigenvalues**2).sum())

    stds = embeddings.std(dim=0)
    dead = int((stds < 1e-3 * stds.max()).sum())
    cumulative = np.cumsum(shares)

    def share_top(k: int) -> float:
        return float(cumulative[min(k, len(cumulative)) - 1])

    return {
        "dimension": int(embeddings.shape[1]),
        "num_embeddings": int(len(embeddings)),
        "effective_rank": effective_rank,
        "rankme": rankme,
        "participation_ratio": participation,
        "dead_dimensions": dead,
        "variance_share_top1": share_top(1),
        "variance_share_top10": share_top(10),
        "variance_share_top50": share_top(50),
        "pcs_for_99pct_variance": int(np.searchsorted(cumulative, 0.99) + 1),
        "eigenvalue_shares": shares.tolist(),
    }


def plot_scree(stats: dict[str, Any], output_dir: Path) -> None:
    shares = np.array(stats["eigenvalue_shares"])
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.semilogy(np.arange(1, len(shares) + 1), shares, color="#2a78d6", lw=1.4)
    ax.axhline(1.0 / len(shares), color="#8a8983", ls="--", lw=1.0)
    ax.text(
        len(shares) * 0.99,
        1.0 / len(shares) * 1.2,
        "isotropic",
        ha="right",
        fontsize=8,
        color="#52514e",
    )
    ax.set_xlabel("principal component")
    ax.set_ylabel("variance share")
    ax.set_title(
        f"Embedding eigenspectrum — effective rank "
        f"{stats['effective_rank']:.1f} / {stats['dimension']}, "
        f"RankMe {stats['rankme']:.1f}",
        fontsize=10,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "eigenspectrum.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Information-floor control: ridge on raw flux / PCA / embedding
# ---------------------------------------------------------------------------


def read_subsampled_spectra(
    train_files: list[Path], keep: np.ndarray, batch_size: int
) -> np.ndarray:
    """Row-aligned read of the kept train spectra as one (K, L) array."""
    chunks = []
    offset = 0
    for array in sp.iter_spectrum_batches(train_files, batch_size):
        mask = keep[offset : offset + len(array)]
        offset += len(array)
        if mask.any():
            chunks.append(array[mask])
    return np.concatenate(chunks)


def ridge_r2(
    features: np.ndarray,
    targets: np.ndarray,
    fit_rows: np.ndarray,
    val_rows: np.ndarray,
    device: torch.device,
) -> float:
    """Best ridge validation R2. `targets` must be row-aligned with `features`
    (same length and order); fit_rows/val_rows index into both."""
    if len(targets) != len(features):
        raise ValueError(
            f"targets ({len(targets)}) not row-aligned with features "
            f"({len(features)})"
        )
    x = torch.from_numpy(features).float()
    mean = x[fit_rows].mean(dim=0)
    std = x[fit_rows].std(dim=0).clamp_min(1e-6)
    fit_x = ((x[fit_rows] - mean) / std).to(device)
    val_x = ((x[val_rows] - mean) / std).to(device)
    fit_y = torch.from_numpy(targets[fit_rows]).float().to(device)
    best = -np.inf
    for l2_value in rp.RIDGE_L2_GRID:
        weights, y_mean = rp.fit_ridge(fit_x, fit_y, l2_value)
        r2 = rp.regression_metrics(
            rp.predict_ridge(val_x, weights, y_mean), targets[val_rows]
        )["r2"]
        best = max(best, r2)
    return float(best)


def information_floor(
    embeddings: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
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
    raw = read_subsampled_spectra(train_files, keep, args.batch_size)

    dim = embeddings.shape[1]
    print("fitting PCA of raw flux at embedding dimensionality ...")
    centered = raw - raw[fit_rows].mean(axis=0)
    scale = raw[fit_rows].std(axis=0) + 1e-6
    whitened = torch.from_numpy((centered / scale).astype(np.float32)).to(device)
    # randomized SVD of the fit rows only; project everything onto its basis
    _, _, v = torch.pca_lowrank(whitened[fit_rows], q=dim, center=False, niter=4)
    pca = (whitened @ v).cpu().numpy()
    del whitened

    sub_embeddings = embeddings[torch.from_numpy(kept_order)].numpy()
    sub_redshifts = redshifts[kept_order]
    results = {
        "subsample": int(len(subsample)),
        "protocol": "ridge validation R2, seed-42 subsample, 10% carve-out "
        "(same as spectra_probe --scan)",
        "ridge_val_r2": {
            "embedding": ridge_r2(
                sub_embeddings, sub_redshifts, fit_rows, val_rows, device
            ),
            "raw_flux": ridge_r2(raw, sub_redshifts, fit_rows, val_rows, device),
            f"raw_flux_pca{dim}": ridge_r2(
                pca, sub_redshifts, fit_rows, val_rows, device
            ),
        },
    }
    return results


# ---------------------------------------------------------------------------
# 3. Preprocessing sanity plot
# ---------------------------------------------------------------------------


def plot_preprocessing(
    args: argparse.Namespace, embed_meta: dict[str, Any], output_dir: Path
) -> None:
    train_files, _ = acp.split_files(args.data_dir)
    array = next(sp.iter_spectrum_batches(train_files[:1], batch_size=2))
    patch_size, num_patches = 20, 389
    usable = patch_size * num_patches
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.6), sharex=True)
    for ax, spectrum in zip(axes, array[:2]):
        crops = spectrum[:usable].reshape(num_patches, patch_size)
        ax.plot(np.arange(len(spectrum)), spectrum, lw=0.4, color="#2a78d6")
        ax.plot(
            np.arange(usable),
            crops.reshape(-1),
            lw=0.4,
            color="#eb6834",
            alpha=0.6,
        )
        for boundary in range(0, usable + 1, patch_size * 50):
            ax.axvline(boundary, color="#e5e4e0", lw=0.6, zorder=0)
        dropped = len(spectrum) - usable
        ax.set_title(
            f"raw spectrum (blue) vs encoder input after patchify (orange) — "
            f"{num_patches}x{patch_size}, {dropped} value(s) dropped at the end",
            fontsize=9,
            loc="left",
        )
    axes[1].set_xlabel("flux index")
    fig.suptitle(
        f"Preprocessing check — checkpoint: {embed_meta.get('checkpoint', '?')}",
        x=0.02,
        ha="left",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_dir / "preprocessing_check.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cache = args.embeddings_dir / args.label / "embeddings.npy"
    meta_path = args.embeddings_dir / args.label / "embedding_metadata.json"
    if not cache.exists():
        raise SystemExit(
            f"No cached embeddings at {cache} — run the spectra_redshift probe first."
        )
    embeddings = torch.from_numpy(np.load(cache))
    embed_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    output_dir = args.output_dir.resolve() / args.label
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = spectrum_stats(embeddings)
    plot_scree(stats, output_dir)
    print(
        f"effective rank {stats['effective_rank']:.1f} / {stats['dimension']} | "
        f"RankMe {stats['rankme']:.1f} | participation {stats['participation_ratio']:.1f} | "
        f"dead dims {stats['dead_dimensions']} | "
        f"top-10 PCs hold {stats['variance_share_top10']:.1%} of variance"
    )

    floor = information_floor(embeddings, args, device)
    for name, value in floor["ridge_val_r2"].items():
        print(f"ridge validation R2 [{name}]: {value:.4f}")

    plot_preprocessing(args, embed_meta, output_dir)

    verdicts = []
    if stats["effective_rank"] < 0.1 * stats["dimension"]:
        verdicts.append(
            "LOW effective rank — embedding is (partially) collapsed."
        )
    baseline = max(
        value
        for name, value in floor["ridge_val_r2"].items()
        if name != "embedding"
    )
    if baseline > floor["ridge_val_r2"]["embedding"] + 0.05:
        verdicts.append(
            "Raw-flux control beats the embedding — the encoder is destroying "
            "information (pretraining problem, not probe problem)."
        )
    if not verdicts:
        verdicts.append(
            "No collapse signature and the embedding beats the raw-flux "
            "controls — the information is in the embedding; look at head "
            "capacity / probe protocol instead."
        )
    for verdict in verdicts:
        print(f"VERDICT: {verdict}")

    rp.atomic_json(
        output_dir / "diagnostics.json",
        {
            "label": args.label,
            "embedding_metadata": embed_meta,
            "spectrum_stats": {
                k: v for k, v in stats.items() if k != "eigenvalue_shares"
            },
            "information_floor": floor,
            "verdicts": verdicts,
        },
    )
    print(f"wrote {output_dir}/diagnostics.json, eigenspectrum.png, "
          "preprocessing_check.png")


if __name__ == "__main__":
    main()
