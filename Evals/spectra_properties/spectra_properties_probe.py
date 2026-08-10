#!/usr/bin/env python
"""PROVABGS physical-property probe for the frozen spectra backbone.

Regresses the four AstroCLIP galaxy properties — log stellar mass, log
mass-weighted metallicity, mass-weighted age, and log sSFR — from frozen
SpectrumTransformerEncoder embeddings of the AstroCLIP cross-match spectra,
with the same ridge / zero-shot kNN / MLP heads, seeds, and 80/20 split as
the redshift probes. Labels come from the PROVABGS BGS EDR posterior catalog
(the original DESI VAC HDF5 or its UniverseTBD/mmu_desi_provabgs parquet
conversion), joined by DESI targetid.

Reference: AstroCLIP's spectrum encoder reaches test R2 of 0.87/0.57/0.43/0.63
(zero-shot kNN) and 0.88/0.58/0.43/0.64 (few-shot MLP) on stellar mass /
metallicity / age / sSFR.

Property derivations (AstroCLIP convention):
  stellar_mass = LOG_MSTAR              log10 M* [Msun]
  metallicity  = log10(Z_MW)            mass-weighted metallicity
  age          = TAGE_MW                mass-weighted age [Gyr]
  ssfr         = log10(AVG_SFR) - LOG_MSTAR   log10 sSFR over last 1 Gyr

Embeddings are shared with Evals/spectra_redshift: the cache under its
results/<label>/ is reused if present and created there otherwise, so running
both probes embeds the spectra only once.

Run:  python spectra_properties_probe.py --checkpoint <family>/step_<N>.pt
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
EVALS_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = EVALS_DIR / "desi_crossmatch" / "data" / "astroclip"
DEFAULT_PROVABGS_DIR = SCRIPT_DIR / "data" / "provabgs"
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
acp = sp.acp  # astroclip_redshift_probe: split_files, read_targets, evaluate_seed
rp = sp.rp  # redshift_probe: heads, metrics, atomic_json

CATALOG_COLUMNS = ["object_id", "LOG_MSTAR", "Z_MW", "TAGE_MW", "AVG_SFR"]
# Column-name variants across catalog formats: the MMU parquet conversion
# (unprefixed) and the original DESI VAC HDF5 (PROVABGS_*-prefixed).
HDF5_KEY_CANDIDATES = {
    "targetid": ("TARGETID", "targetid"),
    "LOG_MSTAR": ("LOG_MSTAR", "PROVABGS_LOGMSTAR", "PROVABGS_LOGMSTAR_BF"),
    "Z_MW": ("Z_MW", "PROVABGS_Z_MW", "PROVABGS_Z_MW_BF"),
    "TAGE_MW": ("TAGE_MW", "PROVABGS_TAGE_MW", "PROVABGS_TAGE_MW_BF"),
    "AVG_SFR": ("AVG_SFR", "PROVABGS_AVGSFR_1GYR", "PROVABGS_AVGSFR_1GYR_BF"),
}
PROPERTIES = ("stellar_mass", "metallicity", "age", "ssfr")
ASTROCLIP_SPECTRUM_R2 = {
    "stellar_mass": {"zero_shot_knn": 0.87, "few_shot_mlp": 0.88},
    "metallicity": {"zero_shot_knn": 0.57, "few_shot_mlp": 0.58},
    "age": {"zero_shot_knn": 0.43, "few_shot_mlp": 0.43},
    "ssfr": {"zero_shot_knn": 0.63, "few_shot_mlp": 0.64},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--provabgs-path",
        type=Path,
        default=DEFAULT_PROVABGS_DIR,
        help="PROVABGS catalog: the original DESI VAC HDF5 file "
        "(BGS_ANY_full.provabgs.sv3.v0.hdf5) or a directory of MMU parquet "
        "shards from download_provabgs.sh.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=EMBEDDINGS_DIR,
        help="Where the shared embeddings.npy cache lives (default: the "
        "spectra_redshift results dir, so both probes embed once).",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_catalog(provabgs_path: Path) -> dict[int, np.ndarray]:
    """targetid -> (LOG_MSTAR, Z_MW, TAGE_MW, AVG_SFR), first occurrence wins.

    Accepts either the original DESI VAC HDF5 file
    (BGS_ANY_full.provabgs.sv3.v0.hdf5) or a directory of MMU parquet shards.
    """
    if provabgs_path.is_file():
        ids, values = read_catalog_hdf5(provabgs_path)
        return build_catalog(ids, values)
    shards = sorted(provabgs_path.rglob("Npix=*.parquet"))
    if not shards:
        raise SystemExit(
            f"No PROVABGS catalog at {provabgs_path} — pass the DESI VAC HDF5 "
            "via --provabgs-path or run download_provabgs.sh first."
        )
    catalog: dict[int, np.ndarray] = {}
    for path in shards:
        table = pq.read_table(path, columns=CATALOG_COLUMNS)
        ids = table.column("object_id").to_numpy(zero_copy_only=False)
        values = np.stack(
            [
                table.column(c).to_numpy(zero_copy_only=False).astype(np.float64)
                for c in CATALOG_COLUMNS[1:]
            ],
            axis=1,
        )
        catalog = build_catalog(ids, values, catalog)
    return catalog


def build_catalog(
    ids: np.ndarray, values: np.ndarray, catalog: dict[int, np.ndarray] | None = None
) -> dict[int, np.ndarray]:
    catalog = catalog if catalog is not None else {}
    for object_id, row in zip(ids, values):
        catalog.setdefault(int(object_id), row)
    return catalog


def read_catalog_hdf5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read (targetid, property matrix) from the original PROVABGS VAC HDF5.

    Handles both layouts h5py files use for tables: one dataset per column,
    or a single dataset with a compound dtype.
    """
    import h5py

    with h5py.File(path, "r") as h5:
        datasets = [k for k in h5.keys() if isinstance(h5[k], h5py.Dataset)]
        compound = (
            h5[datasets[0]]
            if len(datasets) == 1 and h5[datasets[0]].dtype.names
            else None
        )
        available = (
            list(compound.dtype.names) if compound is not None else datasets
        ) or list(h5.keys())
        lower = {name.lower(): name for name in available}

        def column(field: str) -> np.ndarray:
            for candidate in HDF5_KEY_CANDIDATES[field]:
                name = lower.get(candidate.lower())
                if name is not None:
                    # field indexing on a compound dataset reads only that field
                    data = compound[name] if compound is not None else h5[name][...]
                    return np.asarray(data)
            raise SystemExit(
                f"No {field} column in {path} (tried "
                f"{HDF5_KEY_CANDIDATES[field]}); available: {available}"
            )

        ids = column("targetid").astype(np.int64)
        values = np.stack(
            [column(c).astype(np.float64) for c in CATALOG_COLUMNS[1:]], axis=1
        )
    return ids, values


def build_targets(
    targetids: np.ndarray, catalog: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Per-spectrum property matrix (N,4) in PROPERTIES order + validity mask.

    Valid rows are matched in the catalog with LOG_MSTAR > 0 (PROVABGS fit
    succeeded) and strictly positive Z_MW / AVG_SFR / TAGE_MW so the logs are
    finite. One shared mask keeps all four properties on the identical sample.
    """
    raw = np.full((len(targetids), 4), np.nan)
    matched = np.zeros(len(targetids), dtype=bool)
    for i, targetid in enumerate(targetids):
        row = catalog.get(int(targetid))
        if row is not None:
            raw[i] = row
            matched[i] = True
    log_mstar, z_mw, tage, avg_sfr = raw.T
    with np.errstate(invalid="ignore"):
        valid = matched & (log_mstar > 0) & (z_mw > 0) & (avg_sfr > 0) & (tage > 0)
    targets = np.full((len(targetids), len(PROPERTIES)), np.nan)
    targets[valid, 0] = log_mstar[valid]
    targets[valid, 1] = np.log10(z_mw[valid])
    targets[valid, 2] = tage[valid]
    targets[valid, 3] = np.log10(avg_sfr[valid]) - log_mstar[valid]
    stats = {"matched": int(matched.sum()), "valid": int(valid.sum())}
    return targets, valid, stats


def strip_redshift_metrics(repeat: dict[str, Any], y_std: float) -> dict[str, Any]:
    """Return metrics in physical units, dropping the redshift-specific ones.

    Heads are fit on z-scored targets: R2 is affine-invariant, mae/rmse scale
    back by the target std, and nmad/outlier_fraction (both built on the
    photo-z (1+z) error convention) are meaningless here and removed.
    """
    repeat = copy.deepcopy(repeat)
    for head in repeat["heads"].values():
        for split in ("validation", "test"):
            metrics = head[split]
            metrics["mae"] *= y_std
            metrics["rmse"] *= y_std
            del metrics["nmad"], metrics["outlier_fraction"]
    return repeat


def summarize(repeats: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for head in ("ridge", "knn", "mlp"):
        summary[head] = {}
        for split in ("validation", "test"):
            summary[head][split] = {}
            for metric in ("r2", "mae", "rmse"):
                values = [r["heads"][head][split][metric] for r in repeats]
                summary[head][split][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values,
                }
    return summary


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    label = args.label or "_".join(
        [args.checkpoint.resolve().parent.name, args.checkpoint.stem]
    )
    started = time.time()

    embed_args = copy.copy(args)
    embed_args.output_dir = args.embeddings_dir
    embeddings, embed_meta = sp.cached_embeddings(
        label, args.checkpoint, embed_args, device
    )

    _, targetids, is_test = acp.read_targets(args.data_dir)
    if len(targetids) != len(embeddings):
        raise RuntimeError(f"{len(targetids)} targets, {len(embeddings)} embeddings")
    targets, valid, match_stats = build_targets(
        targetids, read_catalog(args.provabgs_path)
    )
    print(
        f"PROVABGS match: {match_stats['matched']}/{len(targetids)} matched, "
        f"{match_stats['valid']} valid "
        f"({int((valid & ~is_test).sum())} train / {int((valid & is_test).sum())} test)"
    )

    train_idx = np.flatnonzero(valid & ~is_test)
    test_idx = np.flatnonzero(valid & is_test)
    mean = embeddings[torch.from_numpy(train_idx)].mean(dim=0)
    std = embeddings[torch.from_numpy(train_idx)].std(dim=0).clamp_min(1e-6)
    train_x = ((embeddings[torch.from_numpy(train_idx)] - mean) / std).to(device)
    test_x = ((embeddings[torch.from_numpy(test_idx)] - mean) / std).to(device)

    properties: dict[str, Any] = {}
    for column, name in enumerate(PROPERTIES):
        train_y = targets[train_idx, column]
        test_y = targets[test_idx, column]
        y_mean, y_std = float(train_y.mean()), float(train_y.std())
        train_y_scaled = (train_y - y_mean) / y_std
        test_y_scaled = (test_y - y_mean) / y_std
        repeats = [
            strip_redshift_metrics(
                acp.evaluate_seed(
                    train_x,
                    train_y_scaled,
                    test_x,
                    test_y_scaled,
                    seed,
                    args.mlp_epochs,
                    device,
                ),
                y_std,
            )
            for seed in args.seeds
        ]
        summary = summarize(repeats)
        properties[name] = {
            "target_mean": y_mean,
            "target_std": y_std,
            "published_reference": ASTROCLIP_SPECTRUM_R2[name],
            "repeats": repeats,
            "summary": summary,
        }
        line = ", ".join(
            f"{head} R2={summary[head]['test']['r2']['mean']:.4f}"
            f"±{summary[head]['test']['r2']['std']:.4f}"
            for head in ("ridge", "knn", "mlp")
        )
        reference = ASTROCLIP_SPECTRUM_R2[name]
        print(
            f"{name}: {line} (AstroCLIP spectrum kNN "
            f"{reference['zero_shot_knn']:.2f} / MLP {reference['few_shot_mlp']:.2f})"
        )

    result = {
        "label": label,
        "modality": "spectra",
        "task": "PROVABGS physical-property regression",
        "sample": "AstroCLIP DESI-LS x DESI EDR cross-match x PROVABGS BGS EDR",
        "catalog": f"{args.provabgs_path.resolve()}, joined on targetid",
        "property_derivations": {
            "stellar_mass": "LOG_MSTAR",
            "metallicity": "log10(Z_MW)",
            "age": "TAGE_MW",
            "ssfr": "log10(AVG_SFR) - LOG_MSTAR",
        },
        "embedding_metadata": embed_meta,
        "match_stats": {
            **match_stats,
            "num_train": int(len(train_idx)),
            "num_test": int(len(test_idx)),
        },
        "seeds": args.seeds,
        "properties": properties,
        "elapsed_seconds": time.time() - started,
    }
    rp.atomic_json(args.output_dir.resolve() / label / "metrics.json", result)


if __name__ == "__main__":
    main()
