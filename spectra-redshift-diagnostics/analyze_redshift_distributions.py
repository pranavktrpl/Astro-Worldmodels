#!/usr/bin/env python3
"""Audit DESI pipeline-redshift distributions in AstroJEPA training corpora."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_SSL_DIR = Path(
    "/mnt/datasets/utbd_pranav/desi_edr_sv3/mmu_desi_edr_sv3/dataset"
)
DEFAULT_ASTROCLIP_DIR = Path("/mnt/datasets/pranav/astroclip/data")
DEFAULT_MMU_XMATCH_DIR = Path(
    "/mnt/datasets/pranav/desi_legacysurvey_xmatch/data"
)
DEFAULT_MANUAL_PAIR_DIR = Path(
    "/mnt/datasets/pranav/desi_dr8_manual_unique_xmatch/pairs"
)

HIGH_Z_THRESHOLD = 0.5
PLOT_MIN_Z = -0.005
PLOT_MAX_Z = 6.0
EXPECTED_SSL_ROWS = 1_126_441
EXPECTED_SSL_TRAIN_ROWS = 1_103_874
EXPECTED_CROSS_MODAL_ROWS = 307_428

BANDS = (
    ("negative", None, 0.0),
    ("0.0 <= z < 0.1", 0.0, 0.1),
    ("0.1 <= z < 0.3", 0.1, 0.3),
    ("0.3 <= z < 0.5", 0.3, 0.5),
    ("0.5 <= z < 1.0", 0.5, 1.0),
    ("1.0 <= z < 2.0", 1.0, 2.0),
    ("2.0 <= z < 3.0", 2.0, 3.0),
    ("z >= 3.0", 3.0, None),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssl-dir", type=Path, default=DEFAULT_SSL_DIR)
    parser.add_argument(
        "--astroclip-dir", type=Path, default=DEFAULT_ASTROCLIP_DIR
    )
    parser.add_argument(
        "--mmu-xmatch-dir", type=Path, default=DEFAULT_MMU_XMATCH_DIR
    )
    parser.add_argument(
        "--manual-pair-dir", type=Path, default=DEFAULT_MANUAL_PAIR_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Regenerate summaries and plots from redshift_values.npz.",
    )
    return parser.parse_args()


def parquet_files(root, recursive=False):
    pattern = "**/*.parquet" if recursive else "*.parquet"
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {root}")
    return files


def read_table(path, columns):
    return pq.ParquetFile(path).read(columns=columns)


def numeric_column(table, name, dtype=np.float64):
    return np.asarray(table[name].to_pylist(), dtype=dtype)


def quality_column(table, name):
    # -1 means the source did not expose a warning flag, 0 is good, 1 is warned.
    return np.asarray(
        [-1 if value is None else int(bool(value)) for value in table[name].to_pylist()],
        dtype=np.int8,
    )


def spectra_split_for_object_id(object_id, seed=42):
    digest = hashlib.blake2b(
        f"{seed}:{object_id}".encode("utf-8"),
        digest_size=8,
        person=b"AstroJV2",
    ).digest()
    bucket = int.from_bytes(digest, byteorder="big") % 100
    if bucket < 98:
        return "train"
    if bucket == 98:
        return "validation"
    return "test"


def load_manual_pair_ids(pair_dir):
    object_ids = []
    for index, path in enumerate(parquet_files(pair_dir), start=1):
        table = read_table(path, ["object_id"])
        object_ids.extend(str(value) for value in table["object_id"].to_pylist())
        if index % 10 == 0:
            print(f"manual pairs: read {index} shards", flush=True)
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("Manual pair dataset contains duplicate spectrum object IDs")
    return object_ids


def scan_ssl(ssl_dir, manual_ids):
    files = parquet_files(ssl_dir, recursive=True)
    manual_set = set(manual_ids)
    manual_lookup = {}
    split_redshifts = defaultdict(list)
    split_quality = defaultdict(list)
    all_redshifts = []
    all_quality = []
    all_object_ids = []

    for index, path in enumerate(files, start=1):
        table = read_table(path, ["object_id", "Z", "ZWARN"])
        object_ids = [str(value) for value in table["object_id"].to_pylist()]
        redshifts = numeric_column(table, "Z")
        quality = quality_column(table, "ZWARN")

        all_redshifts.append(redshifts)
        all_quality.append(quality)
        all_object_ids.extend(object_ids)

        row_splits = [spectra_split_for_object_id(value) for value in object_ids]
        for split in ("train", "validation", "test"):
            selected = np.fromiter(
                (value == split for value in row_splits),
                dtype=bool,
                count=len(row_splits),
            )
            if selected.any():
                split_redshifts[split].append(redshifts[selected])
                split_quality[split].append(quality[selected])

        for object_id, redshift, warning in zip(object_ids, redshifts, quality):
            if object_id in manual_set:
                if object_id in manual_lookup:
                    raise ValueError(f"Duplicate SSL object ID required by manual pairs: {object_id}")
                manual_lookup[object_id] = (float(redshift), int(warning))

        if index % 25 == 0 or index == len(files):
            print(f"SSL: read {index}/{len(files)} shards", flush=True)

    redshift_all = np.concatenate(all_redshifts)
    quality_all = np.concatenate(all_quality)
    if redshift_all.size != EXPECTED_SSL_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SSL_ROWS:,} SSL rows, found {redshift_all.size:,}"
        )
    if len(all_object_ids) != len(set(all_object_ids)):
        raise ValueError("SSL dataset contains duplicate object IDs")

    missing = [object_id for object_id in manual_ids if object_id not in manual_lookup]
    if missing:
        raise ValueError(
            f"Could not recover DESI redshift for {len(missing):,} manual pairs; "
            f"first missing ID: {missing[0]}"
        )

    result = {
        "all_redshift": redshift_all,
        "all_quality": quality_all,
    }
    for split in ("train", "validation", "test"):
        result[f"{split}_redshift"] = np.concatenate(split_redshifts[split])
        result[f"{split}_quality"] = np.concatenate(split_quality[split])
    if result["train_redshift"].size != EXPECTED_SSL_TRAIN_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SSL_TRAIN_ROWS:,} SSL training rows, found "
            f"{result['train_redshift'].size:,}"
        )
    return result, manual_lookup


def scan_astroclip(root):
    redshifts = []
    object_ids = []
    files = parquet_files(root)
    for index, path in enumerate(files, start=1):
        table = read_table(path, ["targetid", "redshift"])
        redshifts.append(numeric_column(table, "redshift"))
        object_ids.extend(str(value) for value in table["targetid"].to_pylist())
        if index % 25 == 0 or index == len(files):
            print(f"AstroCLIP pairs: read {index}/{len(files)} shards", flush=True)
    values = np.concatenate(redshifts)
    quality = np.full(values.shape, -1, dtype=np.int8)
    return values, quality, object_ids


def scan_mmu_xmatch(root):
    redshifts = []
    qualities = []
    object_ids = []
    files = parquet_files(root)
    for index, path in enumerate(files, start=1):
        table = read_table(path, ["object_id_spec", "Z_spec", "ZWARN_spec"])
        redshifts.append(numeric_column(table, "Z_spec"))
        qualities.append(quality_column(table, "ZWARN_spec"))
        object_ids.extend(str(value) for value in table["object_id_spec"].to_pylist())
        if index % 50 == 0 or index == len(files):
            print(f"MMU pairs: read {index}/{len(files)} shards", flush=True)
    return np.concatenate(redshifts), np.concatenate(qualities), object_ids


def recover_manual_redshifts(manual_ids, manual_lookup):
    redshifts = np.asarray(
        [manual_lookup[object_id][0] for object_id in manual_ids], dtype=np.float64
    )
    quality = np.asarray(
        [manual_lookup[object_id][1] for object_id in manual_ids], dtype=np.int8
    )
    return redshifts, quality


def scan_all(args):
    manual_ids = load_manual_pair_ids(args.manual_pair_dir)
    ssl, manual_lookup = scan_ssl(args.ssl_dir, manual_ids)
    astro_z, astro_quality, astro_ids = scan_astroclip(args.astroclip_dir)
    mmu_z, mmu_quality, mmu_ids = scan_mmu_xmatch(args.mmu_xmatch_dir)
    manual_z, manual_quality = recover_manual_redshifts(manual_ids, manual_lookup)

    cross_z = np.concatenate([astro_z, mmu_z, manual_z])
    cross_quality = np.concatenate([astro_quality, mmu_quality, manual_quality])
    if cross_z.size != EXPECTED_CROSS_MODAL_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_CROSS_MODAL_ROWS:,} cross-modal rows, found "
            f"{cross_z.size:,}"
        )

    cross_ids = astro_ids + mmu_ids + manual_ids
    duplicate_rows = len(cross_ids) - len(set(cross_ids))
    return {
        **{f"ssl_{key}": value for key, value in ssl.items()},
        "crossmodal_redshift": cross_z,
        "crossmodal_quality": cross_quality,
        "crossmodal_astroclip_redshift": astro_z,
        "crossmodal_astroclip_quality": astro_quality,
        "crossmodal_mmu_redshift": mmu_z,
        "crossmodal_mmu_quality": mmu_quality,
        "crossmodal_manual_redshift": manual_z,
        "crossmodal_manual_quality": manual_quality,
        "crossmodal_duplicate_object_rows": np.asarray([duplicate_rows]),
    }


def finite_nonnegative(values):
    return np.isfinite(values) & (values >= 0)


def band_count(values, low, high):
    finite = np.isfinite(values)
    if low is None:
        return int(np.count_nonzero(finite & (values < high)))
    if high is None:
        return int(np.count_nonzero(finite & (values >= low)))
    return int(np.count_nonzero(finite & (values >= low) & (values < high)))


def summarize(values, quality):
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    nonnegative_values = finite_values[finite_values >= 0]
    known_quality = quality >= 0
    summary = {
        "rows": int(values.size),
        "finite_redshift": int(finite_values.size),
        "nonfinite_redshift": int(np.count_nonzero(~finite_mask)),
        "negative_redshift": int(np.count_nonzero(finite_values < 0)),
        "nonnegative_finite_redshift": int(nonnegative_values.size),
        "low_z_lt_0_5": int(np.count_nonzero(finite_values < HIGH_Z_THRESHOLD)),
        "high_z_ge_0_5": int(np.count_nonzero(finite_values >= HIGH_Z_THRESHOLD)),
        "z_ge_1": int(np.count_nonzero(finite_values >= 1.0)),
        "z_ge_2": int(np.count_nonzero(finite_values >= 2.0)),
        "z_ge_3": int(np.count_nonzero(finite_values >= 3.0)),
        "z_gt_plot_max": int(np.count_nonzero(finite_values > PLOT_MAX_Z)),
        "quality_flag_known": int(np.count_nonzero(known_quality)),
        "known_zwarn_false": int(np.count_nonzero(known_quality & (quality == 0))),
        "known_zwarn_true": int(np.count_nonzero(known_quality & (quality == 1))),
    }
    if finite_values.size:
        summary.update(
            {
                "minimum": float(finite_values.min()),
                "maximum": float(finite_values.max()),
                "mean": float(finite_values.mean()),
                "median": float(np.median(finite_values)),
                "p90": float(np.quantile(finite_values, 0.90)),
                "p95": float(np.quantile(finite_values, 0.95)),
                "p99": float(np.quantile(finite_values, 0.99)),
                "high_z_fraction_of_all_finite": float(
                    np.count_nonzero(finite_values >= HIGH_Z_THRESHOLD)
                    / finite_values.size
                ),
            }
        )
    if nonnegative_values.size:
        summary["high_z_fraction_of_nonnegative"] = float(
            np.count_nonzero(nonnegative_values >= HIGH_Z_THRESHOLD)
            / nonnegative_values.size
        )
        summary["nonnegative_median"] = float(np.median(nonnegative_values))
    return summary


def source_series(arrays):
    return {
        "AstroCLIP mirror": arrays["crossmodal_astroclip_redshift"],
        "MMU DESI x Legacy": arrays["crossmodal_mmu_redshift"],
        "Manual DESI x DR8": arrays["crossmodal_manual_redshift"],
    }


def plot_distribution(values, output_path, title, sources=None):
    finite_values = values[np.isfinite(values)]
    nonnegative_values = finite_values[finite_values >= 0]
    positive_bins = np.linspace(0.0, PLOT_MAX_Z, 241)
    full_bins = np.concatenate(([PLOT_MIN_Z], positive_bins))
    colors = ("#176B87", "#D95F59", "#4B8B3B")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), constrained_layout=True)

    axes[0].hist(
        nonnegative_values,
        bins=positive_bins,
        color="#28666E",
        alpha=0.9,
        histtype="stepfilled",
    )
    axes[0].set_xlim(0, 1.5)
    axes[0].set_title("Nonnegative low-redshift structure")
    axes[0].set_xlabel("Spectroscopic redshift z")
    axes[0].set_ylabel("Spectrum count per bin")
    axes[0].text(
        0.98,
        0.95,
        f"Negative-Z stellar rows: {np.count_nonzero(finite_values < 0):,}",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )

    axes[1].hist(
        finite_values,
        bins=full_bins,
        color="#202A44",
        linewidth=1.6,
        histtype="step",
        label="Combined" if sources else title,
    )
    if sources:
        for (label, source_values), color in zip(sources.items(), colors):
            selected = source_values[np.isfinite(source_values)]
            axes[1].hist(
                selected,
                bins=full_bins,
                histtype="step",
                linewidth=1.1,
                alpha=0.9,
                color=color,
                label=f"{label} ({selected.size:,})",
            )
    axes[1].set_yscale("log")
    axes[1].set_xlim(PLOT_MIN_Z, PLOT_MAX_Z)
    axes[1].set_title("Full distribution, logarithmic count")
    axes[1].set_xlabel("Spectroscopic redshift z")
    axes[1].set_ylabel("Spectrum count per bin")
    axes[1].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.axvline(
            HIGH_Z_THRESHOLD,
            color="#C73E1D",
            linestyle="--",
            linewidth=1.4,
        )
        axis.grid(axis="y", color="#D5D5D5", linewidth=0.6, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    low_count = int(np.count_nonzero(finite_values < HIGH_Z_THRESHOLD))
    high_count = int(np.count_nonzero(finite_values >= HIGH_Z_THRESHOLD))
    high_fraction = high_count / max(1, finite_values.size)
    fig.suptitle(
        f"{title}\n"
        f"z < 0.5: {low_count:,} | z >= 0.5: {high_count:,} "
        f"({high_fraction:.2%} of finite rows)",
        fontsize=14,
    )
    fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


def write_band_counts(output_path, datasets):
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("dataset", "redshift_band", "count", "percent"),
        )
        writer.writeheader()
        for dataset_name, values in datasets.items():
            denominator = max(1, int(np.count_nonzero(np.isfinite(values))))
            for label, low, high in BANDS:
                count = band_count(values, low, high)
                writer.writerow(
                    {
                        "dataset": dataset_name,
                        "redshift_band": label,
                        "count": count,
                        "percent": f"{100.0 * count / denominator:.6f}",
                    }
                )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.output_dir / "redshift_values.npz"

    if args.reuse_cache:
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        with np.load(cache_path) as cache:
            arrays = {name: cache[name] for name in cache.files}
    else:
        arrays = scan_all(args)
        np.savez_compressed(cache_path, **arrays)

    datasets = {
        "ssl_pretraining_train": arrays["ssl_train_redshift"],
        "crossmodal_all_pairs": arrays["crossmodal_redshift"],
        "crossmodal_astroclip": arrays["crossmodal_astroclip_redshift"],
        "crossmodal_mmu": arrays["crossmodal_mmu_redshift"],
        "crossmodal_manual": arrays["crossmodal_manual_redshift"],
    }
    qualities = {
        "ssl_pretraining_train": arrays["ssl_train_quality"],
        "crossmodal_all_pairs": arrays["crossmodal_quality"],
        "crossmodal_astroclip": arrays["crossmodal_astroclip_quality"],
        "crossmodal_mmu": arrays["crossmodal_mmu_quality"],
        "crossmodal_manual": arrays["crossmodal_manual_quality"],
    }

    summaries = {
        name: summarize(values, qualities[name]) for name, values in datasets.items()
    }
    summaries["ssl_full_corpus"] = summarize(
        arrays["ssl_all_redshift"], arrays["ssl_all_quality"]
    )
    summaries["ssl_validation"] = summarize(
        arrays["ssl_validation_redshift"], arrays["ssl_validation_quality"]
    )
    summaries["ssl_test"] = summarize(
        arrays["ssl_test_redshift"], arrays["ssl_test_quality"]
    )

    output = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "redshift_origin": (
            "DESI spectroscopic pipeline/template-fit values stored as Z, Z_spec, "
            "or AstroCLIP redshift; no neural-backbone predictions"
        ),
        "high_redshift_definition": f"z >= {HIGH_Z_THRESHOLD}",
        "plot_range": [0.0, PLOT_MAX_Z],
        "paths": {
            "ssl": str(args.ssl_dir),
            "astroclip": str(args.astroclip_dir),
            "mmu_xmatch": str(args.mmu_xmatch_dir),
            "manual_pairs": str(args.manual_pair_dir),
        },
        "crossmodal_duplicate_object_rows": int(
            arrays["crossmodal_duplicate_object_rows"][0]
        ),
        "summaries": summaries,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")

    write_band_counts(args.output_dir / "redshift_band_counts.csv", datasets)
    plot_distribution(
        arrays["ssl_train_redshift"],
        args.output_dir / "ssl_pretraining_redshift_distribution.png",
        "DESI spectra used for spectrum-v2 SSL pretraining",
    )
    plot_distribution(
        arrays["crossmodal_redshift"],
        args.output_dir / "crossmodal_redshift_distribution.png",
        "DESI spectra used for 307K cross-modal training",
        sources=source_series(arrays),
    )

    print(json.dumps(summaries, indent=2, sort_keys=True), flush=True)
    print(f"Wrote diagnostics to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
