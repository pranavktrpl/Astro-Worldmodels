#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.cross_modal import cross_modal_split_for_dr8_id


DEFAULT_IMAGE_METADATA = Path(
    "/mnt/datasets/utbd_pranav/galaxies/metadata.parquet"
)
DEFAULT_IMAGE_DATASET = Path(
    "/mnt/datasets/utbd_pranav/galaxies/with_crops"
)
DEFAULT_SPECTRUM_DATASET = Path(
    "/mnt/datasets/utbd_pranav/desi_edr_sv3/mmu_desi_edr_sv3/dataset"
)
DEFAULT_OUTPUT = Path(
    "/mnt/datasets/pranav/desi_dr8_manual_unique_xmatch"
)
DEFAULT_ASTROCLIP_DATASET = Path(
    "/mnt/datasets/pranav/astroclip"
)
DEFAULT_MMU_XMATCH_DATASET = Path(
    "/mnt/datasets/pranav/desi_legacysurvey_xmatch"
)
ARCSEC_PER_RADIAN = 206_264.80624709636


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-match and materialize paired image-DESI training shards"
    )
    parser.add_argument("--image-metadata", type=Path, default=DEFAULT_IMAGE_METADATA)
    parser.add_argument("--image-dataset", type=Path, default=DEFAULT_IMAGE_DATASET)
    parser.add_argument("--spectrum-dataset", type=Path, default=DEFAULT_SPECTRUM_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--radius-arcsec", type=float, default=1.0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--rows-per-shard", type=int, default=1_000)
    parser.add_argument(
        "--exclude-astroclip-dir",
        type=Path,
        default=DEFAULT_ASTROCLIP_DATASET,
    )
    parser.add_argument(
        "--exclude-mmu-dir",
        type=Path,
        default=DEFAULT_MMU_XMATCH_DATASET,
    )
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def unit_vectors(ra_degrees, dec_degrees):
    ra = np.deg2rad(ra_degrees)
    dec = np.deg2rad(dec_degrees)
    cos_dec = np.cos(dec)
    return np.column_stack(
        (
            cos_dec * np.cos(ra),
            cos_dec * np.sin(ra),
            np.sin(dec),
        )
    )


def chord_to_arcsec(chord_distance):
    return 2.0 * np.arcsin(np.clip(chord_distance / 2.0, 0.0, 1.0)) * ARCSEC_PER_RADIAN


def stable_pair_order(dr8_id, object_id, seed):
    digest = hashlib.blake2b(
        f"{seed}:{dr8_id}:{object_id}".encode("utf-8"),
        digest_size=8,
        person=b"AstroJOrd",
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def read_spectrum_index(spectrum_dataset):
    dataset = pads.dataset(
        spectrum_dataset,
        format="parquet",
        partitioning="hive",
    )
    return dataset.scanner(
        columns=["object_id", "ra", "dec", "__filename"],
        use_threads=True,
    ).to_table()


def build_manifest(
    image_metadata,
    spectrum_dataset,
    radius_arcsec,
    split_seed,
):
    print("Reading image coordinates...", flush=True)
    image_table = pq.read_table(
        image_metadata,
        columns=["dr8_id", "ra", "dec"],
    )
    image_ra = np.asarray(image_table["ra"], dtype=np.float64)
    image_dec = np.asarray(image_table["dec"], dtype=np.float64)
    image_finite = np.isfinite(image_ra) & np.isfinite(image_dec)
    if not image_finite.all():
        raise ValueError("Image metadata contains non-finite coordinates")

    print("Reading spectrum coordinates and provenance...", flush=True)
    spectrum_table = read_spectrum_index(spectrum_dataset)
    spectrum_ra = np.asarray(spectrum_table["ra"], dtype=np.float64)
    spectrum_dec = np.asarray(spectrum_table["dec"], dtype=np.float64)
    spectrum_finite = np.isfinite(spectrum_ra) & np.isfinite(spectrum_dec)
    if not spectrum_finite.all():
        raise ValueError("Spectrum dataset contains non-finite coordinates")

    print("Building image sky index and matching spectra...", flush=True)
    tree = cKDTree(
        unit_vectors(image_ra, image_dec),
        compact_nodes=True,
        balanced_tree=True,
    )
    chord_distance, image_indices = tree.query(
        unit_vectors(spectrum_ra, spectrum_dec),
        k=2,
        workers=-1,
    )
    separations = chord_to_arcsec(chord_distance)
    accepted = (
        (separations[:, 0] <= radius_arcsec)
        & (separations[:, 1] > radius_arcsec)
    )
    accepted_spectrum_indices = np.flatnonzero(accepted)
    accepted_image_indices = image_indices[accepted, 0]

    dr8_ids = image_table["dr8_id"].take(
        pa.array(accepted_image_indices, type=pa.int64())
    )
    object_ids = spectrum_table["object_id"].take(
        pa.array(accepted_spectrum_indices, type=pa.int64())
    )
    dr8_id_values = dr8_ids.to_pylist()
    object_id_values = object_ids.to_pylist()
    splits = [
        cross_modal_split_for_dr8_id(dr8_id, split_seed)
        for dr8_id in dr8_id_values
    ]
    shuffle_keys = [
        stable_pair_order(dr8_id, object_id, split_seed)
        for dr8_id, object_id in zip(dr8_id_values, object_id_values)
    ]

    manifest = pa.table(
        {
            "dr8_id": dr8_ids,
            "object_id": object_ids,
            "split": pa.array(splits, type=pa.string()),
            "separation_arcsec": pa.array(
                separations[accepted, 0].astype(np.float32)
            ),
            "image_ra": image_table["ra"].take(
                pa.array(accepted_image_indices, type=pa.int64())
            ),
            "image_dec": image_table["dec"].take(
                pa.array(accepted_image_indices, type=pa.int64())
            ),
            "spectrum_ra": spectrum_table["ra"].take(
                pa.array(accepted_spectrum_indices, type=pa.int64())
            ),
            "spectrum_dec": spectrum_table["dec"].take(
                pa.array(accepted_spectrum_indices, type=pa.int64())
            ),
            "spectrum_source_file": spectrum_table["__filename"].take(
                pa.array(accepted_spectrum_indices, type=pa.int64())
            ),
            "shuffle_key": pa.array(shuffle_keys, type=pa.uint64()),
        }
    ).sort_by([("split", "ascending"), ("shuffle_key", "ascending")])

    stats = {
        "image_rows": image_table.num_rows,
        "spectrum_rows": spectrum_table.num_rows,
        "radius_arcsec": radius_arcsec,
        "nearest_within_radius": int((separations[:, 0] <= radius_arcsec).sum()),
        "ambiguous_second_within_radius": int(
            (
                (separations[:, 0] <= radius_arcsec)
                & (separations[:, 1] <= radius_arcsec)
            ).sum()
        ),
        "accepted_pair_rows": manifest.num_rows,
        "accepted_unique_images": len(set(dr8_id_values)),
        "repeated_spectrum_rows_for_images": (
            manifest.num_rows - len(set(dr8_id_values))
        ),
        "split_rows": {
            split: int(
                pc.sum(pc.equal(manifest["split"], split)).as_py()
            )
            for split in ("train", "validation", "test")
        },
    }
    print(json.dumps(stats, indent=2), flush=True)
    return manifest, stats



def read_target_ids(files, column):
    target_ids = set()
    for path in files:
        values = pq.read_table(path, columns=[column])[column].to_pylist()
        target_ids.update(str(value) for value in values if value is not None)
    return target_ids


def exclude_preloaded_targets(manifest, astroclip_dir, mmu_dir):
    astroclip_files = sorted((astroclip_dir / "data").glob("*.parquet"))
    mmu_files = sorted((mmu_dir / "data").glob("*.parquet"))
    if not astroclip_files:
        raise FileNotFoundError(f"No AstroCLIP shards found under {astroclip_dir}")
    if not mmu_files:
        raise FileNotFoundError(f"No MMU cross-match shards found under {mmu_dir}")

    print("Reading pre-loaded DESI target IDs for de-duplication...", flush=True)
    astroclip_ids = read_target_ids(astroclip_files, "targetid")
    mmu_ids = read_target_ids(mmu_files, "object_id_spec")
    object_ids = [str(value) for value in manifest["object_id"].to_pylist()]

    overlap_astroclip = sum(value in astroclip_ids for value in object_ids)
    overlap_mmu = sum(value in mmu_ids for value in object_ids)
    existing_ids = astroclip_ids | mmu_ids
    keep = pa.array(
        [value not in existing_ids for value in object_ids],
        type=pa.bool_(),
    )
    filtered = manifest.filter(keep)
    stats = {
        "preloaded_astroclip_target_ids": len(astroclip_ids),
        "preloaded_mmu_target_ids": len(mmu_ids),
        "excluded_astroclip_pair_rows": overlap_astroclip,
        "excluded_mmu_pair_rows": overlap_mmu,
        "excluded_existing_pair_rows": manifest.num_rows - filtered.num_rows,
        "new_pair_rows_after_deduplication": filtered.num_rows,
    }
    print(json.dumps(stats, indent=2), flush=True)
    return filtered, stats

def read_selected_rows(dataset_path, key, values, payload_columns):
    dataset = pads.dataset(dataset_path, format="parquet", partitioning="hive")
    selected = dataset.scanner(
        columns=[key, *payload_columns, "__filename"],
        filter=pads.field(key).isin(pa.array(sorted(set(values)))),
        use_threads=True,
    ).to_table()
    selected = selected.rename_columns(
        [
            *selected.column_names[:-1],
            f"{key}_source_file",
        ]
    )
    return selected


def unique_index(table, key):
    result = {}
    duplicates = set()
    for index, value in enumerate(table[key].to_pylist()):
        if value in result:
            duplicates.add(value)
        result[value] = index
    if duplicates:
        examples = sorted(duplicates)[:5]
        raise ValueError(f"Duplicate {key} rows in source data; examples: {examples}")
    return result


def prepare_payloads(manifest, image_dataset, spectrum_dataset):
    dr8_ids = manifest["dr8_id"].to_pylist()
    object_ids = manifest["object_id"].to_pylist()

    print("Selecting matched image payloads...", flush=True)
    images = read_selected_rows(
        image_dataset,
        key="dr8_id",
        values=dr8_ids,
        payload_columns=["image_crop"],
    )
    large_image_type = pa.struct(
        [
            pa.field("bytes", pa.large_binary()),
            pa.field("path", pa.string()),
        ]
    )
    image_column_index = images.schema.get_field_index("image_crop")
    images = images.set_column(
        image_column_index,
        "image_crop",
        pc.cast(images["image_crop"], large_image_type),
    )
    image_index = unique_index(images, "dr8_id")

    print("Selecting matched spectrum payloads...", flush=True)
    spectra = read_selected_rows(
        spectrum_dataset,
        key="object_id",
        values=object_ids,
        payload_columns=["spectrum"],
    )
    spectrum_index = unique_index(spectra, "object_id")

    missing_images = sorted(set(dr8_ids) - set(image_index))
    missing_spectra = sorted(set(object_ids) - set(spectrum_index))
    available = pa.array(
        [
            dr8_id in image_index and object_id in spectrum_index
            for dr8_id, object_id in zip(dr8_ids, object_ids)
        ],
        type=pa.bool_(),
    )
    missing_pair_rows = manifest.num_rows - int(pc.sum(available).as_py())
    if missing_images or missing_spectra:
        print(
            "Excluding unavailable payloads: "
            f"{len(missing_images)} image IDs, "
            f"{len(missing_spectra)} spectrum IDs, "
            f"{missing_pair_rows} pair rows",
            flush=True,
        )
        manifest = manifest.filter(available)

    return manifest, images, spectra, image_index, spectrum_index, {
        "payload_missing_image_ids": len(missing_images),
        "payload_missing_spectrum_ids": len(missing_spectra),
        "payload_missing_pair_rows": missing_pair_rows,
    }


def write_pair_shards(
    manifest,
    images,
    spectra,
    image_index,
    spectrum_index,
    output_dir,
    rows_per_shard,
):
    pair_dir = output_dir / "pairs"
    pair_dir.mkdir(parents=True, exist_ok=False)

    for split in ("train", "validation", "test"):
        split_table = manifest.filter(pc.equal(manifest["split"], split))
        num_shards = math.ceil(split_table.num_rows / rows_per_shard)
        for shard_index in range(num_shards):
            shard = split_table.slice(
                shard_index * rows_per_shard,
                rows_per_shard,
            )
            image_take = pa.array(
                [image_index[value] for value in shard["dr8_id"].to_pylist()],
                type=pa.int64(),
            )
            spectrum_take = pa.array(
                [
                    spectrum_index[value]
                    for value in shard["object_id"].to_pylist()
                ],
                type=pa.int64(),
            )
            shard = (
                shard
                .append_column(
                    "image_crop",
                    images["image_crop"].take(image_take),
                )
                .append_column(
                    "image_source_file",
                    images["dr8_id_source_file"].take(image_take),
                )
                .append_column(
                    "spectrum",
                    spectra["spectrum"].take(spectrum_take),
                )
            )
            path = pair_dir / (
                f"{split}-{shard_index:05d}-of-{num_shards:05d}.parquet"
            )
            pq.write_table(
                shard,
                path,
                compression="zstd",
                compression_level=3,
                row_group_size=min(rows_per_shard, 1_000),
            )
        print(
            f"Wrote {split_table.num_rows:,} {split} rows in {num_shards} shards",
            flush=True,
        )


def main():
    args = parse_args()
    if args.radius_arcsec <= 0:
        raise ValueError("--radius-arcsec must be positive")
    if args.rows_per_shard < 1:
        raise ValueError("--rows-per-shard must be positive")
    if args.output_dir.exists():
        raise FileExistsError(
            f"{args.output_dir} already exists; move or remove it explicitly"
        )

    args.output_dir.mkdir(parents=True)
    manifest, stats = build_manifest(
        image_metadata=args.image_metadata,
        spectrum_dataset=args.spectrum_dataset,
        radius_arcsec=args.radius_arcsec,
        split_seed=args.split_seed,
    )
    matched_pair_rows_before_exclusions = manifest.num_rows
    manifest, exclusion_stats = exclude_preloaded_targets(
        manifest,
        astroclip_dir=args.exclude_astroclip_dir,
        mmu_dir=args.exclude_mmu_dir,
    )

    materialization_stats = {}
    if not args.manifest_only:
        (
            manifest,
            images,
            spectra,
            image_index,
            spectrum_index,
            materialization_stats,
        ) = prepare_payloads(
            manifest,
            image_dataset=args.image_dataset,
            spectrum_dataset=args.spectrum_dataset,
        )
        write_pair_shards(
            manifest,
            images,
            spectra,
            image_index,
            spectrum_index,
            output_dir=args.output_dir,
            rows_per_shard=args.rows_per_shard,
        )

    pq.write_table(
        manifest,
        args.output_dir / "manifest.parquet",
        compression="zstd",
    )
    stats["split_rows"] = {
        split: int(pc.sum(pc.equal(manifest["split"], split)).as_py())
        for split in ("train", "validation", "test")
    }
    stats["materialized_pair_rows"] = manifest.num_rows
    stats["materialized_unique_images"] = len(
        set(manifest["dr8_id"].to_pylist())
    )

    info = {
        **stats,
        **exclusion_stats,
        **materialization_stats,
        "matched_pair_rows_before_exclusions": matched_pair_rows_before_exclusions,
        "image_metadata": str(args.image_metadata),
        "image_dataset": str(args.image_dataset),
        "spectrum_dataset": str(args.spectrum_dataset),
        "exclude_astroclip_dir": str(args.exclude_astroclip_dir),
        "exclude_mmu_dir": str(args.exclude_mmu_dir),
        "output_dir": str(args.output_dir),
        "rows_per_shard": args.rows_per_shard,
        "materialized": not args.manifest_only,
        "builder_pid": os.getpid(),
    }
    with (args.output_dir / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
        handle.write("\n")

    print(f"Paired dataset ready at {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
