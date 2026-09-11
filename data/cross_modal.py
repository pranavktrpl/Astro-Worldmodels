import hashlib
import io
import math
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from PIL import Image

from .AstroTransforms import AstroMultiCropTransform
from .SpectraTransforms import AstroSpectraV2Transform


MMU_XMATCH_FORMAT = "mmu_desi_legacysurvey_xmatch"


def cross_modal_split_for_identity(identity, seed=42):
    digest = hashlib.blake2b(
        f"{seed}:{identity}".encode("utf-8"),
        digest_size=8,
        person=b"AstroJCM",
    ).digest()
    bucket = int.from_bytes(digest, byteorder="big") % 100
    if bucket < 98:
        return "train"
    if bucket == 98:
        return "validation"
    return "test"


def cross_modal_split_for_dr8_id(dr8_id, seed=42):
    return cross_modal_split_for_identity(dr8_id, seed)


def angular_separation_arcsec(ra1, dec1, ra2, dec2):
    ra1, dec1, ra2, dec2 = map(math.radians, (ra1, dec1, ra2, dec2))
    delta_ra = ra2 - ra1
    delta_dec = dec2 - dec1
    haversine = (
        math.sin(delta_dec / 2.0) ** 2
        + math.cos(dec1) * math.cos(dec2) * math.sin(delta_ra / 2.0) ** 2
    )
    angle = 2.0 * math.asin(math.sqrt(min(1.0, max(0.0, haversine))))
    return math.degrees(angle) * 3_600.0


def build_mmu_xmatch_metadata(dataset_dir, split_seed=42):
    dataset_dir = Path(dataset_dir)
    files = sorted((dataset_dir / "data").glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No MMU cross-match shards found under {dataset_dir / 'data'}"
        )

    split_rows = {split: 0 for split in ("train", "validation", "test")}
    file_split_rows = {}
    for path in files:
        identities = pq.read_table(
            path,
            columns=["object_id_ls"],
        )["object_id_ls"].to_pylist()
        counts = {split: 0 for split in split_rows}
        for identity in identities:
            if identity is None:
                raise ValueError(f"Null object_id_ls in {path}")
            split = cross_modal_split_for_identity(identity, split_seed)
            counts[split] += 1
            split_rows[split] += 1
        file_split_rows[str(path.relative_to(dataset_dir))] = counts

    return {
        "dataset_format": MMU_XMATCH_FORMAT,
        "dataset_dir": str(dataset_dir),
        "split_seed": split_seed,
        "pair_rows": sum(split_rows.values()),
        "split_rows": split_rows,
        "parquet_shards": len(files),
        "image_column": "rgb",
        "spectrum_column": "spectrum",
        "split_identity_column": "object_id_ls",
        "file_split_rows": file_split_rows,
    }


def decode_image(value):
    if isinstance(value, Image.Image):
        return value

    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
        if image_bytes is not None:
            with Image.open(io.BytesIO(image_bytes)) as image:
                return image.convert("RGB")
        if image_path is not None:
            with Image.open(image_path) as image:
                return image.convert("RGB")

    raise TypeError(f"Unsupported cached image value: {type(value)!r}")


class PairedImageSpectrumDataset(torch.utils.data.IterableDataset):
    """Stream externally cached, scientifically matched image-spectrum pairs."""

    def __init__(
        self,
        dataset_dir,
        split="train",
        shuffle=True,
        world_size=1,
        rank=0,
        num_workers=1,
        shuffle_buffer_size=10_000,
        split_seed=42,
        spectra_num_views=2,
        spectra_patch_size=20,
        spectra_num_pixels=7781,
        spectra_mask_ratio=0.30,
        spectra_mask_span=(2, 12),
        spectra_disjoint_view_masks=True,
        spectra_min_valid_patch_fraction=0.50,
        spectra_noise_scale=0.50,
        spectra_max_normalized_noise_std=3.0,
        dataset_info=None,
    ):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown paired split: {split}")

        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.shuffle = shuffle
        self.world_size = world_size
        self.rank = rank
        self.loader_num_workers = max(1, num_workers)
        self.shuffle_buffer_size = shuffle_buffer_size
        self.split_seed = split_seed
        self.epoch = 0

        materialized_files = sorted(
            (self.dataset_dir / "pairs").glob(f"{split}-*.parquet")
        )
        native_files = sorted(
            (self.dataset_dir / "data").glob("train-*.parquet")
        )
        if materialized_files:
            self.dataset_format = "materialized_dr8_desi"
            self.files = materialized_files
            file_rows = {
                path: pq.ParquetFile(path).metadata.num_rows
                for path in self.files
            }
        elif native_files:
            if (
                not dataset_info
                or dataset_info.get("dataset_format") != MMU_XMATCH_FORMAT
            ):
                raise ValueError(
                    "MMU cross-match loading requires indexed dataset metadata"
                )
            self.dataset_format = MMU_XMATCH_FORMAT
            relative_counts = dataset_info["file_split_rows"]
            file_rows = {
                path: int(
                    relative_counts[str(path.relative_to(self.dataset_dir))][split]
                )
                for path in native_files
            }
            self.files = [path for path in native_files if file_rows[path] > 0]
        else:
            raise FileNotFoundError(
                f"No supported paired shards found under {self.dataset_dir}"
            )

        rank_files, rank_rows = self._balanced_file_shards(
            self.files,
            self.world_size,
            file_rows,
        )
        self.rows_per_rank = rank_rows
        self.min_rows_per_rank = min(rank_rows)
        self.file_shards, self.worker_rows = self._balanced_file_shards(
            rank_files[self.rank],
            self.loader_num_workers,
            file_rows,
        )

        self.image_transform = AstroMultiCropTransform(Vl=0, Vg=2)
        self.spectrum_transform = AstroSpectraV2Transform(
            num_views=spectra_num_views,
            patch_size=spectra_patch_size,
            expected_num_pixels=spectra_num_pixels,
            mask_ratio=spectra_mask_ratio,
            mask_span=spectra_mask_span,
            disjoint_view_masks=spectra_disjoint_view_masks,
            min_valid_patch_fraction=spectra_min_valid_patch_fraction,
            noise_scale=spectra_noise_scale,
            max_normalized_noise_std=spectra_max_normalized_noise_std,
        )

    @staticmethod
    def _balanced_file_shards(files, num_shards, file_rows):
        if num_shards < 1:
            raise ValueError("num_shards must be positive")
        if len(files) < num_shards:
            raise ValueError(
                f"Need at least {num_shards} paired shards, found {len(files)}"
            )

        assignments = [[] for _ in range(num_shards)]
        assigned_rows = [0 for _ in range(num_shards)]
        files_with_rows = [(path, file_rows[path]) for path in files]
        for path, rows in sorted(
            files_with_rows,
            key=lambda item: item[1],
            reverse=True,
        ):
            shard_id = min(range(num_shards), key=assigned_rows.__getitem__)
            assignments[shard_id].append(path)
            assigned_rows[shard_id] += rows
        return assignments, assigned_rows

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        if num_workers != self.loader_num_workers:
            raise RuntimeError(
                f"Configured {self.loader_num_workers} loader workers, got {num_workers}"
            )

        data_files = [str(path) for path in self.file_shards[worker_id]]
        if self.dataset_format == MMU_XMATCH_FORMAT:
            columns = [
                "rgb",
                "spectrum",
                "object_id_ls",
                "object_id_spec",
                "ra_spec",
                "dec_spec",
                "ra_ls",
                "dec_ls",
            ]
        else:
            columns = [
                "image_crop",
                "spectrum",
                "dr8_id",
                "object_id",
                "separation_arcsec",
            ]
        stream = load_dataset(
            "parquet",
            data_files={"train": data_files},
            columns=columns,
            split="train",
            streaming=True,
        )
        if self.shuffle:
            stream = stream.shuffle(
                seed=self.split_seed,
                buffer_size=self.shuffle_buffer_size,
            )
        stream.set_epoch(self.epoch)

        for sample in stream:
            if self.dataset_format == MMU_XMATCH_FORMAT:
                image_id = sample["object_id_ls"]
                if (
                    cross_modal_split_for_identity(image_id, self.split_seed)
                    != self.split
                ):
                    continue
                image_value = sample["rgb"]
                spectrum_id = sample["object_id_spec"]
                separation_arcsec = angular_separation_arcsec(
                    sample["ra_spec"],
                    sample["dec_spec"],
                    sample["ra_ls"],
                    sample["dec_ls"],
                )
            else:
                image_id = sample["dr8_id"]
                image_value = sample["image_crop"]
                spectrum_id = sample["object_id"]
                separation_arcsec = sample["separation_arcsec"]

            image = decode_image(image_value)
            image_views = self.image_transform.augment_image(image)["global_crops"]
            spectrum = self.spectrum_transform.augment_spectrum(sample["spectrum"])
            if spectrum is None:
                continue

            yield {
                "image_views": image_views,
                "spectrum_views": spectrum["views"],
                "spectrum_valid_pixels": spectrum["valid_pixels"],
                "spectrum_jepa_masks": spectrum["jepa_masks"],
                "normalization_mean": spectrum["normalization_mean"],
                "normalization_log_std": spectrum["normalization_log_std"],
                "normalization_scale_was_floored": spectrum[
                    "normalization_scale_was_floored"
                ],
                "valid_pixel_fraction": spectrum["valid_pixel_fraction"],
                "image_id": image_id,
                "object_id": spectrum_id,
                "separation_arcsec": torch.tensor(
                    separation_arcsec,
                    dtype=torch.float32,
                ),
            }

ASTROCLIP_FORMAT = "astroclip_desi_flux_only"
MATERIALIZED_DR8_DESI_FORMAT = "materialized_dr8_desi"
COMBINED_DESI_FORMAT = "combined_desi_cross_modal"


def astroclip_raw_to_rgb(value, m=0.03, q=20.0):
    """Convert AstroCLIP's raw g/r/z nanomaggies to Legacy Survey RGB."""
    images = np.asarray(value, dtype=np.float32)
    if images.ndim != 3:
        raise ValueError(f"Expected a 3D AstroCLIP image, got {images.shape}")
    if images.shape[0] != 3 and images.shape[-1] == 3:
        images = np.transpose(images, (2, 0, 1))
    if images.shape[0] != 3:
        raise ValueError(f"Expected three AstroCLIP bands, got {images.shape}")

    rgb_scales = {
        "g": (2, 6.0),
        "r": (1, 3.4),
        "z": (0, 2.2),
    }
    intensity = np.zeros(images.shape[1:], dtype=np.float32)
    for image, band in zip(images, ("g", "r", "z")):
        _, scale = rgb_scales[band]
        intensity += np.maximum(0.0, image * scale + m)
    intensity /= 3.0

    stretch = np.arcsinh(q * intensity) / np.sqrt(q)
    denominator = intensity + (intensity == 0.0) * 1.0e-6
    rgb = np.zeros((*intensity.shape, 3), dtype=np.float32)
    for image, band in zip(images, ("g", "r", "z")):
        plane, scale = rgb_scales[band]
        rgb[..., plane] = (image * scale + m) * stretch / denominator
    return np.clip(rgb, 0.0, 1.0)


def _discover_source(dataset_dir, source_index):
    dataset_dir = Path(dataset_dir)
    materialized_files = sorted((dataset_dir / "pairs").glob("*.parquet"))
    native_files = sorted((dataset_dir / "data").glob("*.parquet"))

    if materialized_files:
        dataset_format = MATERIALIZED_DR8_DESI_FORMAT
        files = materialized_files
    elif native_files:
        columns = set(pq.ParquetFile(native_files[0]).schema_arrow.names)
        if {"image", "spectrum", "targetid"}.issubset(columns):
            dataset_format = ASTROCLIP_FORMAT
        elif {"rgb", "spectrum", "object_id_spec"}.issubset(columns):
            dataset_format = MMU_XMATCH_FORMAT
        else:
            raise ValueError(
                f"Could not infer paired DESI format for {dataset_dir}; "
                f"first-shard columns include {sorted(columns)[:12]}"
            )
        files = native_files
    else:
        raise FileNotFoundError(
            f"No paired Parquet shards found under {dataset_dir}"
        )

    records = []
    source_rows = 0
    for path in files:
        rows = pq.ParquetFile(path).metadata.num_rows
        source_rows += rows
        records.append(
            {
                "path": str(path),
                "rows": rows,
                "dataset_format": dataset_format,
                "source_index": source_index,
                "source_name": dataset_dir.name,
            }
        )
    return records, {
        "dataset_dir": str(dataset_dir),
        "dataset_format": dataset_format,
        "pair_rows": source_rows,
        "parquet_shards": len(files),
        "source_index": source_index,
    }


def build_combined_desi_metadata(dataset_dirs):
    normalized = [str(Path(path).resolve()) for path in dataset_dirs]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Combined paired dataset directories must be unique")

    files = []
    sources = []
    for source_index, dataset_dir in enumerate(normalized):
        source_files, source_info = _discover_source(dataset_dir, source_index)
        files.extend(source_files)
        sources.append(source_info)

    return {
        "dataset_format": COMBINED_DESI_FORMAT,
        "dataset_dirs": normalized,
        "pair_rows": sum(source["pair_rows"] for source in sources),
        "parquet_shards": len(files),
        "sources": sources,
        "files": files,
        "uses_all_rows": True,
    }


class CombinedPairedImageSpectrumDataset(torch.utils.data.IterableDataset):
    """Stream all rows from harmonized AstroCLIP, MMU, and manual DESI pairs."""

    def __init__(
        self,
        dataset_info,
        world_size=1,
        rank=0,
        num_workers=1,
        split_seed=42,
        parquet_batch_size=64,
        spectra_num_views=2,
        spectra_patch_size=20,
        spectra_num_pixels=7781,
        spectra_mask_ratio=0.30,
        spectra_mask_span=(2, 12),
        spectra_disjoint_view_masks=True,
        spectra_min_valid_patch_fraction=0.50,
        spectra_noise_scale=0.50,
        spectra_max_normalized_noise_std=3.0,
    ):
        if dataset_info.get("dataset_format") != COMBINED_DESI_FORMAT:
            raise ValueError("Combined DESI loading requires combined metadata")
        if parquet_batch_size < 1:
            raise ValueError("parquet_batch_size must be positive")

        self.dataset_info = dataset_info
        self.world_size = world_size
        self.rank = rank
        self.loader_num_workers = max(1, num_workers)
        self.split_seed = split_seed
        self.parquet_batch_size = parquet_batch_size
        self.epoch = 0

        all_files = list(dataset_info["files"])
        rank_files, rank_rows = self._balanced_file_shards(
            all_files,
            world_size,
        )
        self.rows_per_rank = rank_rows
        self.file_shards_by_rank = []
        self.worker_rows_by_rank = []
        for files_for_rank in rank_files:
            worker_files, worker_rows = self._balanced_file_shards(
                files_for_rank,
                self.loader_num_workers,
            )
            self.file_shards_by_rank.append(worker_files)
            self.worker_rows_by_rank.append(worker_rows)
        self.file_shards = self.file_shards_by_rank[rank]
        self.worker_rows = self.worker_rows_by_rank[rank]

        self.image_transform = AstroMultiCropTransform(Vl=0, Vg=2)
        self.spectrum_transform = AstroSpectraV2Transform(
            num_views=spectra_num_views,
            patch_size=spectra_patch_size,
            expected_num_pixels=spectra_num_pixels,
            mask_ratio=spectra_mask_ratio,
            mask_span=spectra_mask_span,
            disjoint_view_masks=spectra_disjoint_view_masks,
            min_valid_patch_fraction=spectra_min_valid_patch_fraction,
            noise_scale=spectra_noise_scale,
            max_normalized_noise_std=spectra_max_normalized_noise_std,
            allow_flux_only=True,
        )

    @staticmethod
    def _balanced_file_shards(files, num_shards):
        if num_shards < 1:
            raise ValueError("num_shards must be positive")
        if len(files) < num_shards:
            raise ValueError(
                f"Need at least {num_shards} paired shards, found {len(files)}"
            )

        assignments = [[] for _ in range(num_shards)]
        assigned_rows = [0 for _ in range(num_shards)]
        for record in sorted(files, key=lambda item: item["rows"], reverse=True):
            shard_id = min(range(num_shards), key=assigned_rows.__getitem__)
            assignments[shard_id].append(record)
            assigned_rows[shard_id] += record["rows"]
        return assignments, assigned_rows

    def usable_batches_per_rank(self, batch_size):
        return [
            sum(rows // batch_size for rows in worker_rows)
            for worker_rows in self.worker_rows_by_rank
        ]

    def set_epoch(self, epoch):
        self.epoch = epoch

    @staticmethod
    def _columns(dataset_format):
        if dataset_format == ASTROCLIP_FORMAT:
            return ["image", "spectrum", "targetid"]
        if dataset_format == MMU_XMATCH_FORMAT:
            return [
                "rgb",
                "spectrum",
                "object_id_ls",
                "object_id_spec",
                "ra_spec",
                "dec_spec",
                "ra_ls",
                "dec_ls",
            ]
        if dataset_format == MATERIALIZED_DR8_DESI_FORMAT:
            return [
                "image_crop",
                "spectrum",
                "dr8_id",
                "object_id",
                "separation_arcsec",
            ]
        raise ValueError(f"Unknown paired dataset format: {dataset_format}")

    def _iter_file(self, record, rng):
        parquet_file = pq.ParquetFile(record["path"])
        row_groups = list(range(parquet_file.metadata.num_row_groups))
        rng.shuffle(row_groups)
        for row_group in row_groups:
            batches = parquet_file.iter_batches(
                batch_size=self.parquet_batch_size,
                row_groups=[row_group],
                columns=self._columns(record["dataset_format"]),
                use_threads=False,
            )
            for batch in batches:
                samples = batch.to_pylist()
                rng.shuffle(samples)
                for sample in samples:
                    yield sample

    def _prepare_sample(self, sample, record):
        dataset_format = record["dataset_format"]
        if dataset_format == ASTROCLIP_FORMAT:
            image = astroclip_raw_to_rgb(sample["image"])
            image_id = str(sample["targetid"])
            spectrum_id = image_id
            spectrum_value = sample["spectrum"]
            separation_arcsec = float("nan")
        elif dataset_format == MMU_XMATCH_FORMAT:
            image = decode_image(sample["rgb"])
            image_id = str(sample["object_id_ls"])
            spectrum_id = str(sample["object_id_spec"])
            spectrum_value = sample["spectrum"]
            separation_arcsec = angular_separation_arcsec(
                sample["ra_spec"],
                sample["dec_spec"],
                sample["ra_ls"],
                sample["dec_ls"],
            )
        else:
            image = decode_image(sample["image_crop"])
            image_id = str(sample["dr8_id"])
            spectrum_id = str(sample["object_id"])
            spectrum_value = sample["spectrum"]
            separation_arcsec = float(sample["separation_arcsec"])

        image_views = self.image_transform.augment_image(image)["global_crops"]
        spectrum = self.spectrum_transform.augment_spectrum(spectrum_value)
        if spectrum is None:
            return None

        return {
            "image_views": image_views,
            "spectrum_views": spectrum["views"],
            "spectrum_valid_pixels": spectrum["valid_pixels"],
            "spectrum_jepa_masks": spectrum["jepa_masks"],
            "normalization_mean": spectrum["normalization_mean"],
            "normalization_log_std": spectrum["normalization_log_std"],
            "normalization_scale_was_floored": spectrum[
                "normalization_scale_was_floored"
            ],
            "valid_pixel_fraction": spectrum["valid_pixel_fraction"],
            "uncertainty_available": spectrum["uncertainty_available"],
            "image_id": image_id,
            "object_id": spectrum_id,
            "separation_arcsec": torch.tensor(
                separation_arcsec,
                dtype=torch.float32,
            ),
            "source_index": torch.tensor(
                record["source_index"],
                dtype=torch.int64,
            ),
        }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        if num_workers != self.loader_num_workers:
            raise RuntimeError(
                f"Configured {self.loader_num_workers} loader workers, got {num_workers}"
            )

        rng = random.Random(
            self.split_seed
            + 1_000_003 * self.epoch
            + 10_007 * self.rank
            + worker_id
        )
        records = list(self.file_shards[worker_id])
        rng.shuffle(records)
        for record in records:
            for sample in self._iter_file(record, rng):
                prepared = self._prepare_sample(sample, record)
                if prepared is not None:
                    yield prepared
