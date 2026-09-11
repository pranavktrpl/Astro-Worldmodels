#!/usr/bin/env python3
"""Evaluate the new spectrum-v2 and simultaneous cross-modal backbones.

The sample, shipped train/test split, image preprocessing, target definitions,
feature standardization, and ridge-selection protocol match the benchmark code
on the `redshift-regression-eval` branch. Large embeddings are cached outside
the repository by default; metrics and provenance remain in the repository.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
EVALS_DIR = SCRIPT_DIR.parent
REPO_ROOT = EVALS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from models.cross_modal_posttrained import CrossModalPosttrainedModel

from models.cross_modal import CrossModalImageEncoder, CrossModalSpectrumEncoder


DEFAULT_DATA_DIR = Path("/mnt/datasets/pranav/astroclip")
DEFAULT_PROVABGS_PATH = Path(
    "/mnt/datasets/utbd_pranav/catalogs/desi_provabgs/mmu_desi_provabgs/dataset"
)
DEFAULT_CACHE_DIR = Path("/mnt/datasets/pranav/astrojepa_eval_cache")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"

DEFAULT_CHECKPOINTS = {
    "crossmodal-image": REPO_ROOT
    / "checkpoints/CrossModalScratch_MMU95K_DR10_DESI_CrossOnly_SIGReg/last.pt",
    "crossmodal-spectrum": REPO_ROOT
    / "checkpoints/CrossModalScratch_MMU95K_DR10_DESI_CrossOnly_SIGReg/last.pt",
    "spectra-v2": REPO_ROOT
    / "checkpoints/SpectraV2_DESI_GlobalLocalLeJEPA_MeanStd_CorrectedEpochs/complete.pt",
    "posttrained-image": REPO_ROOT
    / "checkpoints/CrossModalPostTrain_DESI307K_AstroCLIPPool_LeJEPA_SIGReg/last.pt",
    "posttrained-spectrum": REPO_ROOT
    / "checkpoints/CrossModalPostTrain_DESI307K_AstroCLIPPool_LeJEPA_SIGReg/last.pt",
}

DEFAULT_LABELS = {
    "crossmodal-image": "crossmodal_scratch_step8396_image",
    "crossmodal-spectrum": "crossmodal_scratch_step8396_spectrum",
    "spectra-v2": "spectra_v2_complete",
    "posttrained-image": "crossmodal_posttrained_307k_last_image",
    "posttrained-spectrum": "crossmodal_posttrained_307k_last_spectrum",
}

IMAGE_ENCODERS = {"crossmodal-image", "posttrained-image"}

RIDGE_L2_GRID = [1e-4, 1e-2, 1.0, 1e2, 1e4]
IMAGE_SEEDS = list(range(42, 52))
SPECTRUM_SEEDS = [42, 43, 44]
OUTLIER_THRESHOLD = 0.05

DR2_RGB_SCALES = {"g": (2, 6.0), "r": (1, 3.4), "z": (0, 2.2)}
DR2_RGB_M = 0.03
DR2_RGB_Q = 20.0
DR2_BANDS = ("g", "r", "z")

PROPERTIES = ("stellar_mass", "metallicity", "age", "ssfr")
PROVABGS_COLUMNS = ["object_id", "LOG_MSTAR", "Z_MW", "TAGE_MW", "AVG_SFR"]
ASTROCLIP_SPECTRUM_R2 = {
    "redshift": 0.98,
    "stellar_mass": 0.88,
    "metallicity": 0.58,
    "age": 0.43,
    "ssfr": 0.64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=tuple(DEFAULT_CHECKPOINTS), required=True)
    parser.add_argument(
        "--stage",
        choices=("extract", "probe", "all"),
        default="all",
        help="Extract embeddings, run probes from a cache, or do both.",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--provabgs-path", type=Path, default=DEFAULT_PROVABGS_PATH)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-size", type=int, default=140)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def split_files(data_dir: Path) -> tuple[list[Path], list[Path]]:
    train = sorted((data_dir / "data").glob("train-*.parquet"))
    test = sorted((data_dir / "data").glob("test-*.parquet"))
    if len(train) != 120 or len(test) != 26:
        raise RuntimeError(
            f"Expected 120 train and 26 test AstroCLIP shards under {data_dir}; "
            f"found {len(train)} and {len(test)}"
        )
    return train, test


def shard_rows(files: Iterable[Path]) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in files)


def read_targets(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train, test = split_files(data_dir)
    redshifts: list[np.ndarray] = []
    targetids: list[np.ndarray] = []
    is_test: list[np.ndarray] = []
    for test_flag, files in ((False, train), (True, test)):
        for path in files:
            table = pq.read_table(path, columns=["redshift", "targetid"])
            redshifts.append(
                table.column("redshift").to_numpy(zero_copy_only=False).astype(np.float64)
            )
            targetids.append(
                table.column("targetid").to_numpy(zero_copy_only=False).astype(np.int64)
            )
            is_test.append(np.full(len(table), test_flag, dtype=bool))
    return np.concatenate(redshifts), np.concatenate(targetids), np.concatenate(is_test)


def dr2_rgb_batch(batch: np.ndarray) -> np.ndarray:
    """Map raw Legacy Survey grz fluxes to the report's RGB representation."""
    intensity = np.zeros(batch.shape[:3], dtype=np.float64)
    for index, band in enumerate(DR2_BANDS):
        _, scale = DR2_RGB_SCALES[band]
        intensity += np.maximum(
            0.0,
            batch[..., index].astype(np.float64) * scale + DR2_RGB_M,
        )
    intensity /= len(DR2_BANDS)
    stretch = np.arcsinh(DR2_RGB_Q * intensity) / np.sqrt(DR2_RGB_Q)
    safe_intensity = np.where(intensity == 0.0, 1e-6, intensity)
    rgb = np.zeros(batch.shape, dtype=np.float32)
    for index, band in enumerate(DR2_BANDS):
        plane, scale = DR2_RGB_SCALES[band]
        rgb[..., plane] = (
            (batch[..., index].astype(np.float64) * scale + DR2_RGB_M)
            * stretch
            / safe_intensity
        )
    return np.clip(rgb, 0.0, 1.0)


def iter_nested_array_batches(
    files: list[Path], column_name: str, batch_size: int, trailing_dims: int
):
    for path in files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=[column_name],
        ):
            column = batch.column(column_name)
            if hasattr(column, "storage"):
                column = column.storage
            flattened = column
            for _ in range(trailing_dims):
                flattened = flattened.flatten()
            values = flattened.to_numpy(zero_copy_only=False)
            yield values, len(column)


def iter_image_batches(files: list[Path], batch_size: int):
    for flat, count in iter_nested_array_batches(files, "image", batch_size, 3):
        side = int(round(math.sqrt(flat.size / (count * 3))))
        yield flat.reshape(count, side, side, 3).astype(np.float32, copy=False)


def iter_spectrum_batches(files: list[Path], batch_size: int):
    for flat, count in iter_nested_array_batches(files, "spectrum", batch_size, 2):
        yield flat.reshape(count, -1).astype(np.float32, copy=False)


def torch_load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )


def crossmodal_checkpoint_metadata(
    checkpoint: dict[str, Any], checkpoint_path: Path
) -> dict[str, Any]:
    config = dict(checkpoint["config"])
    global_step = int(checkpoint["global_step"])
    saved_epoch = int(checkpoint["epoch"])
    saved_step_in_epoch = int(checkpoint["step_in_epoch"])
    configured_total_steps = int(
        config.get(
            "total_steps",
            int(config["epochs"]) * int(config["steps_per_epoch"]),
        )
    )
    training_complete = (
        saved_epoch >= int(config["epochs"])
        and saved_step_in_epoch == 0
        and global_step >= configured_total_steps
    )
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "global_step": global_step,
        "saved_epoch": saved_epoch,
        "saved_step_in_epoch": saved_step_in_epoch,
        "config": config,
        "checkpoint_status": (
            "Completed training checkpoint."
            if training_complete
            else "Intermediate training checkpoint."
        ),
    }


def load_crossmodal_image(
    checkpoint_path: Path, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch_load_checkpoint(checkpoint_path)
    config = dict(checkpoint["config"])
    encoder = CrossModalImageEncoder(
        model_name=str(config["image_model_name"]),
        shared_dim=int(config["shared_dim"]),
        pretrained=False,
    )
    encoder.load_state_dict(checkpoint["image_encoder"], strict=True)
    metadata = crossmodal_checkpoint_metadata(checkpoint, checkpoint_path)
    metadata.update(
        {
            "encoder": "crossmodal-image",
            "raw_embedding_dim": int(encoder.embed_dim),
            "projected_embedding_dim": int(config["shared_dim"]),
        }
    )
    del checkpoint
    encoder.to(device).eval()
    return encoder, metadata


def load_crossmodal_spectrum(
    checkpoint_path: Path, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch_load_checkpoint(checkpoint_path)
    config = dict(checkpoint["config"])
    encoder = CrossModalSpectrumEncoder(
        shared_dim=int(config["shared_dim"]),
        patch_size=int(config["spectra_patch_size"]),
        num_patches=int(config["spectra_num_patches"]),
        embed_dim=int(config["spectra_embed_dim"]),
        depth=int(config["spectra_depth"]),
        num_heads=int(config["spectra_num_heads"]),
        mlp_ratio=float(config["spectra_mlp_ratio"]),
        dropout=float(config["spectra_dropout"]),
        pooling=str(config["spectra_pooling"]),
    )
    encoder.load_state_dict(checkpoint["spectrum_encoder"], strict=True)
    metadata = crossmodal_checkpoint_metadata(checkpoint, checkpoint_path)
    metadata.update(
        {
            "encoder": "crossmodal-spectrum",
            "raw_embedding_dim": int(encoder.embed_dim),
            "projected_embedding_dim": int(config["shared_dim"]),
        }
    )
    del checkpoint
    encoder.to(device).eval()
    return encoder, metadata


def load_spectra_v2(
    checkpoint_path: Path, device: torch.device
) -> tuple[nn.Module, dict[str, Any]]:
    train_spectra = load_module("train_spectra_v2_eval", REPO_ROOT / "train-spectra-v2.py")
    checkpoint = torch_load_checkpoint(checkpoint_path)
    config = dict(checkpoint["cfg"])
    encoder = train_spectra.SpectrumTransformerEncoderV2(
        proj_dim=int(config["proj_dim"]),
        patch_size=int(config["spectra_patch_size"]),
        num_patches=int(config["spectra_num_patches"]),
        embed_dim=int(config["spectra_embed_dim"]),
        depth=int(config["spectra_depth"]),
        num_heads=int(config["spectra_num_heads"]),
        mlp_ratio=float(config["spectra_mlp_ratio"]),
        dropout=float(config["spectra_dropout"]),
        pooling=str(config["spectra_pooling"]),
        local_proj_dim=int(config["spectra_local_proj_dim"]),
    )
    encoder.load_state_dict(checkpoint["model"], strict=True)
    global_step = int(checkpoint["global_step"])
    saved_epoch = int(checkpoint["epoch"])
    saved_step_in_epoch = int(checkpoint["step_in_epoch"])
    configured_total_steps = int(
        config.get(
            "total_steps",
            int(config["epochs"]) * int(config["steps_per_epoch"]),
        )
    )
    training_complete = (
        saved_epoch >= int(config["epochs"])
        and saved_step_in_epoch == 0
        and global_step >= configured_total_steps
    )
    metadata = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "global_step": global_step,
        "saved_epoch": saved_epoch,
        "saved_step_in_epoch": saved_step_in_epoch,
        "config": config,
        "encoder": "spectra-v2",
        "raw_embedding_dim": int(encoder.embed_dim),
        "projected_embedding_dim": int(config["proj_dim"]),
        "checkpoint_status": (
            "Completed training checkpoint."
            if training_complete
            else "Intermediate training checkpoint."
        ),
    }
    del checkpoint
    encoder.to(device).eval()
    return encoder, metadata


def load_posttrained(
    checkpoint_path: Path, device: torch.device
) -> tuple[CrossModalPosttrainedModel, dict[str, Any]]:
    checkpoint = torch_load_checkpoint(checkpoint_path)
    if checkpoint.get("format") != "cross_modal_posttrained_heads_v1":
        raise RuntimeError(
            f"Unsupported post-training checkpoint format: {checkpoint.get('format')}"
        )
    config = dict(checkpoint["config"])
    model = CrossModalPosttrainedModel(
        image_model_name=str(config["image_model_name"]),
        shared_dim=int(config["shared_dim"]),
        spectra_patch_size=int(config["spectra_patch_size"]),
        spectra_num_patches=int(config["spectra_num_patches"]),
        spectra_embed_dim=int(config["spectra_embed_dim"]),
        spectra_depth=int(config["spectra_depth"]),
        spectra_num_heads=int(config["spectra_num_heads"]),
        spectra_mlp_ratio=float(config["spectra_mlp_ratio"]),
        spectra_dropout=float(config["spectra_dropout"]),
        spectra_pooling=str(config["spectra_pooling"]),
        pool_num_heads=int(config["pool_num_heads"]),
        pool_dropout=float(config["pool_dropout"]),
    )
    recorded_sources = checkpoint["source_checkpoints"]
    loaded_sources = model.load_pretrained_backbones(
        recorded_sources["image"]["path"],
        recorded_sources["spectrum"]["path"],
    )
    if loaded_sources != recorded_sources:
        raise RuntimeError(
            "Post-trained checkpoint source provenance differs from the current files"
        )
    model.source_metadata = loaded_sources
    model.load_alignment_state_dict(checkpoint["alignment_heads"])

    global_step = int(checkpoint["global_step"])
    saved_epoch = int(checkpoint["epoch"])
    saved_step_in_epoch = int(checkpoint["step_in_epoch"])
    configured_total_steps = int(
        config.get(
            "total_steps",
            int(config["epochs"]) * int(config["steps_per_epoch"]),
        )
    )
    metadata = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "format": checkpoint["format"],
        "global_step": global_step,
        "saved_epoch": saved_epoch,
        "saved_step_in_epoch": saved_step_in_epoch,
        "config": config,
        "source_checkpoints": recorded_sources,
        "raw_image_embedding_dim": int(model.image_encoder.embed_dim),
        "raw_spectrum_embedding_dim": int(model.spectrum_encoder.embed_dim),
        "projected_embedding_dim": int(config["shared_dim"]),
        "checkpoint_status": (
            "Completed training checkpoint."
            if saved_epoch >= int(config["epochs"])
            and saved_step_in_epoch == 0
            and global_step >= configured_total_steps
            else "Intermediate training checkpoint."
        ),
    }
    del checkpoint
    model.to(device).eval()
    return model, metadata


def amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def projector_output_dim(projector: nn.Module) -> int:
    for module in reversed(list(projector.modules())):
        if isinstance(module, nn.Linear):
            return int(module.out_features)
    raise ValueError(f"No linear output layer found in {type(projector).__name__}")


def prepare_spectrum_batch(
    array: np.ndarray, patch_size: int, num_patches: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one deterministic, unmasked mean/std-normalized evaluation view."""
    flux = torch.from_numpy(array)
    valid = torch.isfinite(flux)
    valid_count = valid.sum(dim=1, keepdim=True).clamp_min(1)
    safe_flux = torch.where(valid, flux, torch.zeros_like(flux))
    location = safe_flux.sum(dim=1, keepdim=True) / valid_count
    centered = torch.where(valid, flux - location, torch.zeros_like(flux))
    variance = centered.square().sum(dim=1, keepdim=True) / valid_count
    scale = variance.sqrt().clamp_min(1e-6)
    normalized = torch.where(valid, centered / scale, torch.zeros_like(flux))

    usable = patch_size * num_patches
    views = normalized[:, :usable].reshape(-1, 1, num_patches, patch_size)
    valid_pixels = valid[:, :usable].reshape(-1, num_patches, patch_size)
    jepa_masks = torch.zeros(
        views.shape[0],
        1,
        num_patches,
        dtype=torch.bool,
    )
    return (
        views.to(device, non_blocking=True),
        valid_pixels.to(device, non_blocking=True),
        jepa_masks.to(device, non_blocking=True),
    )


def cache_paths(cache_dir: Path, label: str) -> dict[str, Path]:
    root = cache_dir / label
    return {
        "root": root,
        "raw": root / "raw.npy",
        "projected": root / "projected.npy",
        "metadata": root / "metadata.json",
    }


def open_embedding_memmaps(
    paths: dict[str, Path], total_rows: int, raw_dim: int, projected_dim: int
) -> dict[str, np.memmap]:
    paths["root"].mkdir(parents=True, exist_ok=True)
    return {
        "raw": np.lib.format.open_memmap(
            paths["raw"].with_suffix(".tmp.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(total_rows, raw_dim),
        ),
        "projected": np.lib.format.open_memmap(
            paths["projected"].with_suffix(".tmp.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(total_rows, projected_dim),
        ),
    }


def finalize_embedding_memmaps(
    arrays: dict[str, np.memmap], paths: dict[str, Path]
) -> None:
    for name, array in arrays.items():
        array.flush()
        del array
        os.replace(paths[name].with_suffix(".tmp.npy"), paths[name])


@torch.inference_mode()
def extract_image_embeddings(
    encoder: CrossModalImageEncoder,
    files: list[Path],
    total_rows: int,
    batch_size: int,
    input_size: int,
    device: torch.device,
    paths: dict[str, Path],
) -> None:
    arrays = open_embedding_memmaps(
        paths,
        total_rows,
        int(encoder.embed_dim),
        projector_output_dim(encoder.shared_proj),
    )
    offset = 0
    progress = tqdm(total=total_rows, desc="cross-modal image embeddings")
    for image_array in iter_image_batches(files, batch_size):
        rgb = dr2_rgb_batch(image_array)
        batch = torch.from_numpy(rgb).permute(0, 3, 1, 2)
        batch = batch.to(device=device, dtype=torch.float32, non_blocking=True)
        batch = F.interpolate(
            batch,
            size=(input_size, input_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        with amp_context(device):
            raw = encoder.backbone(batch)
            projected = encoder.shared_proj(raw)
        stop = offset + len(image_array)
        arrays["raw"][offset:stop] = raw.float().cpu().numpy()
        arrays["projected"][offset:stop] = projected.float().cpu().numpy()
        offset = stop
        progress.update(len(image_array))
    progress.close()
    if offset != total_rows:
        raise RuntimeError(f"Extracted {offset} image rows; expected {total_rows}")
    finalize_embedding_memmaps(arrays, paths)


@torch.inference_mode()
def extract_spectrum_embeddings(
    encoder: nn.Module,
    files: list[Path],
    total_rows: int,
    batch_size: int,
    device: torch.device,
    paths: dict[str, Path],
    encoder_kind: str,
) -> None:
    projected_dim = (
        projector_output_dim(encoder.shared_proj)
        if encoder_kind == "crossmodal-spectrum"
        else projector_output_dim(encoder.global_proj)
    )
    arrays = open_embedding_memmaps(
        paths,
        total_rows,
        int(encoder.embed_dim),
        projected_dim,
    )
    offset = 0
    progress = tqdm(total=total_rows, desc=f"{encoder_kind} embeddings")
    for spectrum_array in iter_spectrum_batches(files, batch_size):
        views, valid_pixels, jepa_masks = prepare_spectrum_batch(
            spectrum_array,
            patch_size=int(encoder.patch_size),
            num_patches=int(encoder.num_patches),
            device=device,
        )
        with amp_context(device):
            outputs = encoder(views, valid_pixels, jepa_masks)
        raw, projected = outputs[0][0], outputs[1][0]
        stop = offset + len(spectrum_array)
        arrays["raw"][offset:stop] = raw.float().cpu().numpy()
        arrays["projected"][offset:stop] = projected.float().cpu().numpy()
        offset = stop
        progress.update(len(spectrum_array))
    progress.close()
    if offset != total_rows:
        raise RuntimeError(f"Extracted {offset} spectrum rows; expected {total_rows}")
    finalize_embedding_memmaps(arrays, paths)


@torch.inference_mode()
def extract_posttrained_image_embeddings(
    model: CrossModalPosttrainedModel,
    files: list[Path],
    total_rows: int,
    batch_size: int,
    input_size: int,
    device: torch.device,
    paths: dict[str, Path],
) -> None:
    arrays = open_embedding_memmaps(
        paths,
        total_rows,
        int(model.image_encoder.embed_dim),
        projector_output_dim(model.image_encoder.shared_proj),
    )
    offset = 0
    progress = tqdm(total=total_rows, desc="post-trained image embeddings")
    for image_array in iter_image_batches(files, batch_size):
        rgb = dr2_rgb_batch(image_array)
        batch = torch.from_numpy(rgb).permute(0, 3, 1, 2)
        batch = batch.to(device=device, dtype=torch.float32, non_blocking=True)
        batch = F.interpolate(
            batch,
            size=(input_size, input_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        with amp_context(device):
            raw, projected = model._encode_image_tokens(batch[:, None])
        raw, projected = raw[0], projected[0]
        stop = offset + len(image_array)
        arrays["raw"][offset:stop] = raw.float().cpu().numpy()
        arrays["projected"][offset:stop] = projected.float().cpu().numpy()
        offset = stop
        progress.update(len(image_array))
    progress.close()
    if offset != total_rows:
        raise RuntimeError(f"Extracted {offset} image rows; expected {total_rows}")
    finalize_embedding_memmaps(arrays, paths)


@torch.inference_mode()
def extract_posttrained_spectrum_embeddings(
    model: CrossModalPosttrainedModel,
    files: list[Path],
    total_rows: int,
    batch_size: int,
    device: torch.device,
    paths: dict[str, Path],
) -> None:
    encoder = model.spectrum_encoder
    arrays = open_embedding_memmaps(
        paths,
        total_rows,
        int(encoder.embed_dim),
        projector_output_dim(encoder.shared_proj),
    )
    offset = 0
    progress = tqdm(total=total_rows, desc="post-trained spectrum embeddings")
    for spectrum_array in iter_spectrum_batches(files, batch_size):
        views, valid_pixels, jepa_masks = prepare_spectrum_batch(
            spectrum_array,
            patch_size=int(encoder.patch_size),
            num_patches=int(encoder.num_patches),
            device=device,
        )
        with amp_context(device):
            raw, projected = model._encode_spectrum_tokens(
                views, valid_pixels, jepa_masks
            )
        raw, projected = raw[0], projected[0]
        stop = offset + len(spectrum_array)
        arrays["raw"][offset:stop] = raw.float().cpu().numpy()
        arrays["projected"][offset:stop] = projected.float().cpu().numpy()
        offset = stop
        progress.update(len(spectrum_array))
    progress.close()
    if offset != total_rows:
        raise RuntimeError(f"Extracted {offset} spectrum rows; expected {total_rows}")
    finalize_embedding_memmaps(arrays, paths)


def extract(args: argparse.Namespace, checkpoint: Path, label: str) -> dict[str, Any]:
    paths = cache_paths(args.cache_dir, label)
    if (
        paths["raw"].exists()
        and paths["projected"].exists()
        and paths["metadata"].exists()
        and not args.overwrite
    ):
        print(f"Using cached embeddings under {paths['root']}")
        return json.loads(paths["metadata"].read_text())

    train_files, test_files = split_files(args.data_dir)
    files = train_files + test_files
    train_rows = shard_rows(train_files)
    test_rows = shard_rows(test_files)
    total_rows = train_rows + test_rows
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")

    if args.encoder in IMAGE_ENCODERS:
        batch_size = args.batch_size or 256
        if args.encoder == "posttrained-image":
            encoder, metadata = load_posttrained(checkpoint, device)
            metadata["encoder"] = args.encoder
            extract_posttrained_image_embeddings(
                encoder,
                files,
                total_rows,
                batch_size,
                args.input_size,
                device,
                paths,
            )
        else:
            encoder, metadata = load_crossmodal_image(checkpoint, device)
            extract_image_embeddings(
                encoder,
                files,
                total_rows,
                batch_size,
                args.input_size,
                device,
                paths,
            )
        preprocessing = {
            "image": "dr2_rgb_arcsinh_to_0_1_then_bicubic_resize",
            "input_size": args.input_size,
            "dr2_rgb": {
                "scales": DR2_RGB_SCALES,
                "m": DR2_RGB_M,
                "q": DR2_RGB_Q,
            },
        }
    else:
        batch_size = args.batch_size or 128
        if args.encoder == "crossmodal-spectrum":
            encoder, metadata = load_crossmodal_spectrum(checkpoint, device)
        elif args.encoder == "posttrained-spectrum":
            encoder, metadata = load_posttrained(checkpoint, device)
            metadata["encoder"] = args.encoder
        else:
            encoder, metadata = load_spectra_v2(checkpoint, device)
        if args.encoder == "posttrained-spectrum":
            extract_posttrained_spectrum_embeddings(
                encoder,
                files,
                total_rows,
                batch_size,
                device,
                paths,
            )
        else:
            extract_spectrum_embeddings(
                encoder,
                files,
                total_rows,
                batch_size,
                device,
                paths,
                args.encoder,
            )
        preprocessing = {
            "spectrum": (
                "finite-flux mean/std normalization; first 7780 values; "
                "389x20 patches; one noise-free unmasked view"
            ),
            "validity_limitation": (
                "The exact AstroCLIP comparison parquet provides flux only. "
                "Finite flux values are treated as valid because ivar and "
                "DESI pipeline masks are unavailable in this mirror."
            ),
        }

    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    metadata.update(
        {
            "label": label,
            "data_dir": str(args.data_dir.resolve()),
            "dataset": "mhsotoudeh/astroclip parquet mirror",
            "train_rows": train_rows,
            "test_rows": test_rows,
            "total_rows": total_rows,
            "train_shards": len(train_files),
            "test_shards": len(test_files),
            "row_order": "all train shards by filename, then all test shards by filename",
            "preprocessing": preprocessing,
            "embedding_paths": {
                name: str(paths[name].resolve()) for name in ("raw", "projected")
            },
            "extracted_at_unix": time.time(),
        }
    )
    atomic_json(paths["metadata"], metadata)
    return metadata


def regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    residuals = predictions - targets
    scaled = residuals / (1.0 + targets)
    denominator = float(((targets - targets.mean()) ** 2).sum())
    return {
        "r2": float(1.0 - (residuals**2).sum() / max(denominator, 1e-12)),
        "mae": float(np.abs(residuals).mean()),
        "rmse": float(np.sqrt((residuals**2).mean())),
        "nmad": float(1.4826 * np.median(np.abs(scaled - np.median(scaled)))),
        "outlier_fraction": float((np.abs(scaled) > OUTLIER_THRESHOLD).mean()),
        "num_examples": int(len(targets)),
    }


def physical_regression_metrics(
    predictions: np.ndarray, targets: np.ndarray
) -> dict[str, float]:
    metrics = regression_metrics(predictions, targets)
    del metrics["nmad"], metrics["outlier_fraction"]
    return metrics


def fit_ridge(
    train_x: torch.Tensor, train_y: torch.Tensor, l2_value: float
) -> tuple[torch.Tensor, float]:
    y_mean = train_y.mean()
    x = train_x.double()
    centered = (train_y - y_mean).double()
    gram = x.T @ x
    gram.diagonal().add_(l2_value)
    weights = torch.linalg.solve(gram, x.T @ centered)
    return weights, float(y_mean)


def predict_ridge(
    features: torch.Tensor, weights: torch.Tensor, y_mean: float
) -> np.ndarray:
    return (features.double() @ weights + y_mean).cpu().numpy()


def evaluate_ridge_seeds(
    train_x: torch.Tensor,
    train_y: np.ndarray,
    test_x: torch.Tensor,
    test_y: np.ndarray,
    seeds: list[int],
    physical_target: bool = False,
) -> dict[str, Any]:
    metric_fn = physical_regression_metrics if physical_target else regression_metrics
    repeats: list[dict[str, Any]] = []
    for seed in seeds:
        order = np.random.RandomState(seed).permutation(len(train_y))
        validation_count = max(1, int(0.1 * len(order)))
        validation_index = order[:validation_count]
        fit_index = order[validation_count:]
        fit_x = train_x[torch.from_numpy(fit_index).to(train_x.device)]
        validation_x = train_x[
            torch.from_numpy(validation_index).to(train_x.device)
        ]
        fit_y = torch.from_numpy(train_y[fit_index]).to(
            train_x.device,
            dtype=torch.float32,
        )
        validation_y = train_y[validation_index]

        candidates: list[dict[str, float]] = []
        best: tuple[float, float, torch.Tensor, float] | None = None
        for l2_value in RIDGE_L2_GRID:
            weights, y_mean = fit_ridge(fit_x, fit_y, l2_value)
            predictions = predict_ridge(validation_x, weights, y_mean)
            validation_r2 = metric_fn(predictions, validation_y)["r2"]
            candidates.append({"l2": l2_value, "validation_r2": validation_r2})
            if best is None or validation_r2 > best[0]:
                best = (validation_r2, l2_value, weights, y_mean)
        assert best is not None
        validation_predictions = predict_ridge(validation_x, best[2], best[3])
        test_predictions = predict_ridge(test_x, best[2], best[3])
        repeat = {
            "seed": seed,
            "selected_l2": best[1],
            "candidates": candidates,
            "validation": metric_fn(validation_predictions, validation_y),
            "test": metric_fn(test_predictions, test_y),
        }
        repeats.append(repeat)
        print(
            f"seed={seed} l2={best[1]:g} "
            f"validation R2={repeat['validation']['r2']:.5f} "
            f"test R2={repeat['test']['r2']:.5f}"
        )

    metric_names = ("r2", "mae", "rmse")
    if not physical_target:
        metric_names += ("nmad", "outlier_fraction")
    summary: dict[str, Any] = {}
    for split in ("validation", "test"):
        summary[split] = {}
        for metric_name in metric_names:
            values = [repeat[split][metric_name] for repeat in repeats]
            summary[split][metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }
    return {"repeats": repeats, "summary": summary}


def standardize_features(
    embeddings: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    train_cpu = torch.from_numpy(np.asarray(embeddings[train_index])).float()
    mean = train_cpu.mean(dim=0)
    std = train_cpu.std(dim=0).clamp_min(1e-6)
    train_x = ((train_cpu - mean) / std).to(device)
    test_cpu = torch.from_numpy(np.asarray(embeddings[test_index])).float()
    test_x = ((test_cpu - mean) / std).to(device)
    return train_x, test_x, {
        "feature_dim": int(train_x.shape[1]),
        "standardization": "shipped-train-split mean/std; std clamped at 1e-6",
        "prestandardization_mean_feature_std": float(std.mean()),
        "prestandardization_min_feature_std": float(std.min()),
        "prestandardization_max_feature_std": float(std.max()),
        "near_constant_dimensions": int((std <= 1.000001e-6).sum()),
    }


def embedding_geometry(train_x: torch.Tensor) -> dict[str, Any]:
    with torch.inference_mode():
        covariance = (train_x.T @ train_x) / max(1, train_x.shape[0] - 1)
        eigenvalues = torch.linalg.eigvalsh(covariance.float()).clamp_min(0).cpu()
    total = float(eigenvalues.sum())
    if total <= 0:
        return {
            "effective_rank": 0.0,
            "participation_ratio": 0.0,
            "largest_eigenvalue_fraction": 0.0,
        }
    probabilities = eigenvalues / total
    nonzero = probabilities > 0
    entropy = float(-(probabilities[nonzero] * probabilities[nonzero].log()).sum())
    return {
        "effective_rank": float(math.exp(entropy)),
        "participation_ratio": float(total**2 / eigenvalues.square().sum()),
        "largest_eigenvalue_fraction": float(eigenvalues[-1] / total),
        "positive_eigenvalues": int((eigenvalues > 1e-7).sum()),
        "feature_dim": int(train_x.shape[1]),
        "definition": (
            "Eigenvalues of the correlation matrix after train-split per-feature "
            "standardization; effective rank is exp(Shannon entropy)."
        ),
    }


def read_provabgs_catalog(path: Path) -> dict[int, np.ndarray]:
    shards = sorted(path.rglob("Npix=*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No MMU PROVABGS parquet shards under {path}")
    catalog: dict[int, np.ndarray] = {}
    for shard in tqdm(shards, desc="PROVABGS catalog", leave=False):
        table = pq.read_table(shard, columns=PROVABGS_COLUMNS)
        object_ids = table.column("object_id").to_pylist()
        values = np.stack(
            [
                table.column(name)
                .to_numpy(zero_copy_only=False)
                .astype(np.float64)
                for name in PROVABGS_COLUMNS[1:]
            ],
            axis=1,
        )
        for object_id, value in zip(object_ids, values):
            catalog.setdefault(int(object_id), value)
    return catalog


def build_property_targets(
    targetids: np.ndarray, catalog: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    raw = np.full((len(targetids), 4), np.nan)
    matched = np.zeros(len(targetids), dtype=bool)
    for index, targetid in enumerate(targetids):
        value = catalog.get(int(targetid))
        if value is not None:
            raw[index] = value
            matched[index] = True
    log_mstar, z_mw, age, average_sfr = raw.T
    valid = (
        matched
        & np.isfinite(raw).all(axis=1)
        & (log_mstar > 0)
        & (z_mw > 0)
        & (age > 0)
        & (average_sfr > 0)
    )
    targets = np.full((len(targetids), len(PROPERTIES)), np.nan)
    targets[valid, 0] = log_mstar[valid]
    targets[valid, 1] = np.log10(z_mw[valid])
    targets[valid, 2] = age[valid]
    targets[valid, 3] = np.log10(average_sfr[valid]) - log_mstar[valid]
    return targets, valid, {
        "catalog_rows": len(catalog),
        "matched": int(matched.sum()),
        "valid": int(valid.sum()),
    }


def probe_representation(
    embeddings: np.ndarray,
    representation: str,
    encoder_kind: str,
    redshifts: np.ndarray,
    targetids: np.ndarray,
    is_test: np.ndarray,
    provabgs_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    train_index = np.flatnonzero(~is_test)
    test_index = np.flatnonzero(is_test)
    train_x, test_x, standardization = standardize_features(
        embeddings,
        train_index,
        test_index,
        device,
    )
    seeds = IMAGE_SEEDS if encoder_kind in IMAGE_ENCODERS else SPECTRUM_SEEDS
    print(f"{representation} redshift ridge")
    redshift = evaluate_ridge_seeds(
        train_x,
        redshifts[train_index],
        test_x,
        redshifts[test_index],
        seeds,
    )
    result: dict[str, Any] = {
        "representation": representation,
        "standardization": standardization,
        "geometry": embedding_geometry(train_x),
        "redshift": {
            **redshift,
            "num_train": int(len(train_index)),
            "num_test": int(len(test_index)),
            "published_reference": (
                {"AstroCLIP_image": 0.79}
                if encoder_kind in IMAGE_ENCODERS
                else {"AstroCLIP_spectrum": ASTROCLIP_SPECTRUM_R2["redshift"]}
            ),
        },
    }
    if encoder_kind in IMAGE_ENCODERS:
        return result

    targets, valid, match_stats = build_property_targets(
        targetids,
        read_provabgs_catalog(provabgs_path),
    )
    property_train_index = np.flatnonzero(valid & ~is_test)
    property_test_index = np.flatnonzero(valid & is_test)
    property_train_x, property_test_x, property_standardization = standardize_features(
        embeddings,
        property_train_index,
        property_test_index,
        device,
    )
    properties: dict[str, Any] = {}
    for column, property_name in enumerate(PROPERTIES):
        train_y = targets[property_train_index, column]
        test_y = targets[property_test_index, column]
        target_mean = float(train_y.mean())
        target_std = float(train_y.std())
        train_y_scaled = (train_y - target_mean) / target_std
        test_y_scaled = (test_y - target_mean) / target_std
        print(f"{representation} {property_name} ridge")
        evaluation = evaluate_ridge_seeds(
            property_train_x,
            train_y_scaled,
            property_test_x,
            test_y_scaled,
            SPECTRUM_SEEDS,
            physical_target=True,
        )
        for repeat in evaluation["repeats"]:
            for split in ("validation", "test"):
                repeat[split]["mae"] *= target_std
                repeat[split]["rmse"] *= target_std
        for split in ("validation", "test"):
            for metric_name in ("mae", "rmse"):
                values = [
                    repeat[split][metric_name] for repeat in evaluation["repeats"]
                ]
                evaluation["summary"][split][metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "values": values,
                }
        properties[property_name] = {
            **evaluation,
            "target_mean": target_mean,
            "target_std": target_std,
            "published_reference": {
                "AstroCLIP_spectrum": ASTROCLIP_SPECTRUM_R2[property_name]
            },
        }
    result["properties"] = {
        "match_stats": {
            **match_stats,
            "num_train": int(len(property_train_index)),
            "num_test": int(len(property_test_index)),
        },
        "standardization": property_standardization,
        "definitions": {
            "stellar_mass": "LOG_MSTAR",
            "metallicity": "log10(Z_MW)",
            "age": "TAGE_MW",
            "ssfr": "log10(AVG_SFR) - LOG_MSTAR",
        },
        "values": properties,
    }
    return result


def probe(args: argparse.Namespace, checkpoint: Path, label: str) -> dict[str, Any]:
    paths = cache_paths(args.cache_dir, label)
    if not paths["metadata"].exists():
        raise FileNotFoundError(f"No extraction metadata at {paths['metadata']}")
    metadata = json.loads(paths["metadata"].read_text())
    redshifts, targetids, is_test = read_targets(args.data_dir)
    if len(redshifts) != int(metadata["total_rows"]):
        raise RuntimeError(
            f"Target count {len(redshifts)} differs from cache count "
            f"{metadata['total_rows']}"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")

    representations: dict[str, Any] = {}
    for representation in ("raw", "projected"):
        embeddings = np.load(paths[representation], mmap_mode="r")
        representations[representation] = probe_representation(
            embeddings,
            representation,
            args.encoder,
            redshifts,
            targetids,
            is_test,
            args.provabgs_path,
            device,
        )
        del embeddings
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result = {
        "label": label,
        "encoder": args.encoder,
        "checkpoint": str(checkpoint.resolve()),
        "sample": "AstroCLIP DESI-LS x DESI EDR cross-match",
        "split": "AstroCLIP shipped train/test split",
        "ridge_l2_grid": RIDGE_L2_GRID,
        "image_seeds": IMAGE_SEEDS,
        "spectrum_seeds": SPECTRUM_SEEDS,
        "embedding_metadata": metadata,
        "representations": representations,
        "completed_at_unix": time.time(),
    }
    output_path = args.output_dir / label / "metrics.json"
    atomic_json(output_path, result)
    print(f"Wrote {output_path}")
    return result


def main() -> None:
    args = parse_args()
    checkpoint = (args.checkpoint or DEFAULT_CHECKPOINTS[args.encoder]).resolve()
    label = args.label or DEFAULT_LABELS[args.encoder]
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    if args.stage in ("extract", "all"):
        extract(args, checkpoint, label)
    if args.stage in ("probe", "all"):
        probe(args, checkpoint, label)


if __name__ == "__main__":
    main()

