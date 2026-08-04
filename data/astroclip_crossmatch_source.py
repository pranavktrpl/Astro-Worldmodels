"""Local parquet image source for continued pretraining on the AstroCLIP
DESI-LS x DESI EDR cross-match (downloaded by
Evals/desi_crossmatch/download_astroclip_desi.sh).

Reads the train-split shards, converts raw grz fluxes to RGB with the Legacy
Survey dr2-style mapping (identical to the eval probe's preprocessing, so the
backbone sees the same domain at continued-pretraining and eval time), and
feeds PIL images through the standard AstroMultiCropTransform.

Only the train shards are ever read here — the test split stays untouched for
evaluation.
"""

import numpy as np
import pyarrow.parquet as pq
import torch
from pathlib import Path
from PIL import Image

# dr2-style RGB constants; canonical copy in
# Evals/desi_crossmatch/astroclip_redshift_probe.py (verified against
# AstroCLIP's ToRGB / legacypipe).
DR2_RGB_SCALES = {"g": (2, 6.0), "r": (1, 3.4), "z": (0, 2.2)}
DR2_RGB_M = 0.03
DR2_RGB_Q = 20.0
BANDS = ("g", "r", "z")


def dr2_rgb_batch(batch: np.ndarray) -> np.ndarray:
    """(B,H,W,3) grz fluxes -> (B,H,W,3) RGB in [0,1]."""
    intensity = np.zeros(batch.shape[:3], dtype=np.float64)
    for i, band in enumerate(BANDS):
        _, scale = DR2_RGB_SCALES[band]
        intensity += np.maximum(0.0, batch[..., i].astype(np.float64) * scale + DR2_RGB_M)
    intensity /= len(BANDS)
    stretch = np.arcsinh(DR2_RGB_Q * intensity) / np.sqrt(DR2_RGB_Q)
    intensity = np.where(intensity == 0.0, 1e-6, intensity)
    rgb = np.zeros(batch.shape, dtype=np.float32)
    for i, band in enumerate(BANDS):
        plane, scale = DR2_RGB_SCALES[band]
        rgb[..., plane] = (
            (batch[..., i].astype(np.float64) * scale + DR2_RGB_M) * stretch / intensity
        )
    return np.clip(rgb, 0.0, 1.0)


def train_shards(data_dir: Path) -> list[Path]:
    files = sorted(Path(data_dir).glob("data/train-*.parquet"))
    if not files:
        raise RuntimeError(
            f"No train shards under {data_dir}/data — run "
            "Evals/desi_crossmatch/download_astroclip_desi.sh first."
        )
    return files


def count_train_rows(data_dir: Path) -> int:
    """Row count from parquet metadata only — no data read."""
    return sum(pq.ParquetFile(f).metadata.num_rows for f in train_shards(data_dir))


class AstroclipCrossmatchImages(torch.utils.data.IterableDataset):
    """Streams multi-crop views from the cross-match train shards.

    Sharding matches MyDataset's convention: shards are split across
    world_size * num_workers consumers; shard files are shuffled per epoch
    with a deterministic seed.
    """

    def __init__(
        self,
        data_dir,
        world_size: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        Vg: int = 2,
        Vl: int = 8,
        read_batch: int = 64,
    ):
        self.files = train_shards(Path(data_dir))
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.read_batch = read_batch
        self.epoch = 0
        from .AstroTransforms import AstroMultiCropTransform

        self.transformer = AstroMultiCropTransform(Vl, Vg)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        num_shards = num_workers * self.world_size
        shard_id = self.rank * num_workers + worker_id

        files = list(self.files)
        if self.shuffle:
            rng = np.random.RandomState(42 + self.epoch)
            rng.shuffle(files)
        files = files[shard_id::num_shards]

        for path in files:
            parquet_file = pq.ParquetFile(path)
            for batch in parquet_file.iter_batches(
                batch_size=self.read_batch, columns=["image"]
            ):
                column = batch.column("image")
                flat = column.flatten().flatten().flatten().to_numpy(
                    zero_copy_only=False
                )
                count = len(column)
                side = int(round((flat.size / (count * 3)) ** 0.5))
                fluxes = flat.reshape(count, side, side, 3).astype(np.float32)
                rgb = (dr2_rgb_batch(fluxes) * 255.0).astype(np.uint8)
                order = np.arange(count)
                if self.shuffle:
                    np.random.shuffle(order)
                for index in order:
                    yield self.transformer.augment_image(
                        Image.fromarray(rgb[index])
                    )
