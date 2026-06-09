#!/usr/bin/env python3
"""Evaluate LeJEPA checkpoints on streamed HF splits and plot loss evolution."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import re
import sys
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import lejepa  # noqa: E402
import timm  # noqa: E402
from data.dataloaders import MyDataset  # noqa: E402
from models.resnet9 import MLP, Resnet9  # noqa: E402


STEP_RE = re.compile(r"^step_(\d+)(?:\.orig)?\.pt$")
SPLIT_COLORS = {
    "train": "#1f77b4",
    "validation": "#ff7f0e",
    "test": "#2ca02c",
}
DEFAULT_IGNORED_DIRECTORIES = {"FirstTrain_VitSmallPatch14_2104"}


@dataclass(frozen=True)
class Runtime:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class CheckpointRef:
    path: Path
    step: int


class TimmEncoder(nn.Module):
    """Architecture used by the ViT checkpoints."""

    def __init__(self, model_name: str, proj_dim: int):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )
        embed_dim = getattr(self.backbone, "num_features", None)
        if embed_dim is None:
            raise ValueError(f"Could not infer num_features for {model_name}")
        self.proj = MLP(
            in_channels=embed_dim,
            hidden_channels=[2 * embed_dim, 2 * embed_dim, proj_dim],
            norm_layer="batch_norm",
        )

    def _encode_views(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, views = x.shape[:2]
        flat_embeddings = self.backbone(x.flatten(0, 1))
        flat_projections = self.proj(flat_embeddings)
        embeddings = flat_embeddings.reshape(batch_size, views, -1).transpose(0, 1)
        projections = flat_projections.reshape(batch_size, views, -1).transpose(0, 1)
        return embeddings, projections

    def forward(
        self, global_x: torch.Tensor, local_x: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_embeddings, global_projections = self._encode_views(global_x)
        if local_x is None:
            return global_embeddings, global_projections
        local_embeddings, local_projections = self._encode_views(local_x)
        return (
            torch.cat([global_embeddings, local_embeddings], dim=0),
            torch.cat([global_projections, local_projections], dim=0),
        )


class Resnet9Encoder(nn.Module):
    """Architecture used by the legacy ResNet9 checkpoint."""

    def __init__(self, proj_dim: int):
        super().__init__()
        self.backbone = Resnet9(num_classes=1, num_channels=3)
        self.proj = MLP(
            in_channels=1024,
            hidden_channels=[2048, 2048, proj_dim],
            norm_layer="batch_norm",
        )

    def _encode_views(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, views = x.shape[:2]
        flat_embeddings = self.backbone(x.flatten(0, 1))
        flat_projections = self.proj(flat_embeddings)
        embeddings = flat_embeddings.reshape(batch_size, views, -1).transpose(0, 1)
        projections = flat_projections.reshape(batch_size, views, -1).transpose(0, 1)
        return embeddings, projections

    def forward(
        self, global_x: torch.Tensor, local_x: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_embeddings, global_projections = self._encode_views(global_x)
        if local_x is None:
            return global_embeddings, global_projections
        local_embeddings, local_projections = self._encode_views(local_x)
        return (
            torch.cat([global_embeddings, local_embeddings], dim=0),
            torch.cat([global_projections, local_projections], dim=0),
        )


class LegacySIGReg(nn.Module):
    """SIGReg implementation used by the pre-LeJEPA-library ResNet9 run."""

    def __init__(self, knots: int = 17, num_slices: int = 256):
        super().__init__()
        self.num_slices = num_slices
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        projections = torch.randn(
            z.size(-1), self.num_slices, device=z.device, dtype=torch.float32
        )
        projections = projections / projections.norm(p=2, dim=0, keepdim=True)
        x_t = (z.float() @ projections).unsqueeze(-1) * self.t
        error = (
            (x_t.cos().mean(-2) - self.phi).square()
            + x_t.sin().mean(-2).square()
        )
        statistic = (error @ self.weights) * z.size(0)
        return statistic.mean()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=REPO_ROOT / "checkpoints",
        help="Root containing one directory per model run.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        action="append",
        default=[],
        help="Evaluate only this checkpoint directory; may be repeated.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("validation", "test"),
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=64,
        help="Batches per rank and split; 0 consumes the full streaming split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fixed seed used to replay the same stochastic sample at every checkpoint.",
    )
    parser.add_argument(
        "--dataset-epoch",
        type=int,
        default=0,
        help="Epoch passed to the pre-existing streaming dataset.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Validation/test batch override. Defaults to checkpoint eval_bs or bs.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=None,
        help="Train batch override. Defaults to each checkpoint's saved bs.",
    )
    parser.add_argument(
        "--train-workers",
        type=int,
        default=None,
        help="Train worker override. Defaults to each checkpoint's saved config.",
    )
    parser.add_argument(
        "--train-shuffle",
        action="store_true",
        help=(
            "Enable the training loader's 50,000-example streaming shuffle buffer. "
            "Disabled by default to avoid refilling it for every checkpoint."
        ),
    )
    parser.add_argument(
        "--include-aliases",
        action="store_true",
        help="Evaluate duplicate checkpoint files that represent the same global step.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute matching cached records.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List discovered checkpoint files without loading models or data.",
    )
    return parser.parse_args()


def setup_runtime() -> Runtime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed evaluation requires CUDA/NCCL.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
    return Runtime(rank=rank, world_size=world_size, local_rank=local_rank, device=device)


def shutdown_runtime(runtime: Runtime) -> None:
    if runtime.world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def collate_crops(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor | None]:
    """Default-collate tensors while preserving the Vl=0 local-crop sentinel."""
    return {
        "global_crops": default_collate([sample["global_crops"] for sample in samples]),
        "local_crops": (
            None
            if samples[0]["local_crops"] is None
            else default_collate([sample["local_crops"] for sample in samples])
        ),
    }


def torch_load(path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "map_location": "cpu",
        "weights_only": False,
    }
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def checkpoint_directories(args: argparse.Namespace) -> list[Path]:
    if args.checkpoint_dir:
        directories = [path.resolve() for path in args.checkpoint_dir]
    else:
        root = args.checkpoint_root.resolve()
        directories = sorted(
            path
            for path in root.iterdir()
            if path.is_dir()
            and path.name not in DEFAULT_IGNORED_DIRECTORIES
            and any(path.glob("*.pt"))
        )
    return directories


def step_from_filename(path: Path) -> int | None:
    match = STEP_RE.match(path.name)
    return int(match.group(1)) if match else None


def checkpoint_priority(path: Path, step: int) -> tuple[int, str]:
    if path.name == f"step_{step}.pt":
        return (0, path.name)
    if path.name == "complete.pt":
        return (1, path.name)
    if path.name.startswith("last_epoch_"):
        return (2, path.name)
    if ".orig." in path.name:
        return (4, path.name)
    return (3, path.name)


def discover_checkpoints(
    directory: Path, include_aliases: bool
) -> tuple[list[CheckpointRef], list[dict[str, str]]]:
    refs: list[CheckpointRef] = []
    errors: list[dict[str, str]] = []
    for path in sorted(directory.glob("*.pt")):
        filename_step = step_from_filename(path)
        if filename_step is not None:
            refs.append(CheckpointRef(path=path, step=filename_step))
            continue
        try:
            checkpoint = torch_load(path)
            refs.append(
                CheckpointRef(path=path, step=int(checkpoint.get("global_step", -1)))
            )
            del checkpoint
        except Exception as exc:
            errors.append({"checkpoint": str(path), "error": repr(exc)})

    if not include_aliases:
        by_step: dict[int, CheckpointRef] = {}
        for ref in refs:
            current = by_step.get(ref.step)
            if current is None or checkpoint_priority(
                ref.path, ref.step
            ) < checkpoint_priority(current.path, current.step):
                by_step[ref.step] = ref
        refs = list(by_step.values())
    return sorted(refs, key=lambda ref: (ref.step, ref.path.name)), errors


def architecture_name(state_dict: dict[str, torch.Tensor], cfg: dict[str, Any]) -> str:
    if any(key.startswith("backbone.conv.") for key in state_dict):
        return "resnet9"
    if cfg.get("model_name"):
        return "timm"
    raise ValueError("Could not infer checkpoint architecture.")


def build_model(
    state_dict: dict[str, torch.Tensor], cfg: dict[str, Any], device: torch.device
) -> tuple[nn.Module, str]:
    architecture = architecture_name(state_dict, cfg)
    if architecture == "resnet9":
        model: nn.Module = Resnet9Encoder(proj_dim=int(cfg["proj_dim"]))
    else:
        model = TimmEncoder(
            model_name=str(cfg["model_name"]),
            proj_dim=int(cfg["proj_dim"]),
        )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, architecture


def build_sigreg(
    cfg: dict[str, Any], architecture: str, device: torch.device
) -> nn.Module:
    if architecture == "resnet9" and "sigreg_num_slices" not in cfg:
        return LegacySIGReg(knots=17, num_slices=256).to(device)
    univariate_test = lejepa.univariate.EppsPulley(
        n_points=int(cfg.get("sigreg_num_points", 17))
    )
    return lejepa.multivariate.SlicingUnivariateTest(
        univariate_test=univariate_test,
        num_slices=int(cfg.get("sigreg_num_slices", 1024)),
    ).to(device)


def compute_lejepa_loss(
    projections: torch.Tensor,
    sigreg_fn: nn.Module,
    lambd: float,
    num_global_views: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global_projections = projections[:num_global_views]
    centers = global_projections.mean(0)
    invariance = (centers.unsqueeze(0) - projections).square().mean()
    sigreg = torch.stack(
        [sigreg_fn(projections[view]) for view in range(projections.size(0))]
    ).mean()
    loss = (1 - lambd) * invariance + lambd * sigreg
    return loss, invariance, sigreg


def loader_settings(
    split: str, cfg: dict[str, Any], args: argparse.Namespace, runtime: Runtime
) -> tuple[int, int, bool, int, int]:
    if split == "train":
        batch_size = args.train_batch_size or int(cfg["bs"])
        workers = (
            args.train_workers
            if args.train_workers is not None
            else int(cfg.get("num_workers", 1))
        )
        return (
            batch_size,
            workers,
            bool(args.train_shuffle),
            runtime.world_size,
            runtime.rank,
        )
    if runtime.world_size != 1:
        raise RuntimeError(
            f"{split} must run with one GPU process. Launch it without torchrun."
        )
    batch_size = args.eval_batch_size or int(cfg.get("eval_bs", cfg["bs"]))
    return batch_size, 1, False, 1, 0


def build_loader(
    split: str,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    runtime: Runtime,
) -> tuple[DataLoader, int, int]:
    batch_size, workers, shuffle, data_world_size, data_rank = loader_settings(
        split, cfg, args, runtime
    )
    dataset = MyDataset(
        split=split,
        dataset=str(cfg.get("dataset_name", "Smith42/galaxies")),
        columns=list(cfg.get("columns", ["image_crop"])),
        shuffle=shuffle,
        world_size=data_world_size,
        rank=data_rank,
        Vg=int(cfg.get("Vg", 2)),
        Vl=int(cfg.get("Vl", 8)),
    )
    dataset.set_epoch(args.dataset_epoch)
    generator = torch.Generator()
    generator.manual_seed(args.seed + runtime.rank * 100_003)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=runtime.device.type == "cuda",
        persistent_workers=False,
        timeout=600 if workers > 0 else 0,
        drop_last=True,
        collate_fn=collate_crops,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return loader, batch_size, workers


def amp_context(device: torch.device, amp_dtype_name: str):
    if device.type != "cuda":
        return nullcontext()
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16
    return autocast("cuda", dtype=amp_dtype)


def reduce_statistics(
    values: torch.Tensor, runtime: Runtime
) -> tuple[float, float, int]:
    stats = torch.stack(
        [values.sum(), values.square().sum(), values.new_tensor(values.numel())]
    ).to(dtype=torch.float64)
    if runtime.world_size > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    count = int(stats[2].item())
    mean = float(stats[0].item() / count)
    variance = max(0.0, float(stats[1].item() / count) - mean * mean)
    standard_error = math.sqrt(variance / count) if count > 1 else 0.0
    return mean, standard_error, count


@torch.inference_mode()
def evaluate_split(
    model: nn.Module,
    sigreg_fn: nn.Module,
    cfg: dict[str, Any],
    split: str,
    args: argparse.Namespace,
    runtime: Runtime,
    description: str,
) -> dict[str, Any]:
    seed_everything(args.seed + runtime.rank * 100_003)
    loader, batch_size, workers = build_loader(split, cfg, args, runtime)
    metrics: dict[str, list[torch.Tensor]] = {
        "loss": [],
        "invariance": [],
        "sigreg": [],
    }
    batches = (
        itertools.islice(loader, args.max_batches)
        if args.max_batches
        else iter(loader)
    )
    progress = tqdm(
        batches,
        total=args.max_batches or None,
        desc=description,
        disable=not runtime.is_main,
        leave=False,
    )
    for batch in progress:
        global_crops = batch["global_crops"].to(runtime.device, non_blocking=True)
        local_crops = batch["local_crops"]
        if local_crops is not None:
            local_crops = local_crops.to(runtime.device, non_blocking=True)
        with amp_context(runtime.device, str(cfg.get("amp_dtype", "bf16"))):
            _, projections = model(global_crops, local_crops)
            loss, invariance, sigreg = compute_lejepa_loss(
                projections=projections,
                sigreg_fn=sigreg_fn,
                lambd=float(cfg.get("lambd", 0.05)),
                num_global_views=int(cfg.get("Vg", 2)),
            )
        metrics["loss"].append(loss.detach().float())
        metrics["invariance"].append(invariance.detach().float())
        metrics["sigreg"].append(sigreg.detach().float())

    if not metrics["loss"]:
        raise RuntimeError(f"No complete batches were produced for split={split}.")

    result: dict[str, Any] = {
        "split": split,
        "batch_size_per_rank": batch_size,
        "num_workers_per_rank": workers,
        "world_size": runtime.world_size,
        "max_batches_per_rank": args.max_batches,
        "dataset_epoch": args.dataset_epoch,
        "seed": args.seed,
        "drop_last": True,
        "shuffle": bool(split == "train" and args.train_shuffle),
    }
    for name, tensors in metrics.items():
        mean, standard_error, batch_count = reduce_statistics(
            torch.stack(tensors), runtime
        )
        result[name] = mean
        result[f"{name}_standard_error"] = standard_error
        result["evaluated_batches"] = batch_count
    result["evaluated_examples"] = batch_count * batch_size
    return result


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def atomic_json_dump(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def record_matches(
    record: dict[str, Any],
    ref: CheckpointRef,
    split: str,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    runtime: Runtime,
) -> bool:
    batch_size, workers, _, _, _ = loader_settings(split, cfg, args, runtime)
    return all(
        (
            record.get("checkpoint") == ref.path.name,
            record.get("split") == split,
            record.get("checkpoint_size") == ref.path.stat().st_size,
            record.get("max_batches_per_rank") == args.max_batches,
            record.get("seed") == args.seed,
            record.get("dataset_epoch") == args.dataset_epoch,
            record.get("batch_size_per_rank") == batch_size,
            record.get("num_workers_per_rank") == workers,
            record.get("world_size") == runtime.world_size,
            record.get("shuffle") == bool(split == "train" and args.train_shuffle),
        )
    )


def upsert_record(records: list[dict[str, Any]], new_record: dict[str, Any]) -> None:
    records[:] = [
        record
        for record in records
        if not (
            record.get("checkpoint") == new_record.get("checkpoint")
            and record.get("split") == new_record.get("split")
        )
    ]
    records.append(new_record)
    records.sort(key=lambda record: (int(record["global_step"]), record["split"]))


def write_csv(directory: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    fields = [
        "global_step",
        "checkpoint",
        "split",
        "loss",
        "loss_standard_error",
        "invariance",
        "invariance_standard_error",
        "sigreg",
        "sigreg_standard_error",
        "evaluated_batches",
        "evaluated_examples",
        "batch_size_per_rank",
        "num_workers_per_rank",
        "world_size",
        "max_batches_per_rank",
        "seed",
        "dataset_epoch",
        "shuffle",
        "architecture",
        "model_name",
    ]
    path = directory / "loss_evolution_metrics.csv"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def plot_metric(
    directory: Path,
    records: list[dict[str, Any]],
    metric: str,
    filename: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    plotted = False
    for split in ("train", "validation", "test"):
        split_records = sorted(
            (record for record in records if record["split"] == split),
            key=lambda record: int(record["global_step"]),
        )
        if not split_records:
            continue
        steps = np.asarray([record["global_step"] for record in split_records])
        values = np.asarray([record[metric] for record in split_records])
        errors = np.asarray(
            [record.get(f"{metric}_standard_error", 0.0) for record in split_records]
        )
        color = SPLIT_COLORS[split]
        ax.plot(steps, values, marker="o", linewidth=1.8, color=color, label=split)
        ax.fill_between(
            steps, values - errors, values + errors, color=color, alpha=0.16
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Model checkpoint (global training step)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{directory.name}: {ylabel} evolution")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / filename, dpi=180)
    fig.savefig(directory / Path(filename).with_suffix(".pdf"))
    plt.close(fig)


def plot_generalization_gap(
    directory: Path, records: list[dict[str, Any]]
) -> None:
    train_by_step = {
        int(record["global_step"]): record
        for record in records
        if record["split"] == "train"
    }
    validation_by_step = {
        int(record["global_step"]): record
        for record in records
        if record["split"] == "validation"
    }
    common_steps = sorted(set(train_by_step) & set(validation_by_step))
    if not common_steps:
        return
    gaps = [
        validation_by_step[step]["loss"] - train_by_step[step]["loss"]
        for step in common_steps
    ]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.plot(common_steps, gaps, marker="o", color="#9467bd", linewidth=1.8)
    ax.set_xlabel("Model checkpoint (global training step)")
    ax.set_ylabel("Validation loss - train loss")
    ax.set_title(f"{directory.name}: generalization gap")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(directory / "generalization_gap.png", dpi=180)
    fig.savefig(directory / "generalization_gap.pdf")
    plt.close(fig)


def write_best_checkpoint(
    directory: Path, records: list[dict[str, Any]]
) -> None:
    validation_records = [
        record for record in records if record["split"] == "validation"
    ]
    if not validation_records:
        return
    best = min(validation_records, key=lambda record: float(record["loss"]))
    step = int(best["global_step"])
    by_split = {
        record["split"]: record
        for record in records
        if int(record["global_step"]) == step
    }
    payload = {
        "selection_rule": "minimum streamed validation LeJEPA loss",
        "global_step": step,
        "checkpoint": best["checkpoint"],
        "validation_loss": best["loss"],
        "validation_loss_standard_error": best["loss_standard_error"],
        "train_loss": by_split.get("train", {}).get("loss"),
        "test_loss": by_split.get("test", {}).get("loss"),
        "generalization_gap": (
            best["loss"] - by_split["train"]["loss"]
            if "train" in by_split
            else None
        ),
        "sampling": {
            key: best[key]
            for key in (
                "evaluated_batches",
                "evaluated_examples",
                "batch_size_per_rank",
                "world_size",
                "max_batches_per_rank",
                "seed",
                "dataset_epoch",
            )
        },
    }
    atomic_json_dump(directory / "best_checkpoint.json", payload)


def write_outputs(directory: Path, records: list[dict[str, Any]]) -> None:
    atomic_json_dump(directory / "loss_evolution_metrics.json", records)
    write_csv(directory, records)
    plot_metric(directory, records, "loss", "loss_evolution.png", "LeJEPA loss")
    plot_metric(
        directory,
        records,
        "invariance",
        "invariance_evolution.png",
        "Invariance loss",
    )
    plot_metric(
        directory, records, "sigreg", "sigreg_evolution.png", "SIGReg loss"
    )
    plot_generalization_gap(directory, records)
    write_best_checkpoint(directory, records)


def append_error(
    errors: list[dict[str, str]], path: Path, split: str | None, exc: Exception
) -> None:
    errors.append(
        {
            "checkpoint": str(path),
            "split": split or "",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }
    )


def evaluate_directory(
    directory: Path, args: argparse.Namespace, runtime: Runtime
) -> None:
    refs, discovery_errors = discover_checkpoints(directory, args.include_aliases)
    if runtime.is_main:
        print(f"\n{directory}: {len(refs)} checkpoint candidate(s)")
        for ref in refs:
            print(f"  step={ref.step:>7}  {ref.path.name}")
    if args.list_only:
        return

    metrics_path = directory / "loss_evolution_metrics.json"
    records = load_json(metrics_path, [])
    if not isinstance(records, list):
        records = []
    errors: list[dict[str, str]] = list(discovery_errors)

    for ref in refs:
        checkpoint: dict[str, Any] | None = None
        model: nn.Module | None = None
        sigreg_fn: nn.Module | None = None
        try:
            checkpoint = torch_load(ref.path)
            cfg = dict(checkpoint.get("cfg") or {})
            actual_step = int(checkpoint.get("global_step", ref.step))
            ref = CheckpointRef(path=ref.path, step=actual_step)
            state_dict = checkpoint["model"]
            model, architecture = build_model(state_dict, cfg, runtime.device)
            sigreg_fn = build_sigreg(cfg, architecture, runtime.device)
            del state_dict
            del checkpoint
            checkpoint = None

            for split in args.splits:
                cached = next(
                    (
                        record
                        for record in records
                        if record_matches(record, ref, split, args, cfg, runtime)
                    ),
                    None,
                )
                if cached is not None and not args.overwrite:
                    if runtime.is_main:
                        print(f"  cached step={ref.step} split={split}")
                    continue
                description = f"{directory.name} step={ref.step} {split}"
                result = evaluate_split(
                    model=model,
                    sigreg_fn=sigreg_fn,
                    cfg=cfg,
                    split=split,
                    args=args,
                    runtime=runtime,
                    description=description,
                )
                result.update(
                    {
                        "checkpoint": ref.path.name,
                        "checkpoint_path": str(ref.path),
                        "checkpoint_size": ref.path.stat().st_size,
                        "global_step": ref.step,
                        "architecture": architecture,
                        "model_name": cfg.get("model_name", architecture),
                    }
                )
                if runtime.is_main:
                    upsert_record(records, result)
                    write_outputs(directory, records)
                    print(
                        f"  step={ref.step} split={split} "
                        f"loss={result['loss']:.6f} "
                        f"+/- {result['loss_standard_error']:.6f}"
                    )
                if runtime.world_size > 1:
                    dist.barrier()
        except Exception as exc:
            if runtime.is_main:
                append_error(errors, ref.path, None, exc)
                print(f"  ERROR {ref.path.name}: {exc}", file=sys.stderr)
        finally:
            del checkpoint, model, sigreg_fn
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()
            if runtime.world_size > 1:
                dist.barrier()

    if runtime.is_main:
        if errors:
            atomic_json_dump(directory / "loss_evaluation_errors.json", errors)
        elif (directory / "loss_evaluation_errors.json").exists():
            (directory / "loss_evaluation_errors.json").unlink()
        if records:
            write_outputs(directory, records)


def main() -> None:
    args = parse_args()
    runtime = setup_runtime()
    try:
        if runtime.world_size > 1 and any(
            split != "train" for split in args.splits
        ):
            raise RuntimeError(
                "Distributed mode is only allowed for train. Run validation/test "
                "without torchrun to enforce one GPU and one worker."
            )
        directories = checkpoint_directories(args)
        if not directories:
            raise FileNotFoundError("No checkpoint directories were discovered.")
        for directory in directories:
            evaluate_directory(directory, args, runtime)
    finally:
        shutdown_runtime(runtime)


if __name__ == "__main__":
    main()
