#!/usr/bin/env python3
"""Probe every pre-alignment image checkpoint and plot redshift R2 by step."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import VisionTransformer
from tqdm.auto import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SOURCE_DIR = (
    REPO_ROOT / "checkpoints/VitLargePatch14_OfficialTrain5_Epoch5_2504"
)
CPT_DIR = REPO_ROOT / "checkpoints/ViTL14_LeJEPA_CPT10_LocalMMU_FromStep52000"
CPT_SOURCE_STEP = 52_000
ASTRODINO_CHECKPOINT = Path(
    "/mnt/datasets/utbd_pranav/models/astroclip/astrodino/astrodino.ckpt"
)
OUTPUT_DIR = SCRIPT_DIR / "results/image_pre_alignment_checkpoint_series"
ASTRODINO_STEP = 250_000
PLOT_SEED = 42


def load_benchmark_module():
    path = SCRIPT_DIR / "evaluate_frozen_backbones.py"
    spec = importlib.util.spec_from_file_location("image_series_benchmark", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = load_benchmark_module()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--worker-kind", choices=("ours", "ours_cpt", "astrodino")
    )
    parser.add_argument("--worker-checkpoint", type=Path)
    parser.add_argument("--worker-step", type=int)
    parser.add_argument("--worker-label")
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def task_list() -> list[dict[str, Any]]:
    tasks = []
    for checkpoint in sorted(
        SOURCE_DIR.glob("step_*.pt"),
        key=lambda path: int(path.stem.removeprefix("step_")),
    ):
        step = int(checkpoint.stem.removeprefix("step_"))
        tasks.append(
            {
                "kind": "ours",
                "checkpoint": checkpoint,
                "step": step,
                "label": f"astrojepa_step_{step:06d}",
            }
        )
    tasks.append(
        {
            "kind": "ours",
            "checkpoint": SOURCE_DIR / "complete.pt",
            "step": 54_940,
            "label": "astrojepa_complete_054940",
        }
    )
    first_cpt_epoch = CPT_DIR / "epoch_01.pt"
    if first_cpt_epoch.exists():
        checkpoint = torch.load(
            first_cpt_epoch,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        cpt_steps = int(checkpoint["global_step"])
        del checkpoint
        tasks.append(
            {
                "kind": "ours_cpt",
                "checkpoint": first_cpt_epoch,
                "step": CPT_SOURCE_STEP + cpt_steps,
                "label": "astrojepa_cpt_epoch_01",
            }
        )
    tasks.append(
        {
            "kind": "astrodino",
            "checkpoint": ASTRODINO_CHECKPOINT,
            "step": ASTRODINO_STEP,
            "label": "astroclip_astrodino_pre_alignment",
        }
    )
    return tasks


def load_ours(path: Path, device: torch.device):
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=True, mmap=True
    )
    config = dict(checkpoint["cfg"])
    model = timm.create_model(
        str(config["model_name"]),
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True,
        dynamic_img_pad=True,
    )
    state = {
        key.removeprefix("backbone."): value
        for key, value in checkpoint["model"].items()
        if key.startswith("backbone.")
    }
    model.load_state_dict(state, strict=True)
    metadata = {
        "format": "astrojepa_image_pre_alignment",
        "source_global_step": int(checkpoint["global_step"]),
        "source_epoch": int(checkpoint["epoch"]),
        "input_size": 140,
        "model_name": str(config["model_name"]),
    }
    del checkpoint, state
    return model.to(device).eval(), metadata


def load_astrodino(path: Path, device: torch.device):
    model = VisionTransformer(
        img_size=144,
        patch_size=12,
        in_chans=3,
        num_classes=0,
        global_pool="token",
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        init_values=1e-5,
    )
    teacher = torch.load(
        path, map_location="cpu", weights_only=False, mmap=True
    )["teacher"]
    state = {}
    for key, value in teacher.items():
        if not key.startswith("backbone.") or key == "backbone.mask_token":
            continue
        key = key.removeprefix("backbone.")
        parts = key.split(".")
        if parts[0] == "blocks":
            key = ".".join(["blocks", parts[2], *parts[3:]])
        state[key] = value
    model.load_state_dict(state, strict=True)
    del teacher, state
    return model.to(device).eval(), {
        "format": "official_astroclip_astrodino_teacher",
        "source_global_step": ASTRODINO_STEP,
        "input_size": 144,
        "model_name": "AstroDINO ViT-L/12",
        "source": "https://huggingface.co/polymathic-ai/astrodino",
    }


@torch.inference_mode()
def extract_embeddings(model, input_size, device, batch_size):
    train_files, test_files = benchmark.split_files(benchmark.DEFAULT_DATA_DIR)
    files = train_files + test_files
    total_rows = benchmark.shard_rows(files)
    embeddings = np.empty((total_rows, 1024), dtype=np.float32)
    offset = 0
    progress = tqdm(total=total_rows, desc=f"embeddings cuda:{device.index}")
    for image_array in benchmark.iter_image_batches(files, batch_size):
        rgb = benchmark.dr2_rgb_batch(image_array)
        batch = torch.from_numpy(rgb).permute(0, 3, 1, 2)
        batch = batch.to(device=device, dtype=torch.float32, non_blocking=True)
        if batch.shape[-1] != input_size:
            batch = F.interpolate(
                batch,
                size=(input_size, input_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        with benchmark.amp_context(device):
            encoded = model(batch)
        stop = offset + len(image_array)
        embeddings[offset:stop] = encoded.float().cpu().numpy()
        offset = stop
        progress.update(len(image_array))
    progress.close()
    if offset != total_rows:
        raise RuntimeError(f"Extracted {offset} rows, expected {total_rows}")
    return embeddings


def atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def worker(args):
    if not all(
        (
            args.worker_kind,
            args.worker_checkpoint,
            args.worker_step is not None,
            args.worker_label,
            args.device,
        )
    ):
        raise ValueError("Incomplete worker arguments")
    output = OUTPUT_DIR / f"{args.worker_label}.json"
    if output.exists() and not args.overwrite:
        print(f"Using {output}")
        return

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    if args.worker_kind in {"ours", "ours_cpt"}:
        model, metadata = load_ours(args.worker_checkpoint, device)
    else:
        model, metadata = load_astrodino(args.worker_checkpoint, device)

    embeddings = extract_embeddings(
        model, int(metadata["input_size"]), device, args.batch_size
    )
    del model
    torch.cuda.empty_cache()

    redshifts, _, is_test = benchmark.read_targets(benchmark.DEFAULT_DATA_DIR)
    train_index = np.flatnonzero(~is_test)
    test_index = np.flatnonzero(is_test)
    train_x, test_x, standardization = benchmark.standardize_features(
        embeddings, train_index, test_index, device
    )
    evaluation = benchmark.evaluate_ridge_seeds(
        train_x,
        redshifts[train_index],
        test_x,
        redshifts[test_index],
        benchmark.IMAGE_SEEDS,
    )
    result = {
        "label": args.worker_label,
        "kind": args.worker_kind,
        "step": int(args.worker_step),
        "checkpoint": str(args.worker_checkpoint.resolve()),
        "checkpoint_size_bytes": args.worker_checkpoint.stat().st_size,
        "metadata": metadata,
        "dataset": {
            "train_rows": int(len(train_index)),
            "test_rows": int(len(test_index)),
            "preprocessing": "official DR2 RGB; native backbone input size",
        },
        "standardization": standardization,
        "redshift": evaluation,
    }
    atomic_json(output, result)
    print(f"Wrote {output}")


def launch_task(task, device, batch_size, overwrite):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-kind",
        task["kind"],
        "--worker-checkpoint",
        str(task["checkpoint"]),
        "--worker-step",
        str(task["step"]),
        "--worker-label",
        task["label"],
        "--device",
        f"cuda:{device}",
        "--batch-size",
        str(batch_size),
    ]
    if overwrite:
        command.append("--overwrite")
    log_path = OUTPUT_DIR / f"{task['label']}.log"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    return task["label"]


def run_device_lane(tasks, device, batch_size, overwrite):
    completed = []
    for task in tasks:
        completed.append(
            launch_task(task, device, batch_size, overwrite)
        )
    return completed


def aggregate_and_plot(tasks):
    records = []
    for task in tasks:
        metrics = json.loads((OUTPUT_DIR / f"{task['label']}.json").read_text())
        repeat = next(
            item
            for item in metrics["redshift"]["repeats"]
            if item["seed"] == PLOT_SEED
        )
        records.append(
            {
                "label": task["label"],
                "kind": task["kind"],
                "step": task["step"],
                "probe_seed": PLOT_SEED,
                "selected_l2": repeat["selected_l2"],
                "test_r2": repeat["test"]["r2"],
            }
        )
    records.sort(key=lambda row: row["step"])

    csv_path = OUTPUT_DIR / "image_redshift_by_step.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    atomic_json(OUTPUT_DIR / "image_redshift_by_step.json", records)

    ours = [record for record in records if record["kind"] == "ours"]
    cpt = [record for record in records if record["kind"] == "ours_cpt"]
    astro = next(record for record in records if record["kind"] == "astrodino")
    fig, axis = plt.subplots(figsize=(10, 5.8))
    axis.plot(
        [row["step"] for row in ours],
        [row["test_r2"] for row in ours],
        color="#176B87",
        marker="o",
        linewidth=2,
        markersize=5,
        label="AstroJEPA pre-alignment",
    )
    if cpt:
        source = next(row for row in ours if row["step"] == CPT_SOURCE_STEP)
        continuation = [source, *sorted(cpt, key=lambda row: row["step"])]
        axis.plot(
            [row["step"] for row in continuation],
            [row["test_r2"] for row in continuation],
            color="#2E8B57",
            marker="s",
            linewidth=2,
            linestyle="--",
            markersize=6,
            label="AstroJEPA local-data CPT",
        )
    axis.plot(
        [astro["step"]],
        [astro["test_r2"]],
        color="#C23B22",
        marker="D",
        markersize=8,
        linestyle="none",
        label="AstroCLIP AstroDINO pre-alignment",
    )
    axis.annotate(
        f"AstroDINO\nR2={astro['test_r2']:.3f}",
        (astro["step"], astro["test_r2"]),
        xytext=(-12, 18),
        textcoords="offset points",
        ha="right",
    )
    axis.set_xlabel("Unimodal pretraining optimizer step")
    axis.set_ylabel(f"Frozen image redshift ridge test R2 (seed {PLOT_SEED})")
    axis.set_title("Pre-alignment image redshift representation by checkpoint")
    axis.grid(True, alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "image_redshift_step_vs_r2.png", dpi=180)
    fig.savefig(OUTPUT_DIR / "image_redshift_step_vs_r2.pdf")
    plt.close(fig)


def coordinator(args):
    devices = [int(value) for value in args.devices.split(",")]
    tasks = task_list()
    pending = [
        task
        for task in tasks
        if args.overwrite or not (OUTPUT_DIR / f"{task['label']}.json").exists()
    ]
    lanes = [pending[index::len(devices)] for index in range(len(devices))]
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = {
            executor.submit(
                run_device_lane,
                lane,
                devices[index],
                args.batch_size,
                args.overwrite,
            ): devices[index]
            for index, lane in enumerate(lanes)
            if lane
        }
        for future in as_completed(futures):
            labels = ", ".join(future.result())
            print(
                f"Completed cuda:{futures[future]} lane: {labels}",
                flush=True,
            )
    aggregate_and_plot(tasks)
    print(f"Wrote checkpoint-series results under {OUTPUT_DIR}")


def main():
    args = parse_args()
    if args.worker_kind:
        worker(args)
    else:
        coordinator(args)


if __name__ == "__main__":
    main()
