#!/usr/bin/env python3
"""Benchmark Smith42/galaxies using astroPT's native HF streaming pattern."""

import argparse
import gc
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader


def materialize_image(sample):
    """Force the same PIL-to-NumPy decode used at the start of astroPT's map."""
    image = np.array(sample["image"], copy=True)
    return {"pixels": np.int64(image.size)}


def distributed_context():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        dist.init_process_group("gloo")
    return world_size, rank


def build_dataset(args, world_size, rank):
    dataset = load_dataset(
        args.dataset,
        revision=args.revision,
        split=args.split,
        streaming=True,
    )
    dataset = dataset.select_columns("image")
    dataset = dataset.map(materialize_image).remove_columns("image")

    # This is astroPT's ordering: rank split first, buffered shuffle second.
    if world_size > 1:
        dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    if args.shuffle:
        dataset = dataset.shuffle(seed=1337, buffer_size=args.buffer_size)
    return dataset


def reduce_result(count, elapsed, world_size):
    count_tensor = torch.tensor(count, dtype=torch.int64)
    elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    return int(count_tensor.item()), float(elapsed_tensor.item())


def next_batch_on_all_ranks(iterator, world_size):
    try:
        batch = next(iterator)
        available = 1
    except StopIteration:
        batch = None
        available = 0

    if world_size > 1:
        available_tensor = torch.tensor(available, dtype=torch.int32)
        dist.all_reduce(available_tensor, op=dist.ReduceOp.MIN)
        available = int(available_tensor.item())
    return batch if available else None


def report_progress(args, count, started, world_size, rank):
    elapsed = time.perf_counter() - started
    global_count, wall_time = reduce_result(count, elapsed, world_size)
    if rank != 0:
        return
    rate = global_count / wall_time
    remaining = max(0, args.expected_samples - global_count)
    eta = remaining / rate if rate > 0 else float("inf")
    print(
        f"  global samples={global_count:,}/{args.expected_samples:,} "
        f"elapsed={wall_time:,.1f}s rate={rate:,.1f} samples/s "
        f"ETA={eta / 3600:.2f}h total~={(wall_time + eta) / 3600:.2f}h",
        flush=True,
    )


def run_once(args, workers, world_size, rank):
    dataset = build_dataset(args, world_size, rank)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=workers,
        pin_memory=True,
    )
    local_limit = None
    if args.limit is not None:
        local_limit = (args.limit + world_size - 1) // world_size

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        print(f"starting workers/rank={workers}; waiting for first batch...", flush=True)

    started = time.perf_counter()
    first_batch_time = None
    count = 0
    batch_index = 0
    iterator = iter(loader)
    while True:
        batch = next_batch_on_all_ranks(iterator, world_size)
        if batch is None:
            break
        batch_index += 1

        if first_batch_time is None:
            first_batch_time = time.perf_counter() - started
            if rank == 0:
                print(f"  first batch after {first_batch_time:,.2f}s", flush=True)

        count += int(batch["pixels"].numel())
        if world_size > 1 and args.sync_every_batch:
            dist.barrier()
        if args.log_every_batches > 0 and batch_index % args.log_every_batches == 0:
            report_progress(args, count, started, world_size, rank)
        if local_limit is not None and count >= local_limit:
            break

    elapsed = time.perf_counter() - started
    total_count, wall_time = reduce_result(count, elapsed, world_size)
    if rank == 0:
        print(
            f"workers/rank={workers:<2} ranks={world_size:<2} "
            f"samples={total_count:,} seconds={wall_time:,.2f} "
            f"samples/s={total_count / wall_time:,.2f} "
            f"hours={wall_time / 3600:.3f}",
            flush=True,
        )

    del iterator, loader, dataset
    gc.collect()
    if world_size > 1:
        dist.barrier()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Smith42/galaxies")
    parser.add_argument("--revision", default="v2.0")
    parser.add_argument("--split", default="train")
    parser.add_argument("--workers", type=int, nargs="+", default=[32])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--buffer-size", type=int, default=1_000)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, help="Approximate global sample limit")
    parser.add_argument("--expected-samples", type=int, default=8_474_566)
    parser.add_argument("--log-every-batches", type=int, default=100)
    parser.add_argument(
        "--sync-every-batch", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main():
    args = parse_args()
    world_size, rank = distributed_context()
    if rank == 0:
        print(
            f"dataset={args.dataset}@{args.revision} split={args.split} "
            f"shuffle={args.shuffle} buffer={args.buffer_size:,} "
            f"batch_size={args.batch_size} sync_every_batch={args.sync_every_batch}",
            flush=True,
        )
    for workers in args.workers:
        run_once(args, workers, world_size, rank)
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
