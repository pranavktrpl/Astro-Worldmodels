import os
import time
from pathlib import Path
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from configs.config import TrainConfig
from data.dataloaders import MyDataset


def setup_ddp():
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=60),
    )
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return world_size, rank, local_rank, device


def is_main_process(rank):
    return rank == 0


def build_debug_loader(cfg, world_size, rank, drop_last):
    ds = MyDataset(
        split="train",
        dataset=cfg.dataset_name,
        columns=cfg.columns,
        shuffle=True,
        world_size=world_size,
        rank=rank,
        Vg=cfg.Vg,
        Vl=cfg.Vl,
    )

    loader = DataLoader(
        ds,
        batch_size=cfg.bs,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        timeout=600 if cfg.num_workers > 0 else 0,
        drop_last=drop_last,
    )

    return ds, loader


def log_line(path, text):
    with open(path, "a") as f:
        f.write(text + "\n")
        f.flush()


def main():
    world_size, rank, local_rank, device = setup_ddp()

    cfg = TrainConfig()

    # Test both later if needed. Start with whatever your training currently uses.
    drop_last = False

    debug_dir = Path(cfg.save_dir) / "dataloader_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    rank_log = debug_dir / f"rank_{rank}.txt"
    if rank_log.exists():
        rank_log.unlink()

    log_line(
        rank_log,
        f"START rank={rank} local_rank={local_rank} world_size={world_size} "
        f"bs={cfg.bs} num_workers={cfg.num_workers} drop_last={drop_last} "
        f"time={time.strftime('%Y-%m-%d %H:%M:%S')}",
    )

    train_ds, train_loader = build_debug_loader(cfg, world_size, rank, drop_last=drop_last)
    train_ds.set_epoch(0)

    iterator = iter(train_loader)

    step = 0
    pbar = tqdm(disable=not is_main_process(rank), desc="debug dataloader epoch")

    while True:
        step += 1

        try:
            batch = next(iterator)
            has_batch = torch.tensor([1], device=device, dtype=torch.int64)

            global_crops = batch["global_crops"]
            local_crops = batch.get("local_crops", None)

            local_bs = global_crops.shape[0]
            global_shape = tuple(global_crops.shape)
            local_shape = None if local_crops is None else tuple(local_crops.shape)

        except StopIteration:
            has_batch = torch.tensor([0], device=device, dtype=torch.int64)
            local_bs = -1
            global_shape = None
            local_shape = None

        # Gather whether each rank still has a batch.
        all_has = [torch.zeros_like(has_batch) for _ in range(world_size)]
        dist.all_gather(all_has, has_batch)
        all_has_vals = [x.item() for x in all_has]

        # Gather batch sizes.
        bs_tensor = torch.tensor([local_bs], device=device, dtype=torch.int64)
        all_bs = [torch.zeros_like(bs_tensor) for _ in range(world_size)]
        dist.all_gather(all_bs, bs_tensor)
        all_bs_vals = [x.item() for x in all_bs]

        if step==21977 or step==21976 or step==21978 or step % 100 == 0 or len(set(all_bs_vals)) != 1 or len(set(all_has_vals)) != 1:
            log_line(
                rank_log,
                f"step={step} rank={rank} has_batch={has_batch.item()} "
                f"all_has={all_has_vals} local_bs={local_bs} all_bs={all_bs_vals} "
                f"global_shape={global_shape} local_shape={local_shape} "
                f"time={time.strftime('%Y-%m-%d %H:%M:%S')}",
            )

        if is_main_process(rank):
            pbar.update(1)
            if step % 100 == 0:
                pbar.set_postfix(all_bs=str(all_bs_vals), all_has=str(all_has_vals))

        # If any rank has ended, stop all ranks together.
        if min(all_has_vals) == 0:
            if is_main_process(rank):
                print(f"\nIterator ended at global debug step={step}")
                print(f"all_has={all_has_vals}")
                print(f"all_bs={all_bs_vals}")
            log_line(
                rank_log,
                f"END step={step} rank={rank} all_has={all_has_vals} all_bs={all_bs_vals}",
            )
            break

        # Hard safety cap: do not run forever.
        if step >= getattr(cfg, "total_steps", 1000000):
            if is_main_process(rank):
                print(f"\nReached cfg.total_steps={cfg.total_steps}; stopping debug.")
            log_line(rank_log, f"STOPPED_AT_TOTAL_STEPS step={step}")
            break

    pbar.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()