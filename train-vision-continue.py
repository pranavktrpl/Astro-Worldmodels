"""Continued pretraining (domain adaptation) of an existing vision checkpoint
on the AstroCLIP cross-match train split.

Reuses train-vision.py wholesale (model, LeJePA loss, AMP, checkpointing) via
importlib; differences from a from-scratch run:

  - model_name/proj_dim are read from the init_from checkpoint's saved cfg,
    so the architecture always matches the weights;
  - only model weights are loaded — optimizer, scheduler, and step counter
    start fresh on a short low-lr cosine schedule (see
    configs/config_continue_astroclip.py);
  - data comes from the local cross-match parquet shards (train split only)
    through the same AstroMultiCropTransform augmentations.

Launch (edit configs/config_continue_astroclip.py first):

    torchrun --standalone --nproc_per_node=4 train-vision-continue.py

Afterwards, select a checkpoint with Evals/checkpoint_loss_curves, then
re-run Evals/desi_crossmatch/astroclip_redshift_probe.py against it to
measure how much of the domain gap closed.
"""

import importlib.util
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tv = load_module("train_vision", REPO_ROOT / "train-vision.py")

from configs.config_continue_astroclip import ContinueAstroclipConfig  # noqa: E402
from data.astroclip_crossmatch_source import (  # noqa: E402
    AstroclipCrossmatchImages,
    count_train_rows,
)


def main() -> None:
    world_size, rank, local_rank, device = tv.setup_ddp()
    tv.seed_everything(seed=42, rank=rank)

    cfg = ContinueAstroclipConfig()

    checkpoint = torch.load(cfg.init_from, map_location="cpu", weights_only=False)
    base_cfg = dict(checkpoint.get("cfg") or {})
    cfg.model_name = base_cfg["model_name"]
    cfg.proj_dim = int(base_cfg.get("proj_dim", 64))

    rows = count_train_rows(cfg.data_dir)
    cfg.steps_per_epoch = max(1, int(rows * 0.9) // (cfg.bs * world_size))
    cfg.total_steps = cfg.epochs * cfg.steps_per_epoch

    if tv.is_main_process(rank):
        print(f"continuing from {cfg.init_from} ({cfg.model_name})")
        print(
            f"{rows} train images, steps_per_epoch={cfg.steps_per_epoch}, "
            f"total_steps={cfg.total_steps}"
        )

    train_ds = AstroclipCrossmatchImages(
        cfg.data_dir,
        world_size=world_size,
        rank=rank,
        shuffle=True,
        Vg=cfg.Vg,
        Vl=cfg.Vl,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.bs,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
        timeout=600 if cfg.num_workers > 0 else 0,
        drop_last=True,
    )

    model = tv.build_model(cfg, device, local_rank)
    raw_model = model.module
    raw_model.load_state_dict(checkpoint["model"])
    if tv.is_main_process(rank):
        total, _ = tv.count_parameters(raw_model)
        print(f"loaded weights: {total:,} params")
    del checkpoint

    sigreg_fn = tv.build_sigreg(cfg, device)
    opt, scheduler = tv.build_optimizer_and_scheduler(cfg, model)
    amp_dtype, scaler = tv.build_amp(cfg)

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    wandb_run = tv.setup_wandb(cfg, rank)

    global_step = 0
    for epoch in range(cfg.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        pbar = tqdm(
            train_loader,
            disable=not tv.is_main_process(rank),
            desc=f"epoch {epoch}",
        )
        for step, batch in enumerate(pbar):
            global_crops = batch["global_crops"].to(device, non_blocking=True)
            local_crops = batch["local_crops"]
            if local_crops is not None:
                local_crops = local_crops.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=amp_dtype):
                emb, proj = model(global_crops, local_crops)
                loss, sim, sigreg = tv.compute_lejepa_loss(
                    proj=proj,
                    sigreg_fn=sigreg_fn,
                    lambd=cfg.lambd,
                    num_global_views=cfg.Vg,
                )
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            scheduler.step()
            global_step += 1
            lr = scheduler.get_last_lr()[0]

            if global_step % cfg.log_every == 0:
                loss_mean = tv.ddp_mean(loss.detach(), world_size)
                sim_mean = tv.ddp_mean(sim.detach(), world_size)
                sigreg_mean = tv.ddp_mean(sigreg.detach(), world_size)
                if tv.is_main_process(rank):
                    pbar.set_postfix(
                        loss=f"{loss_mean.item():.4f}",
                        sim=f"{sim_mean.item():.4f}",
                        sigreg=f"{sigreg_mean.item():.4f}",
                        lr=f"{lr:.2e}",
                    )
                    if wandb_run is not None:
                        tv.wandb.log(
                            {
                                "train/loss": loss_mean.item(),
                                "train/sim": sim_mean.item(),
                                "train/sigreg": sigreg_mean.item(),
                                "train/lr": lr,
                                "train/epoch": epoch,
                                "train/step": global_step,
                            },
                            step=global_step,
                        )

            if global_step % cfg.ckpt_every == 0:
                dist.barrier()
                if tv.is_main_process(rank):
                    tv.save_checkpoint(
                        cfg=cfg,
                        model=model,
                        optimizer=opt,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        global_step=global_step,
                        path=os.path.join(cfg.save_dir, f"step_{global_step}.pt"),
                    )
                dist.barrier()

            if (step + 1) >= cfg.steps_per_epoch:
                break

        dist.barrier()

    if tv.is_main_process(rank):
        tv.save_checkpoint(
            cfg=cfg,
            model=model,
            optimizer=opt,
            scheduler=scheduler,
            scaler=scaler,
            epoch=cfg.epochs,
            global_step=global_step,
            path=os.path.join(cfg.save_dir, "complete.pt"),
        )
    dist.barrier()

    if wandb_run is not None:
        tv.wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
