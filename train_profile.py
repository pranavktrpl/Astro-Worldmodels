import os
import random
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from data.dataloaders import MyDataset
from models.resnet9 import Resnet9, MLP
from configs.train_resnet9_profile import TrainConfig


#########################################################
# ConvNet specific fast paths
#########################################################
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


#########################################################
# DDP Setup + Dataloading
#########################################################
def setup_ddp():
    dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    return world_size, rank, local_rank, device


def seed_everything(seed: int, rank: int):
    seed = seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main_process(rank: int):
    return rank == 0


def build_train_loader(cfg, world_size, rank):
    train_ds = MyDataset(
        split="train",
        dataset=cfg.dataset_name,
        columns=cfg.columns,
        shuffle=True,
        world_size=world_size,
        rank=rank,
        Vg=cfg.Vg,
        Vl=cfg.Vl,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.bs,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    return train_ds, train_loader


def build_val_loader(cfg, world_size, rank):
    val_ds = MyDataset(
        split="validation",
        dataset=cfg.dataset_name,
        columns=cfg.columns,
        shuffle=False,
        world_size=1,
        rank=0,
        Vg=cfg.Vg,
        Vl=cfg.Vl,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_bs,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    return val_ds, val_loader


#########################################################
# Model = Backbone + Projector
#########################################################
class Resnet9Encoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()

        self.backbone = Resnet9(num_classes=1, num_channels=3)
        self.proj = MLP(
            in_channels=1024,
            hidden_channels=[2048, 2048, proj_dim],
            norm_layer="batch_norm",
        )

    def _encode_views(self, x):
        # x: [B, V, C, H, W]
        B, V = x.shape[:2]
        flat = x.flatten(0, 1)                        # [B*V, C, H, W]
        flat_emb = self.backbone(flat)                # [B*V, 1024]
        flat_proj = self.proj(flat_emb)               # [B*V, P]

        emb = flat_emb.reshape(B, V, -1).transpose(0, 1)    # [V, B, 1024]
        proj = flat_proj.reshape(B, V, -1).transpose(0, 1)  # [V, B, P]
        return emb, proj

    def forward(self, global_x, local_x=None):
        global_emb, global_proj = self._encode_views(global_x)

        if local_x is None:
            return global_emb, global_proj

        local_emb, local_proj = self._encode_views(local_x)

        emb = torch.cat([global_emb, local_emb], dim=0)     # globals first
        proj = torch.cat([global_proj, local_proj], dim=0)  # globals first
        return emb, proj


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def build_model(cfg, device, local_rank):
    model = Resnet9Encoder(proj_dim=cfg.proj_dim)
    model.apply(init_weights)
    model = model.to(device)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    return model


#########################################################
# Custom LEJEPA Loss + Training requirements
#########################################################
class SIGReg(nn.Module):
    def __init__(self, knots=17, num_slices=256):
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

    def forward(self, z):
        # z: [B, P]
        device = z.device
        P = z.size(-1)

        A = torch.randn(P, self.num_slices, device=device)
        A = A / A.norm(p=2, dim=0, keepdim=True)

        x_t = (z @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-2) - self.phi).square() + x_t.sin().mean(-2).square()

        statistic = (err @ self.weights) * z.size(0)
        return statistic.mean()


def compute_lejepa_loss(proj, sigreg_fn, lambd, num_global_views):
    # proj: [V, B, P]
    global_proj = proj[:num_global_views]             # [Vg, B, P]
    centers = global_proj.mean(0)                     # [B, P]

    sim = (centers.unsqueeze(0) - proj).square().mean()
    sigreg = torch.stack([sigreg_fn(proj[v]) for v in range(proj.size(0))]).mean()

    loss = (1 - lambd) * sim + lambd * sigreg
    return loss, sim, sigreg


def build_optimizer_and_scheduler(cfg, model):
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.wd,
    )

    warmup = LinearLR(
        opt,
        start_factor=0.01,
        total_iters=cfg.warmup_steps,
    )

    cosine = CosineAnnealingLR(
        opt,
        T_max=max(1, cfg.total_steps - cfg.warmup_steps),
        eta_min=cfg.min_lr,
    )

    scheduler = SequentialLR(
        opt,
        schedulers=[warmup, cosine],
        milestones=[cfg.warmup_steps],
    )

    return opt, scheduler


def build_amp(cfg):
    if cfg.amp_dtype == "bf16":
        amp_dtype = torch.bfloat16
        scaler = GradScaler("cuda", enabled=False)
    elif cfg.amp_dtype == "fp16":
        amp_dtype = torch.float16
        scaler = GradScaler("cuda", enabled=True)
    else:
        raise ValueError(f"Unknown amp_dtype: {cfg.amp_dtype}")

    return amp_dtype, scaler


#########################################################
# Checkpoint loading + saving
#########################################################
def save_checkpoint(cfg, model, optimizer, scheduler, scaler, epoch, global_step, path):
    ckpt = {
        "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "cfg": asdict(cfg),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }
    torch.save(ckpt, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device="cuda"):
    ckpt = torch.load(path, map_location=device)

    if isinstance(model, DDP):
        model.module.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt["model"])

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    if "python_rng" in ckpt:
        random.setstate(ckpt["python_rng"])
    if "torch_rng" in ckpt:
        torch.set_rng_state(ckpt["torch_rng"])
    if "cuda_rng" in ckpt:
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])

    start_epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)

    return start_epoch, global_step


def setup_wandb(cfg, rank):
    if not is_main_process(rank):
        return None

    wandb.init(
        entity=cfg.entity,
        project=cfg.project,
        name=cfg.run_name,
        config=asdict(cfg),
        mode="online",
    )
    return wandb


def ddp_mean(x, world_size):
    if not torch.is_tensor(x):
        x = torch.tensor(x, device="cuda")
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= world_size
    return y


def ddp_mean_max(value, device):
    t = torch.tensor([value], device=device, dtype=torch.float64)
    t_mean = t.clone()
    t_max = t.clone()
    dist.all_reduce(t_mean, op=dist.ReduceOp.SUM)
    t_mean /= dist.get_world_size()
    dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
    return t_mean.item(), t_max.item()


def sync_time(device):
    torch.cuda.synchronize(device)
    return time.perf_counter()


#########################################################
# Main training loop
#########################################################
def main():
    world_size, rank, local_rank, device = setup_ddp()
    seed_everything(seed=42, rank=rank)

    if is_main_process(rank):
        print(f"world_size = {world_size}, rank = {rank}, local_rank = {local_rank}")

    cfg = TrainConfig()

    train_ds, train_loader = build_train_loader(cfg, world_size, rank)

    if not cfg.profile_enabled:
        val_ds, val_loader = build_val_loader(cfg, world_size, rank)
    else:
        val_ds, val_loader = None, None

    model = build_model(cfg, device, local_rank)
    sigreg_fn = SIGReg().to(device)
    opt, scheduler = build_optimizer_and_scheduler(cfg, model)
    amp_dtype, scaler = build_amp(cfg)

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    if cfg.profile_enabled:
        wandb_run = None if cfg.profile_disable_wandb else setup_wandb(cfg, rank)
    else:
        wandb_run = setup_wandb(cfg, rank)

    start_epoch = 0
    global_step = 0
    profile_stats = defaultdict(list)
    profile_done = False

    if cfg.resume_path is not None:
        start_epoch, global_step = load_checkpoint(
            cfg.resume_path,
            model=model,
            optimizer=opt,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        if is_main_process(rank):
            print(f"Resumed from {cfg.resume_path} at epoch={start_epoch}, step={global_step}")

    for epoch in range(start_epoch, cfg.epochs):
        train_ds.set_epoch(epoch)
        model.train()

        pbar = tqdm(train_loader, disable=not is_main_process(rank), desc=f"epoch {epoch}")
        end_of_last_iter = time.perf_counter()

        for step, batch in enumerate(pbar):
            # ---------------- data wait ----------------
            data_wait_t = time.perf_counter() - end_of_last_iter

            # ---------------- H2D transfer ----------------
            t0 = sync_time(device)
            global_crops = batch["global_crops"].to(device, non_blocking=True)
            local_crops = batch["local_crops"]
            if local_crops is not None:
                local_crops = local_crops.to(device, non_blocking=True)
            t1 = sync_time(device)
            h2d_t = t1 - t0

            # ---------------- forward + loss ----------------
            opt.zero_grad(set_to_none=True)
            t0 = sync_time(device)
            with autocast("cuda", dtype=amp_dtype):
                emb, proj = model(global_crops, local_crops)
                loss, sim, sigreg = compute_lejepa_loss(
                    proj=proj,
                    sigreg_fn=sigreg_fn,
                    lambd=cfg.lambd,
                    num_global_views=cfg.Vg,
                )
            t1 = sync_time(device)
            forward_loss_t = t1 - t0

            # ---------------- backward ----------------
            t0 = sync_time(device)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            t1 = sync_time(device)
            backward_t = t1 - t0

            # ---------------- optimizer + scheduler ----------------
            t0 = sync_time(device)
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            scheduler.step()
            t1 = sync_time(device)
            optim_sched_t = t1 - t0

            global_step += 1

            # ---------------- metric sync / logging ----------------
            t0 = time.perf_counter()
            if (not cfg.profile_enabled) and (global_step % cfg.log_every == 0):
                loss_mean = ddp_mean(loss.detach(), world_size)
                sim_mean = ddp_mean(sim.detach(), world_size)
                sigreg_mean = ddp_mean(sigreg.detach(), world_size)

                if is_main_process(rank):
                    lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix(
                        loss=f"{loss_mean.item():.4f}",
                        sim=f"{sim_mean.item():.4f}",
                        sigreg=f"{sigreg_mean.item():.4f}",
                        lr=f"{lr:.2e}",
                    )

                    if wandb_run is not None:
                        wandb.log(
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
            metric_log_t = time.perf_counter() - t0

            # ---------------- record profile ----------------
            if cfg.profile_enabled:
                if cfg.profile_warmup_steps <= step < cfg.profile_warmup_steps + cfg.profile_steps:
                    profile_stats["data_wait"].append(data_wait_t)
                    profile_stats["h2d"].append(h2d_t)
                    profile_stats["forward_loss"].append(forward_loss_t)
                    profile_stats["backward"].append(backward_t)
                    profile_stats["optim_sched"].append(optim_sched_t)
                    profile_stats["metric_log"].append(metric_log_t)
                    profile_stats["step_total"].append(
                        data_wait_t + h2d_t + forward_loss_t + backward_t + optim_sched_t + metric_log_t
                    )

                if step >= cfg.profile_warmup_steps + cfg.profile_steps:
                    profile_done = True
                    break

            if (not cfg.profile_enabled) and is_main_process(rank) and (global_step % cfg.ckpt_every == 0):
                save_checkpoint(
                    cfg=cfg,
                    model=model,
                    optimizer=opt,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    global_step=global_step,
                    path=os.path.join(cfg.save_dir, f"step_{global_step}.pt"),
                )

            if global_step >= cfg.total_steps:
                break

            end_of_last_iter = time.perf_counter()

        dist.barrier()

        if cfg.profile_enabled and profile_done:
            break

        if (not cfg.profile_enabled) and is_main_process(rank):
            save_checkpoint(
                cfg=cfg,
                model=model,
                optimizer=opt,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                global_step=global_step,
                path=os.path.join(cfg.save_dir, "last.pt"),
            )

        if global_step >= cfg.total_steps:
            break

    if cfg.profile_enabled:
        local_summary = {}
        for k, vals in profile_stats.items():
            local_summary[k] = sum(vals) / max(1, len(vals))

        dist.barrier()

        summaries = {}
        for k, v in local_summary.items():
            mean_v, max_v = ddp_mean_max(v, device)
            summaries[k] = {"mean": mean_v, "max": max_v}

        if is_main_process(rank):
            print("\n=== PROFILE SUMMARY (seconds / step) ===")
            for k, stats in summaries.items():
                print(f"{k:15s} mean={stats['mean']:.4f}  max={stats['max']:.4f}")

            total = summaries["step_total"]["mean"]
            if total > 0:
                print("\n=== PROFILE BREAKDOWN (%) ===")
                for k in ["data_wait", "h2d", "forward_loss", "backward", "optim_sched", "metric_log"]:
                    pct = 100.0 * summaries[k]["mean"] / total
                    print(f"{k:15s} {pct:6.2f}%")

                global_samples_per_step = cfg.bs * world_size
                global_crops_per_step = global_samples_per_step * (cfg.Vg + cfg.Vl)
                print("\n=== THROUGHPUT ESTIMATES ===")
                print(f"global_samples_per_step: {global_samples_per_step}")
                print(f"global_crops_per_step:   {global_crops_per_step}")
                print(f"samples/sec:             {global_samples_per_step / total:.2f}")
                print(f"crops/sec:               {global_crops_per_step / total:.2f}")

    if is_main_process(rank) and wandb_run is not None:
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
