import os
import random
from dotenv import load_dotenv

load_dotenv()

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from data.dataloaders import MyDataset
from models.resnet9 import Resnet9, MLP

from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import wandb

######################################################### DDP Setup + Dataloading #########################################################
def setup_ddp():
    dist.init_process_group(backend="nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()   #Which process this is globally, so rank is an init in the dataset object we created, cause every GPU gets a custom dataset object
    local_rank = int(os.environ["LOCAL_RANK"])  #Just which GPU this is on the node, same as rank, no need honestly

    torch.cuda.set_device(local_rank)  #Binds the process to the GPU, for if we stop mid way, and we want to resume, the process no is not mixed and we stick to same GPU always
    device = torch.device("cuda", local_rank)

    return world_size, rank, local_rank, device

def seed_everything(seed, rank):
    seed = seed + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_main_process(rank):
    return rank == 0         #So we log only one main process, not all
    
def build_train_loader(cfg, world_size, rank):
    train_ds = MyDataset(
        split="train", dataset=cfg.dataset_name, columns=cfg.columns, shuffle=True, world_size=world_size, rank=rank, Vg=cfg.Vg, Vl=cfg.Vl
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.bs, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=(cfg.num_workers > 0),     #Currently yields {"image_crop": [Vg, 3, H, W]}
    ) 
    return train_ds, train_loader

def build_val_loader(cfg, world_size, rank):
    val_ds = MyDataset(
        split="validation", dataset=cfg.dataset_name, columns=cfg.columns, shuffle=False, world_size=1, rank=0, Vg=cfg.Vg, Vl=cfg.Vl
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.eval_bs, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=(cfg.num_workers > 0),
    )
    return val_ds, val_loader


######################################################### Model = Backbone + Projector #########################################################
class Resnet9Encoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()

        self.backbone = Resnet9(num_classes=1, num_channels=3)
        self.proj = MLP(
            in_channels=1024,
            hidden_channels=[2048, 2048, proj_dim],
            norm_layer="batch_norm",
        )

    def forward(self, x):
        N, V = x.shape[:2]

        flat_emb = self.backbone(x.flatten(0, 1))                      # [N*V, 1024]
        emb = flat_emb.reshape(N, V, -1).transpose(0, 1)  # [V, N, 1024]

        proj = self.proj(flat_emb).reshape(N, V, -1).transpose(0, 1)  #[V, N, proj_dim]

        return emb, proj


def build_model(cfg, device, local_rank):
    model = Resnet9Encoder(proj_dim=cfg.proj_dim).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    return model

######################################################### Custom LEJEPA Loss + Training requirements #########################################################
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

######################################################### Checkpoint loading + saving #########################################################

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
        mode="online",   # change to "offline" if cluster internet is flaky
    )
    return wandb

def ddp_mean(x, world_size):
    if not torch.is_tensor(x):
        x = torch.tensor(x, device="cuda")
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= world_size
    return y

######################################################### Main training loop #########################################################

def main():
    world_size, rank, local_rank, device = setup_ddp()
    seed_everything(seed=42, rank=rank)

    if is_main_process(rank):
        print(f"world_size = {world_size}, rank = {rank}, local_rank = {local_rank}")

    # temporary config placeholder
    @dataclass
    class CFG:
        dataset_name: str = "Smith42/galaxies"
        columns: list = None
        
        bs: int = 128
        eval_bs: int = 256
        num_workers: int = 1

        embed_dim: int = 1024
        proj_dim: int = 128

        Vg: int = 2
        Vl: int = 0

        lambd: float = 0.05                    #picked from https://github.com/galilai-group/lejepa/blob/main/README.md
        lr: float = 5e-4
        wd: float = 5e-4

        epochs: int = 4
        warmup_steps: int = 1000
        total_steps: int = 100000000  #100 M total stop condition, that's more than 12 epochs
        min_lr: float = 1e-6

        amp_dtype: str = "bf16"  #bf16 or fp16

        entity: str = "pranavktrpl-personal"
        project: str = "astrojepa"
        run_name: str = "resnet9_lejepa_smoke_0504"
        log_every: int = 10
        ckpt_every: int = 8000    #Train size = 8Mill, 128 bs, 4 GPUs, 16k steps per epoch, save every 8k steps, 2 ckpts per epoch
        save_dir: str = "./checkpoints"
        resume_path: str | None = None

    cfg = CFG(columns = ["image_crop"])

    train_ds, train_loader = build_train_loader(cfg, world_size, rank)
    val_ds, val_loader = build_val_loader(cfg, world_size, rank)

    model = build_model(cfg, device, local_rank)

    sigreg_fn = SIGReg().to(device)

    opt, scheduler = build_optimizer_and_scheduler(cfg, model)
    amp_dtype, scaler = build_amp(cfg)

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    wandb_run = setup_wandb(cfg, rank)

    start_epoch = 0
    global_step = 0

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

        for step, batch in enumerate(pbar):
            image_crops = batch["image_crop"].to(device, non_blocking=True)
            
            opt.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=amp_dtype):
                emb, proj = model(image_crops)
                
                loss, sim, sigreg = compute_lejepa_loss(proj = proj, sigreg_fn = sigreg_fn, lambd = cfg.lambd, num_global_views = cfg.Vg)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

                scheduler.step()
                global_step += 1
            
                loss_mean = ddp_mean(loss, world_size)
                sim_mean = ddp_mean(sim, world_size)
                sigreg_mean = ddp_mean(sigreg, world_size)
                lr = scheduler.get_last_lr()[0]
                if is_main_process(rank):
                    if global_step % cfg.log_every == 0:
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

                    if global_step % cfg.ckpt_every == 0:
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
        
        dist.barrier()
        
        if is_main_process(rank):
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
    
    if is_main_process(rank) and wandb_run is not None:
        wandb.finish()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()