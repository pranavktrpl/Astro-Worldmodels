import os
from datetime import timedelta
import time

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
from models.resnet9 import MLP #Resnet9, 

import lejepa

import timm

from configs.config import TrainConfig

from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import wandb


######################################################### fast paths #########################################################
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


######################################################### DDP Setup + Dataloading #########################################################
def setup_ddp():
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=60),
    )

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
        train_ds, batch_size=cfg.bs, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=(cfg.num_workers > 0), timeout=600 if cfg.num_workers > 0 else 0     #Currently yields {"image_crop": [Vg, 3, H, W]}
    ) 
    return train_ds, train_loader

# def build_val_loader(cfg, world_size, rank):
#     val_ds = MyDataset(
#         split="validation", dataset=cfg.dataset_name, columns=cfg.columns, shuffle=False, world_size=1, rank=0, Vg=cfg.Vg, Vl=cfg.Vl
#     )
#     val_loader = DataLoader(
#         val_ds, batch_size=cfg.eval_bs, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=(cfg.num_workers > 0),
#     )
#     return val_ds, val_loader


######################################################### Model = Backbone + Projector #########################################################
class TimmEncoder(nn.Module):
    def __init__(self, model_name: str, proj_dim: int, pretrained: bool = False):
        super().__init__()

        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,   # remove classifier, return pooled embedding
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )

        embed_dim = getattr(self.backbone, "num_features", None)
        if embed_dim is None:
            raise ValueError(f"Could not infer num_features for {model_name}")

        self.embed_dim = embed_dim
        self.proj = MLP(
            in_channels=embed_dim,
            hidden_channels=[2 * embed_dim, 2 * embed_dim, proj_dim],
            norm_layer="batch_norm",
        )

    def _encode_views(self, x):
        B, V = x.shape[:2]
        flat = x.flatten(0, 1)            # [B*V, C, H, W]
        flat_emb = self.backbone(flat)    # [B*V, D]
        flat_proj = self.proj(flat_emb)   # [B*V, P]

        emb = flat_emb.reshape(B, V, -1).transpose(0, 1)
        proj = flat_proj.reshape(B, V, -1).transpose(0, 1)
        return emb, proj


    def forward(self, global_x, local_x=None):
        global_emb, global_proj = self._encode_views(global_x)

        if local_x is None:
            return global_emb, global_proj

        local_emb, local_proj = self._encode_views(local_x)

        emb = torch.cat([global_emb, local_emb], dim=0)
        proj = torch.cat([global_proj, local_proj], dim=0)
        return emb, proj
    
        
def init_projector_weights(m):                      #Kaiming weight initialization - Only for the projector, the model initialization is backbone dependent, which is handled by timm
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        if m.weight is not None:
            nn.init.ones_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def build_model(cfg, device, local_rank):
    model = TimmEncoder(
        model_name=cfg.model_name,
        proj_dim=cfg.proj_dim,
        pretrained=cfg.pretrained_backbone,
    )
    model.proj.apply(init_projector_weights)  #model linear layers auto by timm, only help for MLP projector (CUSTOM WRITTEN)
    model = model.to(device)
    model = DDP(model, 
                device_ids=[local_rank], 
                output_device=local_rank,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                static_graph=True,
    )
    return model

######################################################### Custom LEJEPA Loss + Training requirements #########################################################
def build_sigreg(cfg, device):
    univariate_test = lejepa.univariate.EppsPulley(
        n_points=cfg.sigreg_num_points
    )
    sigreg_fn = lejepa.multivariate.SlicingUnivariateTest(
        univariate_test=univariate_test,
        num_slices=cfg.sigreg_num_slices,
    )
    return sigreg_fn.to(device)

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

######################################################### Checkpoint loading + saving + other utilities ##########################################################
def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable

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

# def save_checkpoint_atomic(cfg, model, optimizer, scheduler, scaler, epoch, global_step, path):
#     tmp_path = path + ".tmp"
#     save_checkpoint(
#         cfg=cfg,
#         model=model,
#         optimizer=optimizer,
#         scheduler=scheduler,
#         scaler=scaler,
#         epoch=epoch,
#         global_step=global_step,
#         path=tmp_path,
#     )
#     os.replace(tmp_path, path)

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

def rank_heartbeat(cfg, rank, step, tag):
    debug_dir = os.path.join(cfg.save_dir, "debug_logs")
    log_path = os.path.join(debug_dir, f"rank_{rank}.txt")
    with open(log_path, "a") as f:
        f.write(f"[rank {rank}] step={step} {tag} t={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.flush()

def feature_std_stats(x):   #Only to monitor collapse of embeddings and projections
    # x: [V, B, D]
    x_flat = x.transpose(0, 1).reshape(-1, x.size(-1))   # [B*V, D]
    std_per_dim = x_flat.std(dim=0)
    mean_std = std_per_dim.mean()
    min_std = std_per_dim.min()
    max_std = std_per_dim.max()
    return mean_std, min_std, max_std

######################################################### Main training loop #########################################################

def main():
    world_size, rank, local_rank, device = setup_ddp()
    seed_everything(seed=42, rank=rank)

    if is_main_process(rank):
        print(f"world_size = {world_size}, rank = {rank}, local_rank = {local_rank}")

    # temporary config placeholder
    cfg = TrainConfig()
    heartbeat_every = 100 #500 steps
    debug_dir = os.path.join(cfg.save_dir, "debug_logs")
    os.makedirs(debug_dir, exist_ok=True)

    
    train_ds, train_loader = build_train_loader(cfg, world_size, rank)
    # val_ds, val_loader = build_val_loader(cfg, world_size, rank)

    model = build_model(cfg, device, local_rank)
    raw_model = model.module if isinstance(model, DDP) else model

    backbone_total, backbone_trainable = count_parameters(raw_model.backbone)
    proj_total, proj_trainable = count_parameters(raw_model.proj)
    model_total, model_trainable = count_parameters(raw_model)

    if is_main_process(rank):
        print(f"backbone params: {backbone_total:,}")
        print(f"projector params: {proj_total:,}")
        print(f"total params: {model_total:,}")

    sigreg_fn = build_sigreg(cfg, device)

    opt, scheduler = build_optimizer_and_scheduler(cfg, model)
    amp_dtype, scaler = build_amp(cfg)

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    wandb_run = setup_wandb(cfg, rank)
    # print(f"Config: {cfg}")
    if wandb_run is not None:
        wandb.config.update({
            "model_name": cfg.model_name,
            "backbone_num_params": backbone_total,
            "backbone_num_params_m": backbone_total / 1e6,
            "projector_num_params": proj_total,
            "projector_num_params_m": proj_total / 1e6,
            "model_num_params": model_total,
            "model_num_params_m": model_total / 1e6,
            "trainable_num_params": model_trainable,
            "trainable_num_params_m": model_trainable / 1e6,
        }, allow_val_change=True)
    
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
            step_id = global_step + 1
            if step_id % heartbeat_every == 0:
                rank_heartbeat(cfg, rank, step_id, "batch_fetched")
            # image_crops = batch["image_crop"].to(device, non_blocking=True)
            global_crops = batch["global_crops"].to(device, non_blocking=True)

            local_crops = batch["local_crops"]
            if local_crops is not None:
                local_crops = local_crops.to(device, non_blocking=True)
            if step_id % heartbeat_every == 0:
                rank_heartbeat(cfg, rank, step_id, "h2d_done")
            
            opt.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=amp_dtype):
                # Keep only forward + loss in autocast:
                emb, proj = model(global_crops, local_crops)
                loss, sim, sigreg = compute_lejepa_loss(proj = proj, sigreg_fn = sigreg_fn, lambd = cfg.lambd, num_global_views = cfg.Vg)
            if step_id % heartbeat_every == 0:
                rank_heartbeat(cfg, rank, step_id, "forward_loss_done")

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if step_id % heartbeat_every == 0:
                    rank_heartbeat(cfg, rank, step_id, "backward_done")
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if step_id % heartbeat_every == 0:
                    rank_heartbeat(cfg, rank, step_id, "backward_done")
                opt.step()
            

            scheduler.step()
            global_step += 1
            if step_id % heartbeat_every == 0:
                rank_heartbeat(cfg, rank, step_id, "optim_done")
            
            lr = scheduler.get_last_lr()[0]
            
            if global_step % cfg.log_every == 0:
                with torch.no_grad():
                    emb_mean_std, emb_min_std, emb_max_std = feature_std_stats(emb.detach())
                    proj_mean_std, proj_min_std, proj_max_std = feature_std_stats(proj.detach())
                
                loss_mean = ddp_mean(loss.detach(), world_size)
                sim_mean = ddp_mean(sim.detach(), world_size)
                sigreg_mean = ddp_mean(sigreg.detach(), world_size)
                
                emb_mean_std_mean = ddp_mean(emb_mean_std.detach(), world_size)
                emb_min_std_mean = ddp_mean(emb_min_std.detach(), world_size)
                proj_mean_std_mean = ddp_mean(proj_mean_std.detach(), world_size)
                proj_min_std_mean = ddp_mean(proj_min_std.detach(), world_size)

                if is_main_process(rank):
                    # print(f"loss = {loss_mean.item():.4f}, sim = {sim_mean.item():.4f}, sigreg = {sigreg_mean.item():.4f}, lr = {lr:.2e}")
                    pbar.set_postfix(
                        loss=f"{loss_mean.item():.4f}",
                        sim=f"{sim_mean.item():.4f}",
                        sigreg=f"{sigreg_mean.item():.4f}",
                        emb_std=f"{emb_mean_std_mean.item():.4f}",
                        proj_std=f"{proj_mean_std_mean.item():.4f}",
                        lr=f"{lr:.2e}",
                    )

                    if wandb_run is not None:
                        wandb.log(
                            {
                                "train/loss": loss_mean.item(),
                                "train/sim": sim_mean.item(),
                                "train/sigreg": sigreg_mean.item(),
                                "train/emb_mean_std": emb_mean_std_mean.item(),
                                "train/emb_min_std": emb_min_std_mean.item(),
                                "train/proj_mean_std": proj_mean_std_mean.item(),
                                "train/proj_min_std": proj_min_std_mean.item(),
                                "train/lr": lr,
                                "train/epoch": epoch,
                                "train/step": global_step,
                            },
                            step=global_step,
                        )

            if global_step % cfg.ckpt_every == 0:
                dist.barrier()
                if is_main_process(rank):
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
                dist.barrier()

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
                path=os.path.join(cfg.save_dir, f"last_epoch_{epoch}.pt"),
            )
        if global_step >= cfg.total_steps:
            break
    
    if is_main_process(rank):
        if is_main_process(rank):
            save_checkpoint(
                cfg=cfg,
                model=model,
                optimizer=opt,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch + 1,
                global_step=global_step,
                path=os.path.join(cfg.save_dir, f"complete.pt"),
            )
    
    if wandb_run is not None:
        wandb.finish()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()