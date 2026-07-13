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

from data.dataloaders import DesiSpectraDataset  # Spectra-specific: stream DESI patch crops instead of galaxy image crops.
from models.resnet9 import MLP #Resnet9, 

import lejepa

from configs.config import TrainConfig

from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


@dataclass
class SpectraTrainConfig(TrainConfig):
    # Spectra-specific: keep the LeJEPA optimizer/loss/checkpoint regimen, but point this script at DESI spectra.
    model_name: str = "desi_utbd_spectrum_transformer"
    pretrained_backbone: bool = False

    # Spectra-specific: DESI HF stream has one train split with spectra stored under sample["spectrum"].
    dataset_name: str = "UniverseTBD/mmu_desi_edr_sv3"#"MultimodalUniverse/desi"
    columns: list[str] = None
    spectra_dataset_name: str = "UniverseTBD/mmu_desi_edr_sv3"#"MultimodalUniverse/desi"
    spectra_columns: list[str] = None
    train_num_spectra: int = 1126441

    # Spectra-specific: 7781 flux values -> drop final value -> 389 ordered patches of length 20.
    spectra_patch_size: int = 20
    spectra_num_patches: int = 389
    spectra_pad_value: float = 0.0
    spectra_global_scale: tuple[float, float] = (0.90, 1.0)
    spectra_local_scale: tuple[float, float] = (0.35, 0.50)

    # Spectra-specific: transformer backbone knobs for 1D spectra patches.
    spectra_embed_dim: int = 768
    spectra_depth: int = 12
    spectra_num_heads: int = 12
    spectra_mlp_ratio: float = 4.0
    spectra_dropout: float = 0.0
    spectra_pooling: str = "cls"

    # Spectra-specific: image batch size is too large for 10 views of 390-token spectra sequences.
    bs: int = 16
    num_workers: int = 1

    # Spectra-specific: keep LeJEPA view counts and loss defaults, but save/log under spectra names.
    project: str = "astrojepa"
    run_name: str = "SPECTRA_run4_bs16_2806_Epoch5_utbd_desi"
    save_dir: str = "./checkpoints/SPECTRA_run4_bs16_2806_Epoch5_utbd_desi"
    resume_path: str | None = None
    wandb_run_id: str | None = None
    wandb_resume: str = "allow"

    def __post_init__(self):
        # Spectra-specific: avoid mutable list defaults while preserving TrainConfig-style fields.
        if self.columns is None:
            self.columns = ["spectrum"]
        if self.spectra_columns is None:
            self.spectra_columns = ["spectrum"]


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
    # Spectra-specific: use the DESI iterable dataset that returns patchified spectra crops and masks.
    train_ds = DesiSpectraDataset(
        split="train",
        dataset=getattr(cfg, "spectra_dataset_name", "MultimodalUniverse/desi"),
        columns=getattr(cfg, "spectra_columns", ["spectrum"]),
        shuffle=True,
        world_size=world_size,
        rank=rank,
        Vg=cfg.Vg,
        Vl=cfg.Vl,
        patch_size=getattr(cfg, "spectra_patch_size", 20),
        pad_value=getattr(cfg, "spectra_pad_value", 0.0),
        global_scale=getattr(cfg, "spectra_global_scale", (0.947, 1.0)),
        local_scale=getattr(cfg, "spectra_local_scale", (0.20, 0.394)),
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.bs, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=(cfg.num_workers > 0), timeout=600 if cfg.num_workers > 0 else 0, drop_last=True     # Spectra-specific: yields crops [B, V, 389, 20] plus masks [B, V, 389].
    ) 
    return train_ds, train_loader

# def build_val_loader(cfg, world_size, rank):
#     val_ds = DesiSpectraDataset(
#         split="validation", dataset=cfg.dataset_name, columns=cfg.columns, shuffle=False, world_size=1, rank=0, Vg=cfg.Vg, Vl=cfg.Vl
#     )
#     val_loader = DataLoader(
#         val_ds, batch_size=cfg.eval_bs, num_workers=cfg.num_workers, pin_memory=True, persistent_workers=(cfg.num_workers > 0),
#     )
#     return val_ds, val_loader


######################################################### Model = Backbone + Projector #########################################################
class SpectrumTransformerEncoder(nn.Module):
    # Spectra-specific: transformer encoder for patchified DESI spectra with explicit PAD masks.
    def __init__(
        self,
        proj_dim: int,
        patch_size: int = 20,
        num_patches: int = 389,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        pooling: str = "cls",
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.pooling = pooling

        # Spectra-specific: each ordered 20-flux patch becomes one transformer token.
        self.patch_embed = nn.Linear(patch_size, embed_dim)

        # Spectra-specific: cropped-out patches use this learned PAD token, not their numeric zero values.
        self.pad_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Spectra-specific: CLS pooling returns a pooled embedding, not class logits.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Spectra-specific: learned absolute positions preserve the 389 DESI patch order plus CLS.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Spectra-specific: no causal mask; attention only ignores PAD positions via src_key_padding_mask.
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self.proj = MLP(
            in_channels=embed_dim,
            hidden_channels=[2 * embed_dim, 2 * embed_dim, proj_dim],
            norm_layer="batch_norm",
        )

        self._init_spectrum_tokens()

    def _init_spectrum_tokens(self):
        # Spectra-specific: initialize learned spectral tokens/positions like ViT-style transformer params.
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.zeros_(self.pad_token)
        nn.init.normal_(self.pos_embed, std=0.02)

    def _pool(self, hidden, token_mask):
        if self.pooling == "cls":
            return hidden[:, 0]

        if self.pooling == "masked_mean":
            # Spectra-specific: mean-pool only real patch tokens; PAD tokens never contribute.
            patch_hidden = hidden[:, 1:]
            weights = token_mask[:, 1:].to(dtype=patch_hidden.dtype).unsqueeze(-1)
            denom = weights.sum(dim=1).clamp_min(1.0)
            return (patch_hidden * weights).sum(dim=1) / denom

        raise ValueError(f"Unknown spectra pooling mode: {self.pooling}")

    def _encode_views(self, crops, masks):
        # Spectra-specific: crops are [B, V, 389, 20] and masks are [B, V, 389].
        B, V, N, P = crops.shape
        if N != self.num_patches:
            raise ValueError(f"Expected {self.num_patches} spectra patches, got {N}")
        if P != self.patch_size:
            raise ValueError(f"Expected spectra patch size {self.patch_size}, got {P}")

        flat_crops = crops.flatten(0, 1)                  # [B*V, 389, 20]
        flat_masks = masks.flatten(0, 1).bool()           # [B*V, 389], True means real patch.

        tokens = self.patch_embed(flat_crops)             # [B*V, 389, D]
        pad_tokens = self.pad_token.expand(tokens.shape[0], tokens.shape[1], -1)
        tokens = torch.where(flat_masks.unsqueeze(-1), tokens, pad_tokens)

        cls_tokens = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)   # [B*V, 390, D]

        cls_mask = torch.ones(flat_masks.shape[0], 1, dtype=torch.bool, device=flat_masks.device)
        token_mask = torch.cat([cls_mask, flat_masks], dim=1)  # [B*V, 390], True means attendable token.

        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        key_padding_mask = ~token_mask                    # Spectra-specific: PyTorch expects True for ignored PAD tokens.

        hidden = self.transformer(tokens, src_key_padding_mask=key_padding_mask)
        hidden = self.norm(hidden)

        flat_emb = self._pool(hidden, token_mask)         # [B*V, D]
        flat_proj = self.proj(flat_emb)                   # [B*V, P]

        emb = flat_emb.reshape(B, V, -1).transpose(0, 1)
        proj = flat_proj.reshape(B, V, -1).transpose(0, 1)
        return emb, proj

    def forward(self, global_crops, global_masks, local_crops=None, local_masks=None):
        # Spectra-specific: masks are required because numeric zero can be real flux, not necessarily PAD.
        global_emb, global_proj = self._encode_views(global_crops, global_masks)

        if local_crops is None:
            return global_emb, global_proj

        if local_masks is None:
            raise ValueError("local_masks must be provided when local_crops is provided")

        local_emb, local_proj = self._encode_views(local_crops, local_masks)

        emb = torch.cat([global_emb, local_emb], dim=0)
        proj = torch.cat([global_proj, local_proj], dim=0)
        return emb, proj
    
        
def init_projector_weights(m):                      #Kaiming weight initialization - Only for the projector; the spectra encoder uses its own transformer initialization.
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
    # Spectra-specific: replace the image encoder with an explicit 1D spectra transformer encoder.
    model = SpectrumTransformerEncoder(
        proj_dim=cfg.proj_dim,
        patch_size=cfg.spectra_patch_size,
        num_patches=cfg.spectra_num_patches,
        embed_dim=cfg.spectra_embed_dim,
        depth=cfg.spectra_depth,
        num_heads=cfg.spectra_num_heads,
        mlp_ratio=cfg.spectra_mlp_ratio,
        dropout=cfg.spectra_dropout,
        pooling=cfg.spectra_pooling,
    )
    model.proj.apply(init_projector_weights)  # Spectra-specific: reuse the image projector initialization exactly.
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

def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, device="cuda"):
    ckpt = torch.load(path, map_location="cpu")

    if isinstance(model, DDP):
        model.module.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt["model"])

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        optimizer_to(optimizer, device)

    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    if "python_rng" in ckpt:
        random.setstate(ckpt["python_rng"])
    if "torch_rng" in ckpt:
        torch.set_rng_state(ckpt["torch_rng"].cpu())
    if "cuda_rng" in ckpt:
        torch.cuda.set_rng_state_all([x.cpu() for x in ckpt["cuda_rng"]])

    start_epoch = ckpt.get("epoch", 0)
    global_step = ckpt.get("global_step", 0)

    return start_epoch, global_step

def setup_wandb(cfg, rank):
    if not is_main_process(rank):
        return None

    # Spectra-specific: keep W&B lazy so importing/testing the spectra model does not block on logging setup.
    import wandb

    run = wandb.init(
        entity=cfg.entity,
        project=cfg.project,
        name=cfg.run_name,
        id=cfg.wandb_run_id,
        resume=cfg.wandb_resume if cfg.wandb_run_id is not None else None,
        config=asdict(cfg),
        mode="online",   # change to "offline" if cluster internet is flaky
    )
    return run

def ddp_mean(x, world_size):
    if not torch.is_tensor(x):
        x = torch.tensor(x, device="cuda")
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= world_size
    return y

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

    # Spectra-specific: use DESI spectra defaults instead of the image TrainConfig defaults.
    cfg = SpectraTrainConfig()
    cfg.safe_samples_per_rank_per_epoch = max(1, cfg.train_num_spectra // world_size)
    cfg.steps_per_epoch = cfg.safe_samples_per_rank_per_epoch // cfg.bs
    cfg.total_steps = cfg.epochs * cfg.steps_per_epoch
    
    if is_main_process(rank):
        print(f"steps_per_epoch = {cfg.steps_per_epoch}")
        print(f"total_steps = {cfg.total_steps}")

    train_ds, train_loader = build_train_loader(cfg, world_size, rank)
    # val_ds, val_loader = build_val_loader(cfg, world_size, rank)

    model = build_model(cfg, device, local_rank)
    raw_model = model.module if isinstance(model, DDP) else model

    # Spectra-specific: count the full spectra encoder as the backbone analogue, excluding the projector.
    proj_total, proj_trainable = count_parameters(raw_model.proj)
    model_total, model_trainable = count_parameters(raw_model)
    backbone_total = model_total - proj_total
    backbone_trainable = model_trainable - proj_trainable

    if is_main_process(rank):
        print(f"spectra encoder params: {backbone_total:,}")
        print(f"projector params: {proj_total:,}")
        print(f"total params: {model_total:,}")

    sigreg_fn = build_sigreg(cfg, device)

    opt, scheduler = build_optimizer_and_scheduler(cfg, model)
    amp_dtype, scaler = build_amp(cfg)

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    wandb_run = setup_wandb(cfg, rank)
    # print(f"Config: {cfg}")
    if wandb_run is not None:
        wandb_run.config.update({
            "model_name": cfg.model_name,
            # Spectra-specific: log patch/model geometry for DESI spectra runs.
            "spectra_patch_size": cfg.spectra_patch_size,
            "spectra_num_patches": cfg.spectra_num_patches,
            "spectra_global_scale": cfg.spectra_global_scale,
            "spectra_local_scale": cfg.spectra_local_scale,
            "spectra_embed_dim": cfg.spectra_embed_dim,
            "spectra_depth": cfg.spectra_depth,
            "spectra_num_heads": cfg.spectra_num_heads,
            "spectra_pooling": cfg.spectra_pooling,
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
    # resume_step_in_epoch = 0

    if cfg.resume_path is not None:
        start_epoch, global_step = load_checkpoint(
            cfg.resume_path,
            model=model,
            optimizer=opt,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        # resume_step_in_epoch = global_step % cfg.steps_per_epoch
        if is_main_process(rank):
            print(f"Resumed from {cfg.resume_path} at epoch={start_epoch}, step={global_step}")#, resume_step_in_epoch={resume_step_in_epoch}")

    for epoch in range(start_epoch, cfg.epochs):
        train_ds.set_epoch(epoch)
        model.train()

        pbar = tqdm(range(cfg.steps_per_epoch), disable=not is_main_process(rank), desc=f"epoch {epoch}")
        # epoch_step_offset = resume_step_in_epoch if epoch == start_epoch else 0
        train_iter = iter(train_loader)

        for step in pbar:
            try:
                batch = next(train_iter)
                has_batch = torch.tensor(1, device=device, dtype=torch.int32)
            except StopIteration:
                batch = None
                has_batch = torch.tensor(0, device=device, dtype=torch.int32)

            # DDP streaming safety: all ranks must agree that the next batch exists before any forward/loss collective.
            dist.all_reduce(has_batch, op=dist.ReduceOp.MIN)
            if has_batch.item() == 0:
                if is_main_process(rank):
                    print(f"Stopping epoch {epoch} early at local step {step}: at least one rank exhausted its HF stream shard.")
                break

            step_id = global_step + 1
            # Spectra-specific: crops are patch sequences [B, Vg, 389, 20], not image tensors.
            global_crops = batch["global_crops"].to(device, non_blocking=True)
            # Spectra-specific: masks tell the model which patch slots are real; zero-valued flux can still be real.
            global_masks = batch["global_masks"].to(device, non_blocking=True).bool()

            local_crops = batch["local_crops"]
            if local_crops is not None:
                local_crops = local_crops.to(device, non_blocking=True)
            # Spectra-specific: local masks follow the same real=True, PAD=False convention as global masks.
            local_masks = batch["local_masks"]
            if local_masks is not None:
                local_masks = local_masks.to(device, non_blocking=True).bool()
            
            opt.zero_grad(set_to_none=True)

            with autocast("cuda", dtype=amp_dtype):
                # Keep only forward + loss in autocast:
                # Spectra-specific: pass masks explicitly so PAD tokens are ignored by attention and pooling.
                emb, proj = model(global_crops, global_masks, local_crops, local_masks)
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
                        wandb_run.log(
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
        dist.barrier()
        if global_step >= cfg.total_steps:
            break
    
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
    dist.barrier()
    
    if wandb_run is not None:
        wandb_run.finish()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
