import argparse
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import wandb

from data.cross_modal import build_combined_desi_metadata
from data.cross_modal_clip import CombinedPairedImageSpectrumCLIPDataset
from models.cross_modal_posttrained_clip import CrossModalPosttrainedCLIPModel


@dataclass
class CrossModalPostTrainCLIPConfig:
    pair_dataset_dirs: tuple[str, ...] = (
        "/mnt/datasets/pranav/astroclip",
        "/mnt/datasets/pranav/desi_legacysurvey_xmatch",
        "/mnt/datasets/pranav/desi_dr8_manual_unique_xmatch",
    )
    split_seed: int = 42
    train_num_pairs: int | None = None
    parquet_batch_size: int = 64

    image_model_name: str = "vit_large_patch14_dinov2.lvd142m"
    image_checkpoint: str = (
        "./checkpoints/VitLargePatch14_OfficialTrain5_Epoch5_2504/step_52000.pt"
    )
    spectrum_checkpoint: str = (
        "./checkpoints/SpectraV2_DESI_GlobalLocalLeJEPA_MeanStd_CorrectedEpochs/complete.pt"
    )
    shared_dim: int = 512
    pool_num_heads: int = 4
    pool_dropout: float = 0.1

    spectra_num_pixels: int = 7781
    spectra_patch_size: int = 20
    spectra_num_patches: int = 389
    spectra_num_views: int = 2
    spectra_mask_ratio: float = 0.0
    spectra_mask_span: tuple[int, int] = (2, 12)
    spectra_disjoint_view_masks: bool = False
    spectra_min_valid_patch_fraction: float = 0.50
    spectra_noise_scale: float = 0.0
    spectra_max_normalized_noise_std: float = 3.0
    spectra_embed_dim: int = 768
    spectra_depth: int = 12
    spectra_num_heads: int = 12
    spectra_mlp_ratio: float = 4.0
    spectra_dropout: float = 0.0
    spectra_pooling: str = "cls"

    logit_scale: float = 15.5
    alignment_view_index: int = 1

    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 10
    max_steps: int | None = None
    steps_per_epoch: int | None = None
    total_steps: int | None = None

    lr: float = 1.0e-4
    weight_decay: float = 5.0e-2
    min_lr: float = 1.0e-6
    warmup_steps: int = 1_000
    grad_clip_norm: float = 1.0
    amp_dtype: str = "bf16"

    seed: int = 42
    entity: str = "pranavktrpl-personal"
    project: str = "astrojepa"
    run_name: str = "CrossModalPostTrain_DESI307K_AstroCLIP_InfoNCE"
    wandb_mode: str = "online"
    wandb_run_id: str | None = None
    wandb_resume: str = "allow"

    log_every: int = 10
    save_dir: str = (
        "./checkpoints/CrossModalPostTrain_DESI307K_AstroCLIP_InfoNCE"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="AstroCLIP-align frozen pretrained image and DESI encoders"
    )
    parser.add_argument(
        "--pair-dataset-dir",
        action="append",
        dest="pair_dataset_dirs",
        help="Repeat once per paired DESI source directory.",
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--shared-dim", type=int)
    parser.add_argument("--pool-num-heads", type=int)
    parser.add_argument("--pool-dropout", type=float)
    parser.add_argument("--logit-scale", type=float)
    parser.add_argument("--image-checkpoint")
    parser.add_argument("--spectrum-checkpoint")
    parser.add_argument("--run-name")
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--save-dir")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
    )
    return parser.parse_args()


def apply_cli_overrides(config, args):
    overrides = {
        "pair_dataset_dirs": (
            tuple(args.pair_dataset_dirs)
            if args.pair_dataset_dirs is not None
            else None
        ),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "warmup_steps": args.warmup_steps,
        "lr": args.learning_rate,
        "shared_dim": args.shared_dim,
        "pool_num_heads": args.pool_num_heads,
        "pool_dropout": args.pool_dropout,
        "logit_scale": args.logit_scale,
        "image_checkpoint": args.image_checkpoint,
        "spectrum_checkpoint": args.spectrum_checkpoint,
        "run_name": args.run_name,
        "wandb_run_id": args.wandb_run_id,
        "save_dir": args.save_dir,
        "wandb_mode": args.wandb_mode,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(config, name, value)
    return config


def validate_checkpoint_layout(config):
    save_dir = Path(config.save_dir).resolve()
    sources = {
        "image": Path(config.image_checkpoint).resolve(),
        "spectrum": Path(config.spectrum_checkpoint).resolve(),
    }
    for label, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"Missing {label} source checkpoint: {source}")
        if source == save_dir or source.is_relative_to(save_dir):
            raise ValueError(
                f"The {label} source checkpoint must not be inside save_dir: "
                f"source={source}, save_dir={save_dir}"
            )
    if len(set(sources.values())) != len(sources):
        raise ValueError("Image and spectrum source checkpoints must be different")
    return sources


def assert_safe_output_path(config, path):
    path = Path(path).resolve()
    save_dir = Path(config.save_dir).resolve()
    if not path.is_relative_to(save_dir):
        raise ValueError(f"Refusing checkpoint write outside save_dir: {path}")
    protected = {
        Path(config.image_checkpoint).resolve(),
        Path(config.spectrum_checkpoint).resolve(),
    }
    if path in protected:
        raise ValueError(f"Refusing to overwrite a source checkpoint: {path}")


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=60),
        device_id=device,
    )
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    return world_size, rank, local_rank, device


def is_main_process(rank):
    return rank == 0


def seed_everything(seed, rank):
    process_seed = seed + rank
    random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)


def load_pair_metadata(config, rank):
    payload = [None]
    if is_main_process(rank):
        payload[0] = build_combined_desi_metadata(config.pair_dataset_dirs)
    dist.broadcast_object_list(payload, src=0)
    info = payload[0]
    if info is None:
        raise RuntimeError("Failed to broadcast combined paired dataset metadata")
    config.train_num_pairs = int(info["pair_rows"])
    return info


def build_train_loader(config, world_size, rank, dataset_info):
    dataset = CombinedPairedImageSpectrumCLIPDataset(
        dataset_info=dataset_info,
        world_size=world_size,
        rank=rank,
        num_workers=max(1, config.num_workers),
        split_seed=config.split_seed,
        parquet_batch_size=config.parquet_batch_size,
        spectra_num_views=config.spectra_num_views,
        spectra_patch_size=config.spectra_patch_size,
        spectra_num_pixels=config.spectra_num_pixels,
        spectra_mask_ratio=config.spectra_mask_ratio,
        spectra_mask_span=config.spectra_mask_span,
        spectra_disjoint_view_masks=config.spectra_disjoint_view_masks,
        spectra_min_valid_patch_fraction=config.spectra_min_valid_patch_fraction,
        spectra_noise_scale=config.spectra_noise_scale,
        spectra_max_normalized_noise_std=config.spectra_max_normalized_noise_std,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2 if config.num_workers > 0 else None,
        timeout=600 if config.num_workers > 0 else 0,
        drop_last=True,
    )
    return dataset, loader


def build_model(config, device, local_rank, rank):
    model = CrossModalPosttrainedCLIPModel(
        image_model_name=config.image_model_name,
        shared_dim=config.shared_dim,
        spectra_patch_size=config.spectra_patch_size,
        spectra_num_patches=config.spectra_num_patches,
        spectra_embed_dim=config.spectra_embed_dim,
        spectra_depth=config.spectra_depth,
        spectra_num_heads=config.spectra_num_heads,
        spectra_mlp_ratio=config.spectra_mlp_ratio,
        spectra_dropout=config.spectra_dropout,
        spectra_pooling=config.spectra_pooling,
        pool_num_heads=config.pool_num_heads,
        pool_dropout=config.pool_dropout,
    )
    source_payload = [None]
    if is_main_process(rank):
        source_payload[0] = model.load_pretrained_backbones(
            config.image_checkpoint,
            config.spectrum_checkpoint,
        )
    dist.broadcast_object_list(source_payload, src=0)
    if source_payload[0] is None:
        raise RuntimeError("Failed to broadcast pretrained source metadata")
    model.source_metadata = source_payload[0]
    model = model.to(device)
    distributed_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=True,
    )
    return distributed_model


def gather_with_grad(features):
    if dist.get_world_size() == 1:
        return features
    return torch.cat(dist_nn.all_gather(features), dim=0)


def compute_distributed_astroclip_loss(image_features, spectrum_features, logit_scale):
    """Official symmetric CLIP loss with the DDP-global batch as negatives."""
    if image_features.ndim != 2 or spectrum_features.ndim != 2:
        raise ValueError("CLIP features must have shape [batch, dimension]")
    if image_features.shape != spectrum_features.shape:
        raise ValueError(
            f"Image/spectrum feature mismatch: {image_features.shape} vs "
            f"{spectrum_features.shape}"
        )

    image = F.normalize(image_features.float(), dim=-1, eps=1.0e-3)
    spectrum = F.normalize(spectrum_features.float(), dim=-1, eps=1.0e-3)
    all_image = gather_with_grad(image)
    all_spectrum = gather_with_grad(spectrum)
    local_batch = image.shape[0]
    labels = dist.get_rank() * local_batch + torch.arange(
        local_batch, device=image.device
    )

    image_logits = logit_scale * image @ all_spectrum.T
    spectrum_logits = logit_scale * spectrum @ all_image.T
    image_loss = F.cross_entropy(image_logits, labels)
    spectrum_loss = F.cross_entropy(spectrum_logits, labels)
    return {
        "loss": 0.5 * (image_loss + spectrum_loss),
        "image_to_spectrum_loss": image_loss,
        "spectrum_to_image_loss": spectrum_loss,
        "image_to_spectrum_top1": (image_logits.argmax(dim=1) == labels).float().mean(),
        "spectrum_to_image_top1": (
            spectrum_logits.argmax(dim=1) == labels
        ).float().mean(),
        "positive_cosine": (image * spectrum).sum(dim=-1).mean(),
        "logit_scale": image.new_tensor(logit_scale),
    }


def build_optimizer_and_scheduler(config, model):
    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = (
            parameter.ndim <= 1
            or name.endswith(".bias")
            or "cls_token" in name
            or "mask_token" in name
            or "pos_embed" in name
        )
        (no_decay_parameters if no_decay else decay_parameters).append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay_parameters,
                "weight_decay": config.weight_decay,
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            },
        ],
        lr=config.lr,
    )
    warmup_steps = min(
        config.warmup_steps,
        max(1, config.total_steps - 1),
    )
    warmup = LinearLR(
        optimizer,
        start_factor=0.01,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.total_steps - warmup_steps),
        eta_min=config.min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )
    return optimizer, scheduler


def build_amp(config):
    if config.amp_dtype == "bf16":
        return torch.bfloat16, GradScaler("cuda", enabled=False)
    if config.amp_dtype == "fp16":
        return torch.float16, GradScaler("cuda", enabled=True)
    raise ValueError(f"Unknown AMP dtype: {config.amp_dtype}")


def count_parameters(module):
    return sum(parameter.numel() for parameter in module.parameters())


def count_trainable_parameters(module):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def ddp_mean(value, world_size):
    reduced = value.detach().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return reduced / world_size


def ddp_finite_mean(values):
    values = values.detach().float()
    finite = torch.isfinite(values)
    total = torch.where(finite, values, torch.zeros_like(values)).sum()
    count = finite.sum().to(total.dtype)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return total / count.clamp_min(1.0)


def feature_std_stats(features):
    flattened = features.transpose(0, 1).reshape(-1, features.shape[-1]).float()
    per_dimension = flattened.std(dim=0)
    return per_dimension.mean(), per_dimension.min(), per_dimension.max()


def branch_gradient_norm(module):
    squared_norm = torch.zeros((), device=next(module.parameters()).device)
    for parameter in module.parameters():
        if parameter.grad is not None:
            squared_norm = squared_norm + parameter.grad.detach().float().square().sum()
    return squared_norm.sqrt()


def alignment_diagnostics(image_projections, spectrum_projections):
    image = image_projections.float()
    spectrum = spectrum_projections.float()
    paired = (image[:, None] - spectrum[None, :]).square().mean()
    shuffled = (
        image[:, None] - spectrum.roll(shifts=1, dims=1)[None, :]
    ).square().mean()
    image_view_distance = (image[0] - image[1]).square().mean()
    spectrum_view_distance = (spectrum[0] - spectrum[1]).square().mean()
    return {
        "paired_distance": paired,
        "shuffled_distance": shuffled,
        "paired_to_shuffled_ratio": paired / shuffled.clamp_min(1.0e-12),
        "image_view_distance": image_view_distance,
        "spectrum_view_distance": spectrum_view_distance,
    }


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(config, model, epoch, step_in_epoch, global_step):
    raw_model = model.module if isinstance(model, DDP) else model
    return {
        "format": "cross_modal_posttrained_astroclip_adapters_v1",
        "alignment_heads": raw_model.alignment_state_dict(),
        "source_checkpoints": raw_model.source_metadata,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "config": asdict(config),
        "objective": "symmetric_global_batch_infonce",
        "fixed_logit_scale": config.logit_scale,
    }


def save_checkpoint(config, model, epoch, step_in_epoch, global_step, path):
    assert_safe_output_path(config, path)
    atomic_save(
        checkpoint_payload(config, model, epoch, step_in_epoch, global_step),
        path,
    )


def prune_named_checkpoints(save_dir, prefix, keep_last):
    if keep_last < 1:
        raise ValueError("Checkpoint retention count must be positive")
    checkpoints = []
    for path in Path(save_dir).glob(f"{prefix}_*.pt"):
        try:
            number = int(path.stem.removeprefix(f"{prefix}_"))
        except ValueError:
            continue
        checkpoints.append((number, path))
    checkpoints.sort()
    for _, path in checkpoints[:-keep_last]:
        path.unlink()


def prune_periodic_checkpoints(save_dir, keep_last):
    prune_named_checkpoints(save_dir, "step", keep_last)


def prune_epoch_checkpoints(save_dir, keep_last):
    prune_named_checkpoints(save_dir, "epoch", keep_last)



def setup_wandb(
    config,
    rank,
    dataset_info,
    parameter_counts,
    source_metadata,
):
    if not is_main_process(rank) or config.wandb_mode == "disabled":
        return None
    dataset_summary = {
        name: value
        for name, value in dataset_info.items()
        if name != "files"
    }
    run = wandb.init(
        entity=config.entity,
        project=config.project,
        name=config.run_name,
        id=config.wandb_run_id,
        resume=(
            config.wandb_resume
            if config.wandb_run_id is not None
            else None
        ),
        config={
            **asdict(config),
            **parameter_counts,
            "global_batch_size": config.batch_size * dist.get_world_size(),
            "paired_dataset": dataset_summary,
            "source_checkpoints": source_metadata,
            "training_regime": "frozen_backbones_astroclip_adapters",
            "objective": "symmetric_global_batch_infonce",
            "fixed_logit_scale": config.logit_scale,
            "negative_set": "DDP-global batch",
        },
        mode=config.wandb_mode,
    )
    return run


def next_batch_synchronized(iterator, device):
    try:
        batch = next(iterator)
        available = torch.ones((), device=device, dtype=torch.int32)
    except StopIteration:
        batch = None
        available = torch.zeros((), device=device, dtype=torch.int32)
    dist.all_reduce(available, op=dist.ReduceOp.MIN)
    return batch if available.item() == 1 else None


def main():
    config = apply_cli_overrides(CrossModalPostTrainCLIPConfig(), parse_args())
    validate_checkpoint_layout(config)
    world_size, rank, local_rank, device = setup_ddp()
    seed_everything(config.seed, rank)

    dataset_info = load_pair_metadata(config, rank)
    train_dataset, train_loader = build_train_loader(
        config,
        world_size,
        rank,
        dataset_info,
    )
    usable_batches = train_dataset.usable_batches_per_rank(config.batch_size)
    config.steps_per_epoch = min(usable_batches)
    if config.steps_per_epoch < 1:
        raise ValueError("Global batch size exceeds the paired training set")
    natural_total_steps = config.epochs * config.steps_per_epoch
    config.total_steps = (
        min(config.max_steps, natural_total_steps)
        if config.max_steps is not None
        else natural_total_steps
    )
    if config.total_steps < 1:
        raise ValueError("Training must contain at least one optimizer step")

    if is_main_process(rank):
        print(f"world_size={world_size}")
        print(f"paired rows across all sources={config.train_num_pairs:,}")
        print(f"rows per rank={train_dataset.rows_per_rank}")
        print(f"worker rows by rank={train_dataset.worker_rows_by_rank}")
        print(f"usable batches per rank={usable_batches}")
        print(f"steps_per_epoch={config.steps_per_epoch:,}")
        print(f"total_steps={config.total_steps:,}")

    model = build_model(config, device, local_rank, rank)
    raw_model = model.module

    parameter_counts = {
        "image_encoder_num_params": count_parameters(raw_model.image_encoder),
        "image_encoder_trainable_params": count_trainable_parameters(
            raw_model.image_encoder
        ),
        "spectrum_encoder_num_params": count_parameters(raw_model.spectrum_encoder),
        "spectrum_encoder_trainable_params": count_trainable_parameters(
            raw_model.spectrum_encoder
        ),
        "total_num_params": count_parameters(raw_model),
        "total_trainable_params": count_trainable_parameters(raw_model),
    }
    if is_main_process(rank):
        for name, count in parameter_counts.items():
            print(f"{name}={count:,}")
        print(f"source_checkpoints={raw_model.source_metadata}")

    optimizer, scheduler = build_optimizer_and_scheduler(config, model)
    amp_dtype, scaler = build_amp(config)

    if is_main_process(rank):
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    dist.barrier()
    wandb_run = setup_wandb(
        config,
        rank,
        dataset_info,
        parameter_counts,
        raw_model.source_metadata,
    )

    start_epoch = 0
    resume_step_in_epoch = 0
    global_step = 0

    log_started = time.perf_counter()
    logged_samples = 0
    stop_training = False

    for epoch in range(start_epoch, config.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        iterator = iter(train_loader)

        steps_to_skip = resume_step_in_epoch if epoch == start_epoch else 0
        for _ in range(steps_to_skip):
            skipped = next_batch_synchronized(iterator, device)
            if skipped is None:
                raise RuntimeError("Paired stream ended while restoring epoch position")

        progress = tqdm(
            total=config.steps_per_epoch,
            initial=steps_to_skip,
            disable=not is_main_process(rank),
            desc=f"epoch {epoch}",
        )
        step_in_epoch = steps_to_skip

        while step_in_epoch < config.steps_per_epoch:
            if global_step >= config.total_steps:
                stop_training = True
                break

            batch = next_batch_synchronized(iterator, device)
            if batch is None:
                if is_main_process(rank):
                    print(
                        f"Stopping epoch {epoch} at step {step_in_epoch}: "
                        "at least one distributed data shard was exhausted."
                    )
                break

            image_views = batch["image_views"].to(device, non_blocking=True)
            spectrum_views = batch["spectrum_views"].to(device, non_blocking=True)
            spectrum_valid_pixels = batch["spectrum_valid_pixels"].to(
                device,
                non_blocking=True,
            ).bool()
            spectrum_jepa_masks = batch["spectrum_jepa_masks"].to(
                device,
                non_blocking=True,
            ).bool()

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=amp_dtype):
                outputs = model(
                    image_views,
                    spectrum_views,
                    spectrum_valid_pixels,
                    spectrum_jepa_masks,
                )
                losses = compute_distributed_astroclip_loss(
                    outputs["image_projections"][config.alignment_view_index],
                    outputs["spectrum_projections"][config.alignment_view_index],
                    config.logit_scale,
                )
                loss = losses["loss"]

            should_log = (global_step + 1) % config.log_every == 0
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            image_grad_norm = None
            spectrum_grad_norm = None
            if should_log:
                image_grad_norm = branch_gradient_norm(raw_model.image_encoder)
                spectrum_grad_norm = branch_gradient_norm(
                    raw_model.spectrum_encoder
                )

            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.grad_clip_norm,
            )
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            global_step += 1
            step_in_epoch += 1
            progress.update(1)
            logged_samples += world_size * image_views.shape[0]

            if should_log:
                with torch.no_grad():
                    diagnostics = alignment_diagnostics(
                        outputs["image_projections"],
                        outputs["spectrum_projections"],
                    )
                    image_embedding_std = feature_std_stats(
                        outputs["image_embeddings"]
                    )
                    spectrum_embedding_std = feature_std_stats(
                        outputs["spectrum_embeddings"]
                    )
                    image_projection_std = feature_std_stats(
                        outputs["image_projections"]
                    )
                    spectrum_projection_std = feature_std_stats(
                        outputs["spectrum_projections"]
                    )

                metrics = {
                    **{
                        f"train/{name}": ddp_mean(value, world_size).item()
                        for name, value in losses.items()
                    },
                    **{
                        f"alignment/{name}": ddp_mean(value, world_size).item()
                        for name, value in diagnostics.items()
                    },
                    "collapse/image_embedding_mean_std": ddp_mean(
                        image_embedding_std[0], world_size
                    ).item(),
                    "collapse/image_embedding_min_std": ddp_mean(
                        image_embedding_std[1], world_size
                    ).item(),
                    "collapse/spectrum_embedding_mean_std": ddp_mean(
                        spectrum_embedding_std[0], world_size
                    ).item(),
                    "collapse/spectrum_embedding_min_std": ddp_mean(
                        spectrum_embedding_std[1], world_size
                    ).item(),
                    "collapse/image_projection_mean_std": ddp_mean(
                        image_projection_std[0], world_size
                    ).item(),
                    "collapse/image_projection_min_std": ddp_mean(
                        image_projection_std[1], world_size
                    ).item(),
                    "collapse/spectrum_projection_mean_std": ddp_mean(
                        spectrum_projection_std[0], world_size
                    ).item(),
                    "collapse/spectrum_projection_min_std": ddp_mean(
                        spectrum_projection_std[1], world_size
                    ).item(),
                    "optimization/image_grad_norm": ddp_mean(
                        image_grad_norm, world_size
                    ).item(),
                    "optimization/spectrum_grad_norm": ddp_mean(
                        spectrum_grad_norm, world_size
                    ).item(),
                    "optimization/total_grad_norm": ddp_mean(
                        total_grad_norm, world_size
                    ).item(),
                    "optimization/lr": scheduler.get_last_lr()[0],
                    "data/valid_pixel_fraction": ddp_mean(
                        batch["valid_pixel_fraction"].to(device).mean(),
                        world_size,
                    ).item(),
                    "data/uncertainty_available_fraction": ddp_mean(
                        batch["uncertainty_available"].to(device).float().mean(),
                        world_size,
                    ).item(),
                    "data/spectrum_mask_fraction": ddp_mean(
                        spectrum_jepa_masks.float().mean(),
                        world_size,
                    ).item(),
                    "data/mean_known_pair_separation_arcsec": ddp_finite_mean(
                        batch["separation_arcsec"].to(device),
                    ).item(),
                    "runtime/samples_per_second": (
                        logged_samples / max(time.perf_counter() - log_started, 1e-6)
                    ),
                    "runtime/max_memory_gib": (
                        torch.cuda.max_memory_allocated(device) / 2**30
                    ),
                    "train/epoch": epoch + step_in_epoch / config.steps_per_epoch,
                    "train/step": global_step,
                }
                for source in dataset_info["sources"]:
                    source_fraction = (
                        batch["source_index"].to(device) == source["source_index"]
                    ).float().mean()
                    metrics[
                        f"data/source_fraction/{source['dataset_format']}"
                    ] = ddp_mean(source_fraction, world_size).item()

                if is_main_process(rank):
                    progress.set_postfix(
                        loss=f"{metrics['train/loss']:.4f}",
                        i2s=f"{metrics['train/image_to_spectrum_loss']:.4f}",
                        ratio=f"{metrics['alignment/paired_to_shuffled_ratio']:.3f}",
                        lr=f"{metrics['optimization/lr']:.2e}",
                    )
                    if wandb_run is not None:
                        wandb_run.log(metrics, step=global_step)
                log_started = time.perf_counter()
                logged_samples = 0
                torch.cuda.reset_peak_memory_stats(device)

        progress.close()
        resume_step_in_epoch = 0
        dist.barrier()
        if is_main_process(rank):
            completed_epoch = step_in_epoch >= config.steps_per_epoch
            next_epoch = epoch + 1 if completed_epoch else epoch
            next_step = 0 if step_in_epoch >= config.steps_per_epoch else step_in_epoch
            save_checkpoint(
                config, model, next_epoch, next_step, global_step,
                Path(config.save_dir) / "last.pt",
            )
        dist.barrier()

        if stop_training or global_step >= config.total_steps:
            break

    if wandb_run is not None:
        wandb_run.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
