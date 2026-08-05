import os
from dataclasses import dataclass, field


@dataclass
class ContinueAstroclipConfig:
    """Continued pretraining (domain adaptation) on the AstroCLIP cross-match
    train split (~139k images). Purpose: measure how much of the cross-match
    eval gap (0.53 vs AstroCLIP's 0.79 test R2) is faint-population domain
    coverage — see Evals/desi_crossmatch/README.md.

    model_name / proj_dim are NOT set here: they are read from init_from's
    saved cfg so the architecture always matches the checkpoint.
    """

    # which checkpoint to continue from (start with ViT-S: the cheap answer)
    init_from: str = "./checkpoints/VitSmallPatch14_2204/step_21000.pt"

    # filled at runtime from init_from's saved cfg — declared as fields so
    # asdict(cfg) embeds them in saved checkpoints (eval probes read
    # cfg["model_name"] when loading a checkpoint)
    model_name: str | None = None
    proj_dim: int | None = None

    # data
    data_dir: str = "./Evals/desi_crossmatch/data/astroclip"
    bs: int = 96
    num_workers: int = 2
    Vg: int = 2
    Vl: int = 8

    # sigreg (same as original pretraining)
    sigreg_num_points: int = 17
    sigreg_num_slices: int = 1024
    lambd: float = 0.05

    # optim — reduced peak lr vs from-scratch (5e-4): this is adaptation, not
    # pretraining; a high lr would wipe the representation before it adapts.
    lr: float = 5e-5
    wd: float = 5e-2
    min_lr: float = 1e-6
    warmup_steps: int = 200

    # runtime — ~139k images; with bs 96 x 4 GPUs one epoch is ~360 steps.
    epochs: int = 10
    steps_per_epoch: int | None = None  # computed from parquet row counts
    total_steps: int | None = None

    amp_dtype: str = "bf16"
    pretrained_backbone: bool = False  # weights come from init_from

    # logging / checkpointing — entity/project come from .env (see
    # .env.example); an unset entity logs to the API key's default entity
    entity: str | None = field(
        default_factory=lambda: os.environ.get("WANDB_ENTITY") or None
    )
    project: str = field(
        default_factory=lambda: os.environ.get("WANDB_PROJECT", "astrojepa")
    )
    run_name: str = "ContinuePretrain_AstroclipXmatch"
    wandb_run_id: str | None = None
    wandb_resume: str = "allow"
    log_every: int = 10
    ckpt_every: int = 500
    save_dir: str = "./checkpoints/ContinuePretrain_AstroclipXmatch"
