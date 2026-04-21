from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # data
    dataset_name: str = "Smith42/galaxies"
    columns: list[str] = field(default_factory=lambda: ["image_crop"])
    bs: int = 96
    eval_bs: int = 256
    num_workers: int = 1

    # model
    embed_dim: int = 1024
    proj_dim: int = 128
    Vg: int = 2
    Vl: int = 8

    # loss / optim
    lambd: float = 0.05
    lr: float = 5e-4
    wd: float = 5e-4
    min_lr: float = 1e-6
    warmup_steps: int = 1000

    # runtime
    epochs: int = 1
    total_steps: int = 100000000
    amp_dtype: str = "bf16"

    # logging / checkpointing
    entity: str = "pranavktrpl-personal"
    project: str = "astrojepa"
    run_name: str = "resnet9_lejepa_profile_2004"
    log_every: int = 1_000_000
    ckpt_every: int = 1_000_000
    save_dir: str = "./checkpoints/profile_resnet9_2004"
    resume_path: str | None = None

    # profiling
    profile_enabled: bool = True
    profile_warmup_steps: int = 10
    profile_steps: int = 30
    profile_disable_wandb: bool = True
    profile_disable_ckpt: bool = True
