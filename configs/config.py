from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class TrainConfig:
    #sigreg
    sigreg_num_points: int = 17
    sigreg_num_slices: int = 1024 #picked from https://github.com/galilai-group/lejepa/blob/main/README.md


    #model
    model_name: str = "vit_small_patch14_dinov2.lvd142m"
    pretrained_backbone: bool = False
    
    # data
    dataset_name: str = "Smith42/galaxies"
    columns: list[str] = field(default_factory=lambda: ["image_crop"])
    bs: int = 96  #per device
    # eval_bs: int = 256
    num_workers: int = 1

    # model
    # embed_dim: int = 1024 Model dependent
    proj_dim: int = 128
    Vg: int = 2
    Vl: int = 8

    # loss / optim
    lambd: float = 0.05  #picked from https://github.com/galilai-group/lejepa/blob/main/README.md
    lr: float = 5e-4
    wd: float = 5e-2
    min_lr: float = 1e-6
    warmup_steps: int = 1000
    grad_accum_steps: int = 1

    # runtime
    epochs: int = 1
    total_steps: int = 100000000   #100 M total stop condition, that's more than 12 epochs
    train_num_images: int = 80000000  #80M images

    amp_dtype: str = "bf16"    #bf16 or fp16 - picked from https://github.com/galilai-group/lejepa/blob/main/README.md

    # logging / checkpointing
    entity: str = "pranavktrpl-personal"
    project: str = "astrojepa"
    run_name: str = "vit_small_patch14_FirstTrain_2104"
    log_every: int = 10
    ckpt_every: int = 10000  #Train size = 8Mill, 96 bs, 4 GPUs, 22k steps per epoch, save every 10k steps, ~2 ckpts per epoch
    save_dir: str = "./checkpoints/FirstTrain_VitSmallPatch14_2104"
    resume_path: str | None = None