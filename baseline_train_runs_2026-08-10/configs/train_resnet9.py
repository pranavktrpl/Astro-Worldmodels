from dataclasses import dataclass, field
from pathlib import Path

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
    lambd: float = 0.05  #picked from https://github.com/galilai-group/lejepa/blob/main/README.md
    lr: float = 5e-4
    wd: float = 5e-4
    min_lr: float = 1e-6
    warmup_steps: int = 1000

    # runtime
    epochs: int = 4
    total_steps: int = 100000000   #100 M total stop condition, that's more than 12 epochs
    
    amp_dtype: str = "bf16"    #bf16 or fp16

    # logging / checkpointing
    entity: str = "pranavktrpl-personal"
    project: str = "astrojepa"
    run_name: str = "resnet9_lejepa_smokeTest2_2004"
    log_every: int = 10
    ckpt_every: int = 8000  #Train size = 8Mill, 128 bs, 4 GPUs, 16k steps per epoch, save every 8k steps, 2 ckpts per epoch
    save_dir: str = "./checkpoints/smokeTest2_Resnet9_2004"
    resume_path: str | None = None



#         @dataclass
# class CFG:
#     dataset_name: str = "Smith42/galaxies"
#     columns: list = None
    
#     bs: int = 128
#     eval_bs: int = 256
#     num_workers: int = 1

#     embed_dim: int = 1024
#     proj_dim: int = 128

#     Vg: int = 2
#     Vl: int = 0

#     lambd: float = 0.05                    #picked from https://github.com/galilai-group/lejepa/blob/main/README.md
#     lr: float = 5e-4
#     wd: float = 5e-4

#     epochs: int = 4
#     warmup_steps: int = 1000
#     total_steps: int = 100000000  #100 M total stop condition, that's more than 12 epochs
#     min_lr: float = 1e-6

#     amp_dtype: str = "bf16"  #bf16 or fp16

#     entity: str = "pranavktrpl-personal"
#     project: str = "astrojepa"
#     run_name: str = "resnet9_lejepa_smoke_0504"
#     log_every: int = 10
#     ckpt_every: int = 8000    #Train size = 8Mill, 128 bs, 4 GPUs, 16k steps per epoch, save every 8k steps, 2 ckpts per epoch
#     save_dir: str = "./checkpoints"
#     resume_path: str | None = None

