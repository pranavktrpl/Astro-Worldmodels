from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class TrainConfig:
    #sigreg
    sigreg_num_points: int = 17
    sigreg_num_slices: int = 1024 #picked from https://github.com/galilai-group/lejepa/blob/main/README.md


    #model
    model_name: str = "vit_large_patch14_dinov2.lvd142m"
    pretrained_backbone: bool = False
    
    # data
    dataset_name: str = "Smith42/galaxies"
    columns: list[str] = field(default_factory=lambda: ["image_crop"])
    bs: int = 192  #per device 64 bs ==> 30gigs/80gigs occupied in each gpu!, maybe double it? or 2.5 times
    # eval_bs: int = 256
    num_workers: int = 1

    # model
    # embed_dim: int = 1024 Model dependent
    proj_dim: int = 64  #picked from papers ablations.
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
    epochs: int = 5
    safe_samples_per_rank_per_epoch: int = 21976 * 96
    steps_per_epoch: int | None = None
    total_steps: int | None = None   #100 M total stop condition, that's more than 12 epochs
    train_num_images: int = 80000000  #80M images

    amp_dtype: str = "bf16"    #bf16 or fp16 - picked from https://github.com/galilai-group/lejepa/blob/main/README.md

    # logging / checkpointing
    entity: str = "pranavktrpl-personal"
    project: str = "astrojepa"
    run_name: str = "VitLargePatch14_OfficialTrain5_bs192_Epoch5"
    log_every: int = 10
    ckpt_every: int = 4000  #Train size = 8Mill, 96 bs, 4 GPUs, 22k steps per epoch, save every 10k steps, ~2 ckpts per epoch
    save_dir: str = "./checkpoints/VitLargePatch14_OfficialTrain5_Epoch5_2504"
    resume_path: str | None = "./checkpoints/VitLargePatch14_OfficialTrain5_Epoch5_2504/step_12000.pt"
    wandb_run_id: str | None = "najxg1zj"
    wandb_resume: str = "must"   # or "must" if you want it to fail unless the run exists