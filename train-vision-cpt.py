"""Continue the selected image backbone for ten local-data epochs.

This isolated entry point loads model weights only, uses a fresh conservative
schedule, and retains only the final three epoch checkpoints.
"""

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent
SOURCE_CHECKPOINT = ROOT / "checkpoints/VitLargePatch14_OfficialTrain5_Epoch5_2504/step_52000.pt"
SAVE_DIR = ROOT / "checkpoints/ViTL14_LeJEPA_CPT10_LocalMMU_FromStep52000"


def load_training_module():
    spec = importlib.util.spec_from_file_location(
        "astrojepa_train_vision_cpt_base", ROOT / "train-vision.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load train-vision.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_training_module()


@dataclass
class CPTTrainConfig(base.TrainConfig):
    dataset_name: str = "/mnt/datasets/utbd_pranav/galaxies/with_crops"
    epochs: int = 10
    lr: float = 5e-5
    min_lr: float = 1e-6
    warmup_steps: int = 500
    ckpt_every: int = 4000
    run_name: str = "ViTL14_LeJEPA_CPT10_LocalMMU_FromStep52000"
    save_dir: str = str(SAVE_DIR)
    resume_path: str | None = str(SOURCE_CHECKPOINT)
    wandb_run_id: str | None = "vitl14cpt10localv1"
    wandb_resume: str = "allow"
    cpt_epoch_offset: int = 5
    source_global_step: int = 52_000
    source_effective_epochs: float = 4.73
    checkpoint_retention_epochs: int = 3


class OffsetImageDataset(base.MyDataset):
    def set_epoch(self, epoch):
        super().set_epoch(epoch + 5)


def load_source_weights(path, model, optimizer=None, scheduler=None, scaler=None, device="cuda"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    target = model.module if isinstance(model, base.DDP) else model
    target.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    return 0, 0


original_save = base.save_checkpoint


def atomic_full_save(cfg, model, optimizer, scheduler, scaler, epoch, global_step, path):
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    original_save(
        cfg, model, optimizer, scheduler, scaler, epoch, global_step, str(temporary)
    )
    os.replace(temporary, destination)


def retained_save(cfg, model, optimizer, scheduler, scaler, epoch, global_step, path):
    requested = Path(path)
    save_dir = Path(cfg.save_dir)
    if requested.name.startswith("step_"):
        atomic_full_save(
            cfg, model, optimizer, scheduler, scaler, epoch, global_step,
            save_dir / "recovery.pt",
        )
        return
    if requested.name.startswith("last_epoch_"):
        destination = save_dir / f"epoch_{epoch:02d}.pt"
        atomic_full_save(
            cfg, model, optimizer, scheduler, scaler, epoch, global_step, destination
        )
        epochs = sorted(save_dir.glob("epoch_*.pt"))
        for stale in epochs[:-cfg.checkpoint_retention_epochs]:
            stale.unlink()
        return
    if requested.name == "complete.pt":
        final_epoch = save_dir / f"epoch_{epoch:02d}.pt"
        if not final_epoch.exists():
            atomic_full_save(
                cfg, model, optimizer, scheduler, scaler, epoch, global_step,
                final_epoch,
            )
        requested.unlink(missing_ok=True)
        requested.symlink_to(final_epoch.name)
        (save_dir / "recovery.pt").unlink(missing_ok=True)
        return
    raise ValueError(f"Unexpected checkpoint destination: {requested}")


if __name__ == "__main__":
    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(SOURCE_CHECKPOINT)
    base.TrainConfig = CPTTrainConfig
    base.MyDataset = OffsetImageDataset
    base.load_checkpoint = load_source_weights
    base.save_checkpoint = retained_save
    base.main()
