"""Continue spectra-v2 for ten epochs while preserving the original run."""

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent
SOURCE_CHECKPOINT = ROOT / "checkpoints/SpectraV2_DESI_GlobalLocalLeJEPA_MeanStd_CorrectedEpochs/complete.pt"
SAVE_DIR = ROOT / "checkpoints/SpectraV2_DESI_GlobalLocalLeJEPA_CPT10_FromEpoch10"


def load_training_module():
    spec = importlib.util.spec_from_file_location(
        "astrojepa_train_spectra_v2_cpt_base", ROOT / "train-spectra-v2.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load train-spectra-v2.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_training_module()


@dataclass
class CPTSpectraConfig(base.SpectraTrainConfig):
    epochs: int = 10
    lr: float = 5e-5
    min_lr: float = 1e-6
    warmup_steps: int = 500
    run_name: str = "SpectraV2_DESI_GlobalLocalLeJEPA_CPT10_FromEpoch10"
    save_dir: str = str(SAVE_DIR)
    resume_path: str | None = str(SOURCE_CHECKPOINT)
    wandb_run_id: str | None = "spectrav2cpt10v1"
    wandb_resume: str = "allow"
    keep_last_periodic_checkpoints: int = 1
    cpt_epoch_offset: int = 10
    source_global_step: int = 172_390
    source_completed_epochs: int = 10
    checkpoint_retention_epochs: int = 3


class OffsetSpectraDataset(base.DesiSpectraDataset):
    def set_epoch(self, epoch):
        super().set_epoch(epoch + 10)


def load_source_weights(path, model, optimizer=None, scheduler=None, scaler=None, device="cuda"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    target = model.module if isinstance(model, base.DDP) else model
    target.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    return 0, 0, 0


original_save = base.save_checkpoint


def retained_save(
    cfg, model, optimizer, scheduler, scaler, epoch, step_in_epoch, global_step, path
):
    requested = Path(path)
    save_dir = Path(cfg.save_dir)
    if requested.name.startswith("step_"):
        destination = save_dir / "recovery.pt"
    elif requested.name.startswith("last_epoch_"):
        destination = save_dir / f"epoch_{epoch:02d}.pt"
    elif requested.name == "complete.pt":
        final_epoch = save_dir / f"epoch_{epoch:02d}.pt"
        if not final_epoch.exists():
            original_save(
                cfg, model, optimizer, scheduler, scaler, epoch, step_in_epoch,
                global_step, final_epoch,
            )
        requested.unlink(missing_ok=True)
        requested.symlink_to(final_epoch.name)
        (save_dir / "recovery.pt").unlink(missing_ok=True)
        return
    else:
        raise ValueError(f"Unexpected checkpoint destination: {requested}")

    original_save(
        cfg, model, optimizer, scheduler, scaler, epoch, step_in_epoch,
        global_step, destination,
    )
    if destination.name.startswith("epoch_"):
        epochs = sorted(save_dir.glob("epoch_*.pt"))
        for stale in epochs[:-cfg.checkpoint_retention_epochs]:
            stale.unlink()


if __name__ == "__main__":
    if not SOURCE_CHECKPOINT.exists():
        raise FileNotFoundError(SOURCE_CHECKPOINT)
    base.SpectraTrainConfig = CPTSpectraConfig
    base.DesiSpectraDataset = OffsetSpectraDataset
    base.load_checkpoint = load_source_weights
    base.save_checkpoint = retained_save
    base.main()
