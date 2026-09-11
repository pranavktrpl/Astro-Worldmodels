import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.SpectraTransforms import AstroSpectraV2Transform
import data.dataloaders as dataloaders_module
from data.dataloaders import DesiSpectraDataset, spectra_split_for_object_id


def load_training_module():
    path = REPO_ROOT / "train-spectra-v2.py"
    spec = importlib.util.spec_from_file_location("train_spectra_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_spectrum(num_pixels=41):
    flux = torch.linspace(-2.0, 3.0, num_pixels)
    ivar = torch.full((num_pixels,), 4.0)
    mask = torch.zeros(num_pixels, dtype=torch.bool)
    mask[5] = True
    ivar[17] = 0.0
    return {
        "flux": flux,
        "ivar": ivar,
        "lambda": torch.linspace(3600.0, 9800.0, num_pixels),
        "lsf_sigma": torch.full((num_pixels,), 0.87),
        "mask": mask,
    }


def test_object_split_is_deterministic_and_disjoint():
    assignments = [spectra_split_for_object_id(str(i), seed=42) for i in range(10_000)]
    assert assignments == [spectra_split_for_object_id(str(i), seed=42) for i in range(10_000)]
    assert set(assignments) == {"train", "validation", "test"}
    assert 9_600 < assignments.count("train") < 9_950


def test_transform_normalizes_valid_pixels_and_separates_masks():
    torch.manual_seed(7)
    spectrum = synthetic_spectrum()
    transform = AstroSpectraV2Transform(
        patch_size=10,
        expected_num_pixels=41,
        mask_ratio=0.25,
        mask_span=(1, 2),
        min_valid_patch_fraction=0.50,
        noise_scale=0.0,
        wavelength_check_every=1,
    )
    output = transform.augment_spectrum(spectrum)

    valid = torch.isfinite(spectrum["flux"]) & (spectrum["ivar"] > 0) & ~spectrum["mask"]
    expected_mean = spectrum["flux"][valid].mean()
    expected_std = spectrum["flux"][valid].std(correction=0)

    assert output["views"].shape == (2, 4, 10)
    assert output["valid_pixels"].shape == (4, 10)
    assert output["jepa_masks"].shape == (2, 4)
    assert torch.isclose(output["normalization_mean"], expected_mean)
    assert torch.isclose(output["normalization_log_std"].exp(), expected_std)
    assert not torch.any(output["jepa_masks"][0] & output["jepa_masks"][1])
    assert output["views"][:, 0, 5].eq(0).all()
    assert output["views"][:, 1, 7].eq(0).all()


def test_encoder_returns_global_and_position_aligned_local_outputs():
    module = load_training_module()
    model = module.SpectrumTransformerEncoderV2(
        proj_dim=16,
        patch_size=10,
        num_patches=4,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        local_proj_dim=8,
    )
    model.eval()

    views = torch.randn(3, 2, 4, 10)
    valid_pixels = torch.ones(3, 4, 10, dtype=torch.bool)
    valid_pixels[0, 1, :6] = False
    jepa_masks = torch.zeros(3, 2, 4, dtype=torch.bool)
    jepa_masks[:, 0, 0] = True
    jepa_masks[:, 1, 2] = True

    global_emb, global_proj, patch_emb, patch_proj = model(
        views,
        valid_pixels,
        jepa_masks,
    )
    assert global_emb.shape == (2, 3, 32)
    assert global_proj.shape == (2, 3, 16)
    assert patch_emb.shape == (2, 3, 4, 32)
    assert patch_proj.shape == (2, 3, 4, 8)
    assert all(torch.isfinite(value).all() for value in (global_emb, global_proj, patch_emb, patch_proj))


class QuadraticSigReg(torch.nn.Module):
    def forward(self, values):
        return values.square().mean()


def test_global_and_local_losses_both_backpropagate():
    module = load_training_module()
    global_proj = torch.randn(2, 6, 16, requires_grad=True)
    patch_proj = torch.randn(2, 6, 12, 8, requires_grad=True)
    valid_target_patches = torch.ones(6, 12, dtype=torch.bool)
    jepa_masks = torch.zeros(6, 2, 12, dtype=torch.bool)
    jepa_masks[:, 0, :3] = True
    jepa_masks[:, 1, 3:6] = True

    losses = module.compute_lejepa_loss(
        global_proj=global_proj,
        patch_proj=patch_proj,
        valid_target_patches=valid_target_patches,
        jepa_masks=jepa_masks,
        global_sigreg_fn=QuadraticSigReg(),
        local_sigreg_fn=QuadraticSigReg(),
        global_lambd=0.05,
        local_lambd=0.05,
        local_loss_weight=1.0,
        local_sigreg_num_positions=5,
        global_step=11,
    )
    losses["loss"].backward()

    assert all(torch.isfinite(value) for value in losses.values())
    assert global_proj.grad is not None and global_proj.grad.abs().sum() > 0
    assert patch_proj.grad is not None and patch_proj.grad.abs().sum() > 0
    assert losses["local_sigreg_positions"].item() == 5


def test_actual_sigreg_accepts_position_stratified_tokens():
    module = load_training_module()
    global_proj = torch.randn(2, 12, 8, requires_grad=True)
    patch_proj = torch.randn(2, 12, 6, 8, requires_grad=True)
    valid_target_patches = torch.ones(12, 6, dtype=torch.bool)
    jepa_masks = torch.zeros(12, 2, 6, dtype=torch.bool)

    losses = module.compute_lejepa_loss(
        global_proj=global_proj,
        patch_proj=patch_proj,
        valid_target_patches=valid_target_patches,
        jepa_masks=jepa_masks,
        global_sigreg_fn=module.build_sigreg(17, 8, torch.device("cpu")),
        local_sigreg_fn=module.build_sigreg(17, 8, torch.device("cpu")),
        global_lambd=0.05,
        local_lambd=0.05,
        local_loss_weight=1.0,
        local_sigreg_num_positions=3,
        global_step=3,
    )
    losses["loss"].backward()

    assert all(torch.isfinite(value) for value in losses.values())
    assert global_proj.grad is not None
    assert patch_proj.grad is not None


def test_periodic_checkpoint_retention_preserves_milestones():
    module = load_training_module()
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_dir = Path(directory)
        for step in (4_000, 8_000, 12_000, 16_000):
            (checkpoint_dir / f"step_{step}.pt").touch()
        (checkpoint_dir / "last_epoch_0.pt").touch()
        (checkpoint_dir / "complete.pt").touch()

        removed = module.prune_periodic_checkpoints(checkpoint_dir, keep_last=2)

        assert [path.name for path in removed] == ["step_4000.pt", "step_8000.pt"]
        assert sorted(path.name for path in checkpoint_dir.glob("step_*.pt")) == [
            "step_12000.pt",
            "step_16000.pt",
        ]
        assert (checkpoint_dir / "last_epoch_0.pt").is_file()
        assert (checkpoint_dir / "complete.pt").is_file()


def test_spectra_stream_is_partitioned_once_by_rank():
    class EmptyStream:
        def __init__(self):
            self.shard_call = None

        def filter(self, _predicate):
            return self

        def set_epoch(self, _epoch):
            return None

        def shard(self, num_shards, index):
            self.shard_call = (num_shards, index)
            return self

        def __iter__(self):
            return iter(())

    class FakeSource:
        def __init__(self, *_args, **_kwargs):
            self.requested_num_shards = None
            self.loaded_data_files = None
            self.stream = None

        def balanced_file_shards(self, num_shards):
            self.requested_num_shards = num_shards
            return [[f"rank-{rank}.parquet"] for rank in range(num_shards)]

        def load_dataset(self, data_files=None):
            self.loaded_data_files = data_files
            self.stream = EmptyStream()
            return self.stream

    original_source = dataloaders_module.DesiSpectraSource
    original_get_worker_info = torch.utils.data.get_worker_info
    torch.utils.data.get_worker_info = lambda: SimpleNamespace(num_workers=4, id=3)

    try:
        dataloaders_module.DesiSpectraSource = FakeSource
        dataset = dataloaders_module.DesiSpectraDataset(
            world_size=4,
            rank=2,
            loader_num_workers=4,
            shuffle=False,
        )
        assert dataset.ds.requested_num_shards == 4
        assert next(iter(dataset), None) is None
        assert dataset.ds.loaded_data_files == ["rank-2.parquet"]
        assert dataset.ds.stream.shard_call is None
    finally:
        dataloaders_module.DesiSpectraSource = original_source
        torch.utils.data.get_worker_info = original_get_worker_info


def test_local_desi_stream_produces_v2_sample():
    dataset_path = Path(
        "/mnt/datasets/utbd_pranav/desi_edr_sv3/mmu_desi_edr_sv3/dataset"
    )
    assert dataset_path.is_dir()
    dataset = DesiSpectraDataset(
        split="train",
        dataset=str(dataset_path),
        columns=["spectrum", "object_id"],
        shuffle=False,
        world_size=1,
        rank=0,
        noise_scale=0.0,
    )
    sample = next(iter(dataset))

    assert spectra_split_for_object_id(sample["object_id"], 42) == "train"
    assert sample["views"].shape == (2, 389, 20)
    assert sample["valid_pixels"].shape == (389, 20)
    assert sample["jepa_masks"].shape == (2, 389)
    assert not torch.any(sample["jepa_masks"][0] & sample["jepa_masks"][1])


if __name__ == "__main__":
    tests = [
        test_object_split_is_deterministic_and_disjoint,
        test_transform_normalizes_valid_pixels_and_separates_masks,
        test_spectra_stream_is_partitioned_once_by_rank,
        test_encoder_returns_global_and_position_aligned_local_outputs,
        test_global_and_local_losses_both_backpropagate,
        test_actual_sigreg_accepts_position_stratified_tokens,
        test_periodic_checkpoint_retention_preserves_milestones,
        test_local_desi_stream_produces_v2_sample,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
