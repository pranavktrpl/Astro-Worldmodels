import importlib.util
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.SpectraTransformsOverlap import AstroSpectraV2OverlapTransform
from data.dataloaders_overlap import DesiSpectraOverlapDataset


def load_training_module():
    path = REPO_ROOT / "train-spectra-v2-overlap.py"
    spec = importlib.util.spec_from_file_location("train_spectra_v2_overlap", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_spectrum(num_pixels=41):
    flux = torch.linspace(-2.0, 3.0, num_pixels)
    return {
        "flux": flux,
        "ivar": torch.full((num_pixels,), 4.0),
        "lambda": torch.linspace(3600.0, 9800.0, num_pixels),
        "lsf_sigma": torch.full((num_pixels,), 0.87),
        "mask": torch.zeros(num_pixels, dtype=torch.bool),
    }


def test_transform_builds_half_overlapping_patches():
    spectrum = synthetic_spectrum()
    transform = AstroSpectraV2OverlapTransform(
        patch_size=10,
        patch_stride=5,
        expected_num_pixels=41,
        mask_ratio=0.0,
        noise_scale=0.0,
    )
    output = transform.augment_spectrum(spectrum)

    assert transform.num_patches == 7
    assert transform.covered_num_pixels == 40
    assert output["views"].shape == (2, 7, 10)
    assert torch.equal(output["views"][0, 0, 5:], output["views"][0, 1, :5])
    assert torch.equal(
        output["valid_pixels"][0, 5:],
        output["valid_pixels"][1, :5],
    )


def test_overlap_config_is_isolated_and_consistent():
    module = load_training_module()
    config = module.SpectraTrainConfig()

    assert config.spectra_patch_size == 20
    assert config.spectra_patch_stride == 10
    assert config.spectra_num_patches == 777
    assert "OverlapStride10" in config.run_name
    assert "OverlapStride10" in config.save_dir


def test_overlap_encoder_and_unchanged_losses_backpropagate():
    module = load_training_module()
    model = module.SpectrumTransformerEncoderV2(
        proj_dim=16,
        patch_size=10,
        num_patches=7,
        embed_dim=32,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        local_proj_dim=8,
    )
    views = torch.randn(3, 2, 7, 10)
    valid_pixels = torch.ones(3, 7, 10, dtype=torch.bool)
    jepa_masks = torch.zeros(3, 2, 7, dtype=torch.bool)
    jepa_masks[:, 0, :2] = True
    jepa_masks[:, 1, 2:4] = True
    _, global_proj, _, patch_proj = model(views, valid_pixels, jepa_masks)

    class QuadraticSigReg(torch.nn.Module):
        def forward(self, values):
            return values.square().mean()

    losses = module.compute_lejepa_loss(
        global_proj=global_proj,
        patch_proj=patch_proj,
        valid_target_patches=torch.ones(3, 7, dtype=torch.bool),
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
    assert model.patch_embed.weight.grad is not None
    assert model.patch_embed.weight.grad.abs().sum() > 0


def test_local_desi_stream_produces_overlap_sample():
    dataset_path = Path(
        "/mnt/datasets/utbd_pranav/desi_edr_sv3/mmu_desi_edr_sv3/dataset"
    )
    assert dataset_path.is_dir()
    dataset = DesiSpectraOverlapDataset(
        split="train",
        dataset=str(dataset_path),
        columns=["spectrum", "object_id"],
        shuffle=False,
        world_size=1,
        rank=0,
        patch_size=20,
        patch_stride=10,
        noise_scale=0.0,
    )
    sample = next(iter(dataset))

    assert sample["views"].shape == (2, 777, 20)
    assert sample["valid_pixels"].shape == (777, 20)
    assert sample["jepa_masks"].shape == (2, 777)
    assert not torch.any(sample["jepa_masks"][0] & sample["jepa_masks"][1])


if __name__ == "__main__":
    tests = [
        test_transform_builds_half_overlapping_patches,
        test_overlap_config_is_isolated_and_consistent,
        test_overlap_encoder_and_unchanged_losses_backpropagate,
        test_local_desi_stream_produces_overlap_sample,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
