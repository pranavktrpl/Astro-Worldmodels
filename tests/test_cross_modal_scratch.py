import io
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.cross_modal import (
    COMBINED_DESI_FORMAT,
    CombinedPairedImageSpectrumDataset,
    angular_separation_arcsec,
    astroclip_raw_to_rgb,
    cross_modal_split_for_dr8_id,
    cross_modal_split_for_identity,
    decode_image,
)
from data.SpectraTransforms import AstroSpectraV2Transform
from models.cross_modal import (
    CrossModalScratchModel,
    CrossModalSpectrumEncoder,
    compute_cross_modal_lejepa_loss,
)


class ConstantSigreg(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, samples):
        return samples.sum() * 0.0 + self.value


def test_cross_modal_loss_is_exact_mean_of_four_pairings():
    image = torch.tensor(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[1.0, 2.0], [3.0, 4.0]],
        ],
        requires_grad=True,
    )
    spectrum = torch.tensor(
        [
            [[2.0, 0.0], [4.0, 2.0]],
            [[3.0, 1.0], [5.0, 3.0]],
        ],
        requires_grad=True,
    )

    losses = compute_cross_modal_lejepa_loss(
        image,
        spectrum,
        image_sigreg=ConstantSigreg(2.0),
        spectrum_sigreg=ConstantSigreg(4.0),
        lambd=0.2,
    )
    expected_invariance = torch.stack(
        [
            (image[image_view] - spectrum[spectrum_view]).square().mean()
            for image_view in range(2)
            for spectrum_view in range(2)
        ]
    ).mean()
    expected = 0.8 * expected_invariance + 0.2 * 3.0

    assert torch.allclose(losses["cross_invariance"], expected_invariance)
    assert torch.allclose(losses["sigreg"], torch.tensor(3.0))
    assert torch.allclose(losses["loss"], expected)

    losses["loss"].backward()
    assert image.grad is not None
    assert spectrum.grad is not None
    assert image.grad.abs().sum() > 0
    assert spectrum.grad.abs().sum() > 0


def test_cross_modal_split_groups_repeated_image_identity():
    first = cross_modal_split_for_dr8_id("123_456", seed=42)
    second = cross_modal_split_for_dr8_id("123_456", seed=42)
    assert first == second
    assert first in {"train", "validation", "test"}


def test_native_identity_split_uses_the_same_stable_hash():
    identity = "2196p002-11176"
    assert cross_modal_split_for_identity(identity, seed=42) == (
        cross_modal_split_for_dr8_id(identity, seed=42)
    )


def test_angular_separation_is_zero_for_identical_coordinates():
    assert angular_separation_arcsec(12.3, -4.5, 12.3, -4.5) == 0.0


def test_decode_cached_image_bytes():
    source = Image.new("RGB", (8, 8), color=(10, 20, 30))
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")

    decoded = decode_image({"bytes": buffer.getvalue(), "path": None})

    assert decoded.mode == "RGB"
    assert decoded.size == (8, 8)
    assert decoded.getpixel((0, 0)) == (10, 20, 30)


def test_spectrum_encoder_returns_two_global_views():
    encoder = CrossModalSpectrumEncoder(
        shared_dim=8,
        patch_size=4,
        num_patches=6,
        embed_dim=16,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
    )
    views = torch.randn(3, 2, 6, 4)
    valid_pixels = torch.ones(3, 6, 4, dtype=torch.bool)
    masks = torch.zeros(3, 2, 6, dtype=torch.bool)
    masks[:, 0, 1:3] = True
    masks[:, 1, 3:5] = True

    embeddings, projections = encoder(views, valid_pixels, masks)

    assert embeddings.shape == (2, 3, 16)
    assert projections.shape == (2, 3, 8)


def test_reduced_joint_model_forward():
    model = CrossModalScratchModel(
        image_model_name="vit_tiny_patch16_224",
        image_pretrained=False,
        shared_dim=8,
        spectra_patch_size=4,
        spectra_num_patches=6,
        spectra_embed_dim=16,
        spectra_depth=1,
        spectra_num_heads=4,
        spectra_mlp_ratio=2.0,
    )
    image_views = torch.randn(2, 2, 3, 32, 32)
    spectrum_views = torch.randn(2, 2, 6, 4)
    valid_pixels = torch.ones(2, 6, 4, dtype=torch.bool)
    masks = torch.zeros(2, 2, 6, dtype=torch.bool)

    outputs = model(
        image_views,
        spectrum_views,
        valid_pixels,
        masks,
    )

    assert outputs["image_projections"].shape == (2, 2, 8)
    assert outputs["spectrum_projections"].shape == (2, 2, 8)


def test_astroclip_raw_image_is_rendered_to_bounded_rgb():
    raw = np.zeros((12, 12, 3), dtype=np.float32)
    raw[5:7, 5:7, 0] = 2.0

    rgb = astroclip_raw_to_rgb(raw)

    assert rgb.shape == (12, 12, 3)
    assert rgb.dtype == np.float32
    assert np.isfinite(rgb).all()
    assert 0.0 <= float(rgb.min()) <= float(rgb.max()) <= 1.0


def test_flux_only_desi_uses_mean_std_without_invented_uncertainty():
    transform = AstroSpectraV2Transform(
        expected_num_pixels=24,
        patch_size=4,
        allow_flux_only=True,
        noise_scale=1.0,
    )
    flux = torch.linspace(-2.0, 3.0, 24).unsqueeze(-1)

    output = transform.augment_spectrum(flux)

    assert output["views"].shape == (2, 6, 4)
    assert output["valid_pixels"].all()
    assert not output["uncertainty_available"]
    assert torch.equal(output["views"][0], output["views"][1])


def test_combined_file_balancing_accounts_for_every_row():
    files = [
        {
            "path": f"/unused/{index}.parquet",
            "rows": rows,
            "dataset_format": "unused",
            "source_index": 0,
            "source_name": "unused",
        }
        for index, rows in enumerate((11, 10, 9, 8, 7, 6, 5, 4))
    ]
    info = {
        "dataset_format": COMBINED_DESI_FORMAT,
        "files": files,
    }

    dataset = CombinedPairedImageSpectrumDataset(
        info,
        world_size=2,
        rank=0,
        num_workers=2,
        spectra_num_pixels=24,
        spectra_patch_size=4,
    )

    assert sum(dataset.rows_per_rank) == sum(item["rows"] for item in files)
    assert all(
        sum(worker_rows) == rank_rows
        for worker_rows, rank_rows in zip(
            dataset.worker_rows_by_rank,
            dataset.rows_per_rank,
        )
    )


if __name__ == "__main__":
    tests = [
        test_cross_modal_loss_is_exact_mean_of_four_pairings,
        test_cross_modal_split_groups_repeated_image_identity,
        test_decode_cached_image_bytes,
        test_native_identity_split_uses_the_same_stable_hash,
        test_angular_separation_is_zero_for_identical_coordinates,
        test_astroclip_raw_image_is_rendered_to_bounded_rgb,
        test_flux_only_desi_uses_mean_std_without_invented_uncertainty,
        test_combined_file_balancing_accounts_for_every_row,
        test_spectrum_encoder_returns_two_global_views,
        test_reduced_joint_model_forward,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
