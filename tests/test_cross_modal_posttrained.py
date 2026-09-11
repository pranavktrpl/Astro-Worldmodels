from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.cross_modal_posttrained import CrossModalPosttrainedModel
from models.cross_modal_posttrained import QueryAttentionPooler


def build_tiny_model():
    return CrossModalPosttrainedModel(
        image_model_name="vit_tiny_patch16_224",
        shared_dim=8,
        spectra_patch_size=4,
        spectra_num_patches=6,
        spectra_embed_dim=16,
        spectra_depth=1,
        spectra_num_heads=4,
        spectra_mlp_ratio=2.0,
        pool_num_heads=4,
        pool_dropout=0.0,
    )


def test_query_pooler_ignores_padded_tokens():
    torch.manual_seed(3)
    pooler = QueryAttentionPooler(
        input_dim=12,
        shared_dim=8,
        num_heads=4,
        dropout=0.0,
    ).eval()
    tokens = torch.randn(2, 5, 12)
    key_padding_mask = torch.zeros(2, 5, dtype=torch.bool)
    key_padding_mask[:, -1] = True

    first = pooler(tokens, key_padding_mask=key_padding_mask)
    tokens[:, -1] = 1.0e6
    second = pooler(tokens, key_padding_mask=key_padding_mask)

    assert torch.allclose(first, second, atol=1.0e-5, rtol=1.0e-5)


def test_posttrained_model_only_backpropagates_through_poolers():
    torch.manual_seed(5)
    model = build_tiny_model().train()
    image_views = torch.randn(2, 2, 3, 32, 32)
    spectrum_views = torch.randn(2, 2, 6, 4)
    valid_pixels = torch.ones(2, 6, 4, dtype=torch.bool)
    valid_pixels[:, -1] = False
    masks = torch.zeros(2, 2, 6, dtype=torch.bool)
    masks[:, 0, 1:3] = True
    masks[:, 1, 3:5] = True

    outputs = model(image_views, spectrum_views, valid_pixels, masks)
    assert outputs["image_embeddings"].shape == (2, 2, 192)
    assert outputs["spectrum_embeddings"].shape == (2, 2, 16)
    assert outputs["image_projections"].shape == (2, 2, 8)
    assert outputs["spectrum_projections"].shape == (2, 2, 8)

    loss = (
        outputs["image_projections"].square().mean()
        + outputs["spectrum_projections"].square().mean()
    )
    loss.backward()

    assert all(
        parameter.grad is None
        for parameter in model.image_encoder.backbone.parameters()
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.spectrum_encoder.named_parameters()
        if not name.startswith("shared_proj.")
    )
    assert any(
        parameter.grad is not None
        for parameter in model.image_encoder.shared_proj.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.spectrum_encoder.shared_proj.parameters()
    )


def test_strict_source_loading_and_head_only_state(tmp_path):
    torch.manual_seed(7)
    source = build_tiny_model()
    image_path = tmp_path / "image.pt"
    spectrum_path = tmp_path / "spectrum.pt"

    torch.save(
        {
            "model": {
                f"backbone.{name}": value
                for name, value in source.image_encoder.backbone.state_dict().items()
            },
            "cfg": {"model_name": "vit_tiny_patch16_224"},
            "global_step": 12,
            "epoch": 2,
        },
        image_path,
    )
    torch.save(
        {
            "model": {
                name: value
                for name, value in source.spectrum_encoder.state_dict().items()
                if not name.startswith("shared_proj.")
            },
            "cfg": {
                "spectra_patch_size": 4,
                "spectra_num_patches": 6,
                "spectra_embed_dim": 16,
                "spectra_pooling": "cls",
            },
            "global_step": 34,
            "epoch": 4,
        },
        spectrum_path,
    )

    target = build_tiny_model()
    metadata = target.load_pretrained_backbones(image_path, spectrum_path)

    assert metadata["image"]["global_step"] == 12
    assert metadata["spectrum"]["global_step"] == 34
    assert torch.equal(
        target.image_encoder.backbone.cls_token,
        source.image_encoder.backbone.cls_token,
    )
    assert torch.equal(
        target.spectrum_encoder.cls_token,
        source.spectrum_encoder.cls_token,
    )
    alignment_state = target.alignment_state_dict()
    assert set(alignment_state) == {
        "image_attention_pooler",
        "spectrum_attention_pooler",
    }
    assert not any(
        "backbone" in name
        for state in alignment_state.values()
        for name in state
    )
