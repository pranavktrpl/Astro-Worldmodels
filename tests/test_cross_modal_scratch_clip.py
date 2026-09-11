import importlib.util
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

spec = importlib.util.spec_from_file_location(
    "scratch_clip_trainer", REPO_ROOT / "train-cross-modal-scratch-clip.py"
)
trainer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trainer
spec.loader.exec_module(trainer)

from models.cross_modal_scratch_clip import CrossModalScratchCLIPModel


def test_multiview_clip_is_exact_mean_of_four_symmetric_losses():
    image = torch.randn(2, 5, 7, requires_grad=True)
    spectrum = torch.randn(2, 5, 7, requires_grad=True)
    logit_scale = torch.tensor(3.0, requires_grad=True)

    losses = trainer.compute_distributed_multiview_clip_loss(
        image, spectrum, logit_scale
    )
    normalized_image = F.normalize(image.float(), dim=-1, eps=1.0e-3)
    normalized_spectrum = F.normalize(spectrum.float(), dim=-1, eps=1.0e-3)
    labels = torch.arange(image.shape[1])
    expected = []
    for image_view in normalized_image:
        for spectrum_view in normalized_spectrum:
            image_logits = logit_scale * image_view @ spectrum_view.T
            spectrum_logits = logit_scale * spectrum_view @ image_view.T
            expected.append(
                0.5
                * (
                    F.cross_entropy(image_logits, labels)
                    + F.cross_entropy(spectrum_logits, labels)
                )
            )

    assert torch.allclose(losses["loss"], torch.stack(expected).mean())
    losses["loss"].backward()
    assert image.grad is not None and image.grad.abs().sum() > 0
    assert spectrum.grad is not None and spectrum.grad.abs().sum() > 0
    assert logit_scale.grad is not None and logit_scale.grad.abs() > 0


def test_scratch_clip_model_has_bounded_learned_temperature():
    model = CrossModalScratchCLIPModel(
        image_model_name="vit_tiny_patch16_224",
        image_pretrained=False,
        shared_dim=8,
        spectra_patch_size=4,
        spectra_num_patches=6,
        spectra_embed_dim=16,
        spectra_depth=1,
        spectra_num_heads=4,
        spectra_mlp_ratio=2.0,
        initial_temperature=0.07,
    )
    assert model.logit_scale.requires_grad
    assert torch.allclose(
        model.bounded_logit_scale(), torch.tensor(1.0 / 0.07), atol=1.0e-5
    )
    with torch.no_grad():
        model.logit_scale.fill_(100.0)
    assert model.bounded_logit_scale().item() == 100.0
