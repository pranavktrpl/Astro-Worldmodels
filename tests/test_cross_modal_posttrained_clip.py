import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data.cross_modal_clip import AstroCLIPAlignmentImageTransform
from models.cross_modal_posttrained_clip import AstroCLIPAttentionPooler


def test_pooler_and_transform():
    transform = AstroCLIPAlignmentImageTransform(size=140)
    output = transform.augment_image(torch.rand(3, 152, 152))
    assert output["global_crops"].shape == (2, 3, 140, 140)
    assert torch.equal(output["global_crops"][0], output["global_crops"][1])

    image_pooler = AstroCLIPAttentionPooler(64, 32, num_heads=4, residual_mlp=False)
    spectrum_pooler = AstroCLIPAttentionPooler(48, 32, num_heads=4, residual_mlp=True)
    assert image_pooler(torch.randn(5, 11, 64)).shape == (5, 32)
    mask = torch.zeros(5, 9, dtype=torch.bool)
    assert spectrum_pooler(torch.randn(5, 9, 48), mask).shape == (5, 32)


def test_distributed_loss():
    from importlib.util import module_from_spec, spec_from_file_location

    path = REPO_ROOT / "train-cross-modal-posttrained-clip.py"
    spec = spec_from_file_location("clip_trainer_test", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    rank = dist.get_rank()
    torch.manual_seed(100 + rank)
    image = torch.randn(4, 32, device=rank, requires_grad=True)
    spectrum = torch.randn(4, 32, device=rank, requires_grad=True)
    losses = module.compute_distributed_astroclip_loss(image, spectrum, 15.5)
    assert torch.isfinite(losses["loss"])
    assert losses["image_to_spectrum_top1"].ndim == 0
    losses["loss"].backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert spectrum.grad is not None and torch.isfinite(spectrum.grad).all()


if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    test_pooler_and_transform()
    test_distributed_loss()
    if dist.get_rank() == 0:
        print("PASS cross-modal AstroCLIP adapter and distributed loss tests")
    dist.destroy_process_group()
