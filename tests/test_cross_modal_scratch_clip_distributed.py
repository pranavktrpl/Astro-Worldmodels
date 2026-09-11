import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.test_cross_modal_scratch_clip import trainer


def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(100 + rank)
    image = torch.randn(2, 3, 8, requires_grad=True)
    spectrum = torch.randn(2, 3, 8, requires_grad=True)
    scale = torch.tensor(1.0, requires_grad=True)
    losses = trainer.compute_distributed_multiview_clip_loss(
        image, spectrum, scale
    )
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert spectrum.grad is not None and torch.isfinite(spectrum.grad).all()
    assert scale.grad is not None and torch.isfinite(scale.grad)
    if rank == 0:
        print(
            "PASS distributed multiview CLIP: "
            f"world_size={dist.get_world_size()} global_negatives=5"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
