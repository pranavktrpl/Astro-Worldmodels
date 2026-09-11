import math

import torch

from models.cross_modal import CrossModalScratchModel


class CrossModalScratchCLIPModel(CrossModalScratchModel):
    """The scratch cross-modal architecture with CLIP's learned temperature."""

    def __init__(self, *args, initial_temperature=0.07, **kwargs):
        super().__init__(*args, **kwargs)
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        self.logit_scale = torch.nn.Parameter(
            torch.tensor(math.log(1.0 / initial_temperature))
        )

    def bounded_logit_scale(self, maximum=100.0):
        return self.logit_scale.exp().clamp(max=maximum)

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        outputs["logit_scale"] = self.logit_scale.exp()
        return outputs
