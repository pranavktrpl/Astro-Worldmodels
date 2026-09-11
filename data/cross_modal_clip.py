import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

from data.cross_modal import CombinedPairedImageSpectrumDataset


class AstroCLIPAlignmentImageTransform:
    """One AstroCLIP-style geometric view, duplicated for the shared model API."""

    def __init__(self, size=140):
        self.to_tensor = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )
        self.transform = v2.Compose(
            [
                v2.CenterCrop(size),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.RandomRotation(
                    degrees=(0, 360),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                ),
            ]
        )

    def augment_image(self, image):
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        else:
            image = torch.as_tensor(
                np.asarray(image) if not torch.is_tensor(image) else image
            )
            if image.ndim != 3:
                raise ValueError(f"Expected a 3D image, got {tuple(image.shape)}")
            if image.shape[0] != 3 and image.shape[-1] == 3:
                image = image.permute(2, 0, 1)
        view = self.transform(self.to_tensor(image))
        return {"global_crops": torch.stack((view, view)), "local_crops": []}


class CombinedPairedImageSpectrumCLIPDataset(
    CombinedPairedImageSpectrumDataset
):
    """The 307K pair stream with clean spectra and AstroCLIP image augmentation."""

    def __init__(self, *args, **kwargs):
        kwargs.update(
            spectra_mask_ratio=0.0,
            spectra_noise_scale=0.0,
            spectra_disjoint_view_masks=False,
        )
        super().__init__(*args, **kwargs)
        self.image_transform = AstroCLIPAlignmentImageTransform(size=140)
