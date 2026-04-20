"""
An astronomy faithful augmentation pipeline for multi-crop training, inspired by AstroClip paper - https://arxiv.org/pdf/2310.03024.

See section 5.1 "Galaxy Image Pre-Training" .
"""
import torch
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

class AstroMultiCropTransform():
    def __init__(self, Vl=8, Vg=2):
        # self.image = image
        self.Vl = Vl
        self.Vg = Vg
        self.V = Vg + Vl

        self.global_geom = v2.Compose([
            v2.RandomResizedCrop(
                size=144,
                scale=(0.947, 0.947),
                ratio=(1.0, 1.0),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(
                degrees=(0, 180),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
        ])

        self.local_geom = v2.Compose([
            v2.RandomResizedCrop(
                size=60,
                scale=(0.394, 0.394),
                ratio=(1.0, 1.0),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomRotation(
                degrees=(0, 180),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
        ])

        self.global1_extra = v2.Compose([
            RandomAstroGaussianBlur(p=1.0, im_dim=144),
            RandomAstroGaussianNoise(p=1.0, im_dim=144),
        ])

        self.global2_extra = v2.Compose([
            RandomAstroGaussianBlur(p=0.1, im_dim=144),
            RandomAstroGaussianNoise(p=0.1, im_dim=144),
        ])

        self.local_extra = v2.Compose([
            RandomAstroGaussianBlur(p=0.5, im_dim=60),
            RandomAstroGaussianNoise(p=0.5, im_dim=60),
        ])

        self.to_tensor = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def _prep(self, image):
        return image.convert("RGB")

    def _make_global1(self, image):
        x = self.global_geom(image)
        x = self.to_tensor(x)
        x = self.global1_extra(x)
        return x

    def _make_global2(self, image):
        x = self.global_geom(image)
        x = self.to_tensor(x)
        x = self.global2_extra(x)
        return x

    def _make_local(self, image):
        x = self.local_geom(image)
        x = self.to_tensor(x)
        x = self.local_extra(x)
        return x

    def augment_image(self, image):
        image = self._prep(image)

        global_crops = [self._make_global1(image), self._make_global2(image)]
        local_crops = [self._make_local(image) for _ in range(self.Vl)]

        return {
            "global_crops": torch.stack(global_crops),                  # [Vg, 3, 144, 144]
            "local_crops": torch.stack(local_crops) if self.Vl > 0 else None,   # [Vl, 3, 60, 60]
        }