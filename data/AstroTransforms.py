"""
An astronomy faithful augmentation pipeline for multi-crop training, inspired by AstroClip paper - https://arxiv.org/pdf/2310.03024.

See section 5.1 "Galaxy Image Pre-Training" .
"""
import torch
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
import numpy as np
import skimage.filters
import skimage.transform

from typing import List


"""
GaussianBlur and GaussianNoise are copied from the AstroClip's public training repo - https://github.com/PolymathicAI/AstroCLIP/blob/main/astroclip/astrodino/data/augmentations.py
"""
class GaussianBlur:
    """
    Augmentations tuned to the Legacy Survey Data (with minor modifications).

    Code copied from
    https://github.com/georgestein/ssl-legacysurvey/blob/main/ssl_legacysurvey/data_loaders/decals_augmentations.py#L296
    """

    def __init__(
        self,
        scaling: List = [1.0],
        im_dim: int = 144,
        im_ch: int = 3,
        decals: bool = True,
        uniform: bool = False,
    ):
        self.decals = decals
        self.im_ch = im_ch
        self.im_dim = im_dim
        self.uniform = uniform

        # Log normal fit paramaters
        self.shape_dist = np.array([0.2109966, 0.3008485, 0.3471172])
        self.loc_dist = np.array([1.0807153, 1.2394326, 1.1928363])
        self.scale_dist = np.array([1.3153171, 0.9164757, 0.8233702])

        self.sigma_dist = np.log(self.scale_dist)

        self.psf_ch_min = np.array([1.3233109, 1.2667341, 1.2126263])
        self.psf_ch_max = np.array([5.0, 4.5, 4.25])

    def __call__(self, image: np.ndarray):
        # noise in channels is uncorrelated, as images taken at different times/telescopes
        # draw 'true' noise level of each channel from lognormal fits

        # image: torch tensor [C, H, W]
        image = image.clone()

        self.sigma_true = (
            np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
        )

        if self.uniform:
            self.sigma_final = np.random.uniform(self.psf_ch_min, self.psf_ch_max)
        else:
            self.sigma_final = (
                np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
            )

        self.sigma_augment = self.sigma_final**2 - self.sigma_true**2
        self.sigma_augment[self.sigma_augment < 0.0] = 0.0
        self.sigma_augment = np.sqrt(self.sigma_augment)

        for i in range(self.im_ch):
            if self.sigma_augment[i] > 0.0:
                ch = image[i].cpu().numpy()
                ch = skimage.filters.gaussian(ch, sigma=self.sigma_augment[i], mode="reflect")
                image[i] = torch.from_numpy(ch).to(image.device, dtype=image.dtype)

        return image
class GaussianNoise:
    """
    Augmentations tuned to the Legacy Survey Data (with minor modifications).

    Code copied from
    https://github.com/georgestein/ssl-legacysurvey/blob/main/ssl_legacysurvey/data_loaders/decals_augmentations.py#L296
    """

    def __init__(
        self,
        scaling: List = [1.0],
        mean: float = 0,
        im_dim: int = 144,
        im_ch: int = 3,
        decals: bool = True,
        uniform: bool = False,
    ):
        self.mean = mean
        self.decals = decals
        self.im_ch = im_ch
        self.im_dim = im_dim
        self.uniform = uniform

        # Log normal fit paramaters
        self.shape_dist = np.array([0.2264926, 0.2431146, 0.1334844])
        self.loc_dist = np.array([-0.0006735, -0.0023663, -0.0143416])
        self.scale_dist = np.array([0.0037602, 0.0067417, 0.0260779])

        self.sigma_dist = np.log(self.scale_dist)

        # noise in channels is uncorrelated, as images taken at dirrerent times/telescopes
        self.noise_ch_min = np.array([0.001094, 0.001094, 0.001094])
        self.noise_ch_max = np.array([0.013, 0.018, 0.061])

    def __call__(self, image: np.ndarray):
        # draw 'true' noise level of each channel from lognormal fits
        # image: torch tensor [C, H, W]
        image = image.clone()

        self.sigma_true = (
            np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
        )

        if self.uniform:
            self.sigma_final = np.random.uniform(self.noise_ch_min, self.noise_ch_max)
        else:
            self.sigma_final = (
                np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist
            )

        self.sigma_augment = self.sigma_final**2 - self.sigma_true**2
        self.sigma_augment[self.sigma_augment < 0.0] = 0.0
        self.sigma_augment = np.sqrt(self.sigma_augment)

        H, W = image.shape[-2:]
        for i in range(self.im_ch):
            if self.sigma_augment[i] > 0.0:
                noise = np.random.normal(self.mean, self.sigma_augment[i], size=(H, W))
                noise = torch.from_numpy(noise).to(image.device, dtype=image.dtype)
                image[i] = image[i] + noise

        return image

class RandomAstroGaussianBlur(v2.RandomApply):
    """Randomly apply Gaussian blur to the image."""

    def __init__(self, *, p: float = 0.5):
        # keep_p = 1 - p  ## Going by prob definitions in the paper, not in the script
        transform = GaussianBlur()
        super().__init__([transform], p=p)


class RandomAstroGaussianNoise(v2.RandomApply):
    """Randomly apply Gaussian noise to the image."""

    def __init__(self, *, im_dim=144, p: float = 0.5):
        # keep_p = 1 - p  ## Going by prob definitions in the paper, not in the script
        transform = GaussianNoise(im_dim=im_dim)
        super().__init__([transform], p=p)


class AstroMultiCropTransform():
    def __init__(self, Vl=8, Vg=2):
        # self.image = image
        self.Vl = Vl
        self.Vg = Vg
        self.V = Vg + Vl



        self.global_geom = v2.Compose([
            v2.RandomResizedCrop(
                size=140,
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
                size=56,
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
            RandomAstroGaussianBlur(p=1.0),
            RandomAstroGaussianNoise(p=1.0, im_dim=140),
        ])

        self.global2_extra = v2.Compose([
            RandomAstroGaussianBlur(p=0.1),
            RandomAstroGaussianNoise(p=0.1, im_dim=140),
        ])

        self.local_extra = v2.Compose([
            RandomAstroGaussianBlur(p=0.5),
            RandomAstroGaussianNoise(p=0.5, im_dim=56),
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