import torch


class AstroSpectraMultiCropTransform():
    # Spectra-specific: lightweight transform for patchified DESI spectra without image augmentation dependencies.
    def __init__(
        self,
        Vl=8,
        Vg=2,
        global_scale=(0.947, 1.0),
        local_scale=(0.20, 0.394),
        patch_size=20,
        pad_value=0.0,
    ):
        self.Vl = Vl
        self.Vg = Vg
        self.V = Vg + Vl
        self.global_scale = global_scale
        self.local_scale = local_scale
        self.patch_size = patch_size
        self.pad_value = pad_value

    def _sample_scale(self, scale):
        if isinstance(scale, (tuple, list)):
            low, high = scale
            return float(torch.empty(1).uniform_(low, high).item())
        return float(scale)

    def _prep(self, spectrum):
        spectrum = torch.as_tensor(spectrum, dtype=torch.float32)
        return spectrum.squeeze()

    def _patchify(self, spectrum):
        num_patches = spectrum.shape[-1] // self.patch_size
        usable_length = num_patches * self.patch_size
        spectrum = spectrum[:usable_length]
        return spectrum.reshape(num_patches, self.patch_size)

    def _make_mask(self, num_patches, scale):
        scale = self._sample_scale(scale)
        crop_length = max(1, min(num_patches, int(round(num_patches * scale))))
        start = int(torch.randint(0, num_patches - crop_length + 1, (1,)).item())
        end = start + crop_length

        mask = torch.zeros(num_patches, dtype=torch.float32)
        mask[start:end] = 1.0
        return mask

    def _apply_mask(self, patches, mask):
        crop = torch.full_like(patches, self.pad_value)
        crop[mask.bool()] = patches[mask.bool()]
        return crop

    def augment_spectrum(self, spectrum):
        spectrum = self._prep(spectrum)
        patches = self._patchify(spectrum)
        num_patches = patches.shape[0]

        global_masks = [self._make_mask(num_patches, self.global_scale) for _ in range(self.Vg)]
        local_masks = [self._make_mask(num_patches, self.local_scale) for _ in range(self.Vl)]
        global_crops = [self._apply_mask(patches, mask) for mask in global_masks]
        local_crops = [self._apply_mask(patches, mask) for mask in local_masks]

        return {
            "spectrum_patches": patches,                                      # [389, 20]
            "global_crops": torch.stack(global_crops),                        # [Vg, 389, 20]
            "local_crops": torch.stack(local_crops) if self.Vl > 0 else None, # [Vl, 389, 20]
            "global_masks": torch.stack(global_masks),                        # [Vg, 389]
            "local_masks": torch.stack(local_masks) if self.Vl > 0 else None, # [Vl, 389]
        }
