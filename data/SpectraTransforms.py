import torch


class AstroSpectraV2Transform:
    """Prepare fixed-grid DESI spectra for global and local LeJEPA."""

    def __init__(
        self,
        num_views=2,
        patch_size=20,
        expected_num_pixels=7781,
        mask_ratio=0.30,
        mask_span=(2, 12),
        disjoint_view_masks=True,
        min_valid_pixels=20,
        min_valid_patch_fraction=0.50,
        normalization_eps=1e-6,
        noise_scale=0.50,
        max_normalized_noise_std=3.0,
        wavelength_check_every=10_000,
        allow_flux_only=False,
    ):
        if num_views != 2:
            raise ValueError("Spectrum v2 currently requires exactly two LeJEPA views")
        if not 0.0 <= mask_ratio < 0.5 and disjoint_view_masks:
            raise ValueError("Disjoint two-view masks require mask_ratio < 0.5")
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError("mask_ratio must be in [0, 1)")

        self.num_views = num_views
        self.patch_size = patch_size
        self.expected_num_pixels = expected_num_pixels
        self.num_patches = expected_num_pixels // patch_size
        self.usable_num_pixels = self.num_patches * patch_size
        self.mask_ratio = mask_ratio
        self.mask_span = mask_span
        self.disjoint_view_masks = disjoint_view_masks
        self.min_valid_pixels = min_valid_pixels
        self.min_valid_patch_fraction = min_valid_patch_fraction
        self.normalization_eps = normalization_eps
        self.noise_scale = noise_scale
        self.max_normalized_noise_std = max_normalized_noise_std
        self.wavelength_check_every = wavelength_check_every
        self.allow_flux_only = allow_flux_only
        self._reference_wavelength = None
        self._samples_seen = 0

    def _validate_arrays(self, spectrum):
        if not isinstance(spectrum, dict):
            spectrum = {"flux": spectrum}
        required = ("flux", "ivar", "lambda", "lsf_sigma", "mask")
        missing = [name for name in required if name not in spectrum]
        if missing:
            if not self.allow_flux_only or set(spectrum) != {"flux"}:
                raise KeyError(f"DESI spectrum is missing fields: {missing}")
            flux = torch.as_tensor(spectrum["flux"], dtype=torch.float32).reshape(-1)
            if flux.numel() != self.expected_num_pixels:
                raise ValueError(
                    f"Expected {self.expected_num_pixels} flux values, got {flux.numel()}"
                )
            return {"flux": flux, "uncertainty_available": False}

        arrays = {
            "flux": torch.as_tensor(spectrum["flux"], dtype=torch.float32),
            "ivar": torch.as_tensor(spectrum["ivar"], dtype=torch.float32),
            "lambda": torch.as_tensor(spectrum["lambda"], dtype=torch.float32),
            "lsf_sigma": torch.as_tensor(spectrum["lsf_sigma"], dtype=torch.float32),
            "mask": torch.as_tensor(spectrum["mask"], dtype=torch.bool),
        }
        lengths = {name: value.numel() for name, value in arrays.items()}
        if any(length != self.expected_num_pixels for length in lengths.values()):
            raise ValueError(
                f"Expected {self.expected_num_pixels} values in each spectrum field, got {lengths}"
            )
        arrays["uncertainty_available"] = True
        return arrays

    def _validate_wavelength(self, wavelength):
        should_check = (
            self._reference_wavelength is None
            or self.wavelength_check_every > 0
            and self._samples_seen % self.wavelength_check_every == 0
        )
        self._samples_seen += 1
        if not should_check:
            return
        if not torch.isfinite(wavelength).all():
            raise ValueError("DESI wavelength grid contains non-finite values")
        if torch.any(wavelength[1:] < wavelength[:-1]):
            raise ValueError("DESI wavelength grid is not monotonic")
        if self._reference_wavelength is None:
            self._reference_wavelength = wavelength.clone()
        elif not torch.equal(wavelength, self._reference_wavelength):
            raise ValueError("DESI wavelength grid changed within the fixed-grid v2 dataset")

    def _patchify(self, values):
        return values[: self.usable_num_pixels].reshape(self.num_patches, self.patch_size)

    def _make_span_mask(self, eligible, forbidden=None):
        available = eligible.clone()
        if forbidden is not None:
            available &= ~forbidden

        target = min(
            int(round(float(eligible.sum().item()) * self.mask_ratio)),
            int(available.sum().item()),
        )
        mask = torch.zeros_like(eligible)
        if target == 0:
            return mask

        min_span, max_span = self.mask_span
        attempts = 0
        max_attempts = self.num_patches * 8
        while int(mask.sum().item()) < target and attempts < max_attempts:
            span = int(torch.randint(min_span, max_span + 1, (1,)).item())
            span = min(span, self.num_patches)
            start = int(torch.randint(0, self.num_patches - span + 1, (1,)).item())
            candidates = torch.arange(start, start + span)
            candidates = candidates[available[candidates] & ~mask[candidates]]
            remaining = target - int(mask.sum().item())
            mask[candidates[:remaining]] = True
            attempts += 1

        remaining = target - int(mask.sum().item())
        if remaining > 0:
            candidates = torch.nonzero(available & ~mask, as_tuple=False).flatten()
            order = torch.randperm(candidates.numel())[:remaining]
            mask[candidates[order]] = True
        return mask

    def augment_spectrum(self, spectrum):
        arrays = self._validate_arrays(spectrum)
        uncertainty_available = arrays["uncertainty_available"]
        if uncertainty_available:
            self._validate_wavelength(arrays["lambda"])

        flux = arrays["flux"]
        valid = torch.isfinite(flux)
        if uncertainty_available:
            ivar = arrays["ivar"]
            pipeline_mask = arrays["mask"]
            valid &= torch.isfinite(ivar) & (ivar > 0) & ~pipeline_mask
        if int(valid.sum().item()) < self.min_valid_pixels:
            return None

        valid_flux = flux[valid]
        location = valid_flux.mean()
        scale = valid_flux.std(correction=0)
        if not torch.isfinite(location) or not torch.isfinite(scale):
            return None
        scale_was_floored = bool(scale < self.normalization_eps)
        scale = scale.clamp_min(self.normalization_eps)

        normalized = torch.zeros_like(flux)
        normalized[valid] = (flux[valid] - location) / scale

        normalized_noise_std = torch.zeros_like(flux)
        if uncertainty_available:
            normalized_noise_std[valid] = torch.rsqrt(ivar[valid]) / scale
            normalized_noise_std.clamp_(max=self.max_normalized_noise_std)

        normalized_patches = self._patchify(normalized)
        noise_std_patches = self._patchify(normalized_noise_std)
        valid_pixels = self._patchify(valid)
        patch_valid_fraction = valid_pixels.float().mean(dim=-1)
        valid_target_patches = patch_valid_fraction >= self.min_valid_patch_fraction
        if int(valid_target_patches.sum().item()) < 2:
            return None

        views = []
        for _ in range(self.num_views):
            noise = torch.randn_like(normalized_patches) * noise_std_patches
            view = normalized_patches + self.noise_scale * noise
            views.append(torch.where(valid_pixels, view, torch.zeros_like(view)))

        first_mask = self._make_span_mask(valid_target_patches)
        second_mask = self._make_span_mask(
            valid_target_patches,
            forbidden=first_mask if self.disjoint_view_masks else None,
        )

        return {
            "views": torch.stack(views),
            "valid_pixels": valid_pixels,
            "valid_target_patches": valid_target_patches,
            "jepa_masks": torch.stack([first_mask, second_mask]),
            "normalization_mean": location,
            "normalization_log_std": scale.log(),
            "normalization_scale_was_floored": torch.tensor(scale_was_floored),
            "valid_pixel_fraction": valid.float().mean(),
            "uncertainty_available": torch.tensor(uncertainty_available),
        }
