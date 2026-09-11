from .SpectraTransformsOverlap import AstroSpectraV2OverlapTransform
from .dataloaders import DesiSpectraDataset


class DesiSpectraOverlapDataset(DesiSpectraDataset):
    """Existing DESI stream with the isolated overlapping-patch v2 transform."""

    def __init__(self, patch_stride=10, **kwargs):
        super().__init__(**kwargs)
        base = self.transformer
        self.transformer = AstroSpectraV2OverlapTransform(
            patch_stride=patch_stride,
            num_views=base.num_views,
            patch_size=base.patch_size,
            expected_num_pixels=base.expected_num_pixels,
            mask_ratio=base.mask_ratio,
            mask_span=base.mask_span,
            disjoint_view_masks=base.disjoint_view_masks,
            min_valid_pixels=base.min_valid_pixels,
            min_valid_patch_fraction=base.min_valid_patch_fraction,
            normalization_eps=base.normalization_eps,
            noise_scale=base.noise_scale,
            max_normalized_noise_std=base.max_normalized_noise_std,
            wavelength_check_every=base.wavelength_check_every,
            allow_flux_only=base.allow_flux_only,
        )
