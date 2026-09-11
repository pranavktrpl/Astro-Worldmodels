from .SpectraTransforms import AstroSpectraV2Transform


class AstroSpectraV2OverlapTransform(AstroSpectraV2Transform):
    """Spectrum-v2 preprocessing with configurable overlapping wavelength patches."""

    def __init__(self, patch_stride=10, **kwargs):
        patch_size = kwargs.get("patch_size", 20)
        expected_num_pixels = kwargs.get("expected_num_pixels", 7781)
        if not 1 <= patch_stride <= patch_size:
            raise ValueError("patch_stride must be between 1 and patch_size")
        if expected_num_pixels < patch_size:
            raise ValueError("expected_num_pixels must be at least patch_size")

        super().__init__(**kwargs)
        self.patch_stride = patch_stride
        self.num_patches = 1 + (
            self.expected_num_pixels - self.patch_size
        ) // self.patch_stride
        self.covered_num_pixels = (
            (self.num_patches - 1) * self.patch_stride + self.patch_size
        )

    def _patchify(self, values):
        return values[: self.covered_num_pixels].unfold(
            dimension=0,
            size=self.patch_size,
            step=self.patch_stride,
        )
