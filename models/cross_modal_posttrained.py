from pathlib import Path

import torch
import torch.nn as nn

from models.cross_modal import CrossModalImageEncoder
from models.cross_modal import CrossModalSpectrumEncoder


def _torch_load_weights(path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )


def _validate_state_dict(source, target, label):
    source_keys = set(source)
    target_keys = set(target)
    missing = sorted(target_keys - source_keys)
    unexpected = sorted(source_keys - target_keys)
    if missing or unexpected:
        raise RuntimeError(
            f"{label} checkpoint keys do not match the requested backbone: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    mismatched = []
    for name, target_value in target.items():
        source_value = source[name]
        if source_value.shape != target_value.shape:
            mismatched.append(
                (name, tuple(source_value.shape), tuple(target_value.shape))
            )
    if mismatched:
        raise RuntimeError(f"{label} checkpoint shape mismatch: {mismatched[:8]}")


def _source_metadata(path, checkpoint, ignored_heads):
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "global_step": int(checkpoint.get("global_step", -1)),
        "epoch": int(checkpoint.get("epoch", -1)),
        "ignored_pretraining_heads": list(ignored_heads),
    }


class QueryAttentionPooler(nn.Module):
    """AstroCLIP-style learned query followed by a residual MLP."""

    def __init__(self, input_dim, shared_dim, num_heads=4, dropout=0.1):
        super().__init__()
        if shared_dim % num_heads != 0:
            raise ValueError("shared_dim must be divisible by pooler num_heads")
        self.query = nn.Parameter(torch.randn(1, 1, shared_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=shared_dim,
            num_heads=num_heads,
            kdim=input_dim,
            vdim=input_dim,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(shared_dim)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(shared_dim, 4 * shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * shared_dim, shared_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens, key_padding_mask=None):
        query = self.query.expand(tokens.shape[0], -1, -1)
        pooled, _ = self.cross_attention(
            query=query,
            key=tokens,
            value=tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        pooled = self.norm(self.dropout(pooled))
        pooled = pooled + self.mlp(pooled)
        return pooled[:, 0]


class CrossModalPosttrainedModel(nn.Module):
    """Frozen unimodal backbones with learned-query token poolers."""

    def __init__(
        self,
        image_model_name,
        shared_dim,
        spectra_patch_size=20,
        spectra_num_patches=389,
        spectra_embed_dim=768,
        spectra_depth=12,
        spectra_num_heads=12,
        spectra_mlp_ratio=4.0,
        spectra_dropout=0.0,
        spectra_pooling="cls",
        pool_num_heads=4,
        pool_dropout=0.1,
    ):
        super().__init__()
        self.image_model_name = image_model_name
        self.image_encoder = CrossModalImageEncoder(
            model_name=image_model_name,
            shared_dim=shared_dim,
            pretrained=False,
        )
        self.spectrum_encoder = CrossModalSpectrumEncoder(
            shared_dim=shared_dim,
            patch_size=spectra_patch_size,
            num_patches=spectra_num_patches,
            embed_dim=spectra_embed_dim,
            depth=spectra_depth,
            num_heads=spectra_num_heads,
            mlp_ratio=spectra_mlp_ratio,
            dropout=spectra_dropout,
            pooling=spectra_pooling,
        )
        self.image_encoder.shared_proj = QueryAttentionPooler(
            input_dim=self.image_encoder.embed_dim,
            shared_dim=shared_dim,
            num_heads=pool_num_heads,
            dropout=pool_dropout,
        )
        self.spectrum_encoder.shared_proj = QueryAttentionPooler(
            input_dim=self.spectrum_encoder.embed_dim,
            shared_dim=shared_dim,
            num_heads=pool_num_heads,
            dropout=pool_dropout,
        )
        self.freeze_backbones()

    def freeze_backbones(self):
        for parameter in self.image_encoder.backbone.parameters():
            parameter.requires_grad = False
        for name, parameter in self.spectrum_encoder.named_parameters():
            parameter.requires_grad = name.startswith("shared_proj.")

        for parameter in self.image_encoder.shared_proj.parameters():
            parameter.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        self.image_encoder.backbone.eval()
        for name, child in self.spectrum_encoder.named_children():
            if name != "shared_proj":
                child.eval()
        return self

    def load_pretrained_backbones(self, image_checkpoint_path, spectrum_checkpoint_path):
        image_path = Path(image_checkpoint_path).resolve()
        spectrum_path = Path(spectrum_checkpoint_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image checkpoint: {image_path}")
        if not spectrum_path.is_file():
            raise FileNotFoundError(f"Missing spectrum checkpoint: {spectrum_path}")

        image_checkpoint = _torch_load_weights(image_path)
        image_config = dict(image_checkpoint.get("cfg") or {})
        checkpoint_model_name = image_config.get("model_name")
        if checkpoint_model_name != self.image_model_name:
            raise RuntimeError(
                "Image checkpoint architecture mismatch: "
                f"checkpoint={checkpoint_model_name}, requested={self.image_model_name}"
            )
        image_source = {
            name.removeprefix("backbone."): value
            for name, value in image_checkpoint["model"].items()
            if name.startswith("backbone.")
        }
        image_target = self.image_encoder.backbone.state_dict()
        _validate_state_dict(image_source, image_target, "image")
        self.image_encoder.backbone.load_state_dict(image_source, strict=True)
        image_metadata = _source_metadata(
            image_path,
            image_checkpoint,
            ignored_heads=("proj",),
        )
        image_metadata["model_name"] = checkpoint_model_name
        del image_checkpoint, image_source, image_target

        spectrum_checkpoint = _torch_load_weights(spectrum_path)
        spectrum_config = dict(spectrum_checkpoint.get("cfg") or {})
        expected_spectrum_config = {
            "spectra_patch_size": self.spectrum_encoder.patch_size,
            "spectra_num_patches": self.spectrum_encoder.num_patches,
            "spectra_embed_dim": self.spectrum_encoder.embed_dim,
            "spectra_pooling": self.spectrum_encoder.pooling,
        }
        mismatched_config = {
            name: (spectrum_config.get(name), expected)
            for name, expected in expected_spectrum_config.items()
            if spectrum_config.get(name) != expected
        }
        if mismatched_config:
            raise RuntimeError(
                f"Spectrum checkpoint architecture mismatch: {mismatched_config}"
            )

        spectrum_source = {
            name: value
            for name, value in spectrum_checkpoint["model"].items()
            if not name.startswith(("global_proj.", "local_proj."))
        }
        spectrum_target = {
            name: value
            for name, value in self.spectrum_encoder.state_dict().items()
            if not name.startswith("shared_proj.")
        }
        _validate_state_dict(spectrum_source, spectrum_target, "spectrum")
        incompatible = self.spectrum_encoder.load_state_dict(
            spectrum_source,
            strict=False,
        )
        expected_missing = sorted(
            name
            for name in self.spectrum_encoder.state_dict()
            if name.startswith("shared_proj.")
        )
        if sorted(incompatible.missing_keys) != expected_missing:
            raise RuntimeError(
                "Unexpected missing spectrum parameters while loading backbone: "
                f"{incompatible.missing_keys}"
            )
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected spectrum parameters while loading backbone: "
                f"{incompatible.unexpected_keys}"
            )
        spectrum_metadata = _source_metadata(
            spectrum_path,
            spectrum_checkpoint,
            ignored_heads=("global_proj", "local_proj"),
        )
        spectrum_metadata["config"] = expected_spectrum_config
        del spectrum_checkpoint, spectrum_source, spectrum_target

        self.freeze_backbones()
        return {
            "image": image_metadata,
            "spectrum": spectrum_metadata,
        }

    def alignment_state_dict(self):
        return {
            "image_attention_pooler": self.image_encoder.shared_proj.state_dict(),
            "spectrum_attention_pooler": self.spectrum_encoder.shared_proj.state_dict(),
        }

    def load_alignment_state_dict(self, state_dict):
        expected = {"image_attention_pooler", "spectrum_attention_pooler"}
        actual = set(state_dict)
        if actual != expected:
            raise RuntimeError(
                "Alignment checkpoint keys do not match: "
                f"missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            )
        self.image_encoder.shared_proj.load_state_dict(
            state_dict["image_attention_pooler"],
            strict=True,
        )
        self.spectrum_encoder.shared_proj.load_state_dict(
            state_dict["spectrum_attention_pooler"],
            strict=True,
        )

    def _encode_image_tokens(self, views):
        if views.ndim != 5:
            raise ValueError(f"Expected image views [B,V,C,H,W], got {views.shape}")
        batch_size, num_views = views.shape[:2]
        with torch.no_grad():
            tokens = self.image_encoder.backbone.forward_features(
                views.flatten(0, 1)
            )
        if not torch.is_tensor(tokens) or tokens.ndim != 3:
            raise RuntimeError("Image backbone did not return a token sequence")
        raw_cls = tokens[:, 0]
        projections = self.image_encoder.shared_proj(tokens)
        return (
            raw_cls.reshape(batch_size, num_views, -1).transpose(0, 1),
            projections.reshape(batch_size, num_views, -1).transpose(0, 1),
        )

    def _encode_spectrum_tokens(self, views, valid_pixels, jepa_masks):
        if views.ndim != 4:
            raise ValueError(
                f"Expected spectrum views [B,V,N,P], got {tuple(views.shape)}"
            )
        batch_size, num_views, num_patches, patch_size = views.shape
        encoder = self.spectrum_encoder
        if (num_patches, patch_size) != (encoder.num_patches, encoder.patch_size):
            raise ValueError(
                "Unexpected spectrum patch shape: "
                f"{(num_patches, patch_size)}"
            )
        if valid_pixels.shape != (batch_size, num_patches, patch_size):
            raise ValueError(f"Unexpected valid-pixel shape: {valid_pixels.shape}")
        if jepa_masks.shape != (batch_size, num_views, num_patches):
            raise ValueError(f"Unexpected JEPA-mask shape: {jepa_masks.shape}")

        flat_views = views.flatten(0, 1)
        flat_valid_pixels = (
            valid_pixels[:, None]
            .expand(batch_size, num_views, num_patches, patch_size)
            .reshape(batch_size * num_views, num_patches, patch_size)
            .bool()
        )
        flat_jepa_masks = jepa_masks.flatten(0, 1).bool()

        with torch.no_grad():
            tokens = encoder.patch_embed(flat_views)
            tokens = tokens + encoder.validity_embed(
                flat_valid_pixels.to(tokens.dtype) - 1.0
            )
            tokens = torch.where(
                flat_jepa_masks.unsqueeze(-1),
                encoder.mask_token.expand_as(tokens),
                tokens,
            )
            cls_tokens = encoder.cls_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)
            patch_attendable = flat_valid_pixels.any(dim=-1)
            token_attendable = torch.cat(
                [
                    torch.ones(
                        patch_attendable.shape[0],
                        1,
                        dtype=torch.bool,
                        device=patch_attendable.device,
                    ),
                    patch_attendable,
                ],
                dim=1,
            )
            tokens = tokens + encoder.pos_embed[:, : tokens.shape[1]]
            tokens = encoder.transformer(
                tokens,
                src_key_padding_mask=~token_attendable,
            )
            tokens = encoder.norm(tokens)

        raw_cls = encoder._pool(tokens, token_attendable)
        projections = encoder.shared_proj(
            tokens,
            key_padding_mask=~token_attendable,
        )
        return (
            raw_cls.reshape(batch_size, num_views, -1).transpose(0, 1),
            projections.reshape(batch_size, num_views, -1).transpose(0, 1),
        )

    def forward(
        self,
        image_views,
        spectrum_views,
        spectrum_valid_pixels,
        spectrum_jepa_masks,
    ):
        image_embeddings, image_projections = self._encode_image_tokens(image_views)
        spectrum_embeddings, spectrum_projections = self._encode_spectrum_tokens(
            spectrum_views,
            spectrum_valid_pixels,
            spectrum_jepa_masks,
        )
        return {
            "image_embeddings": image_embeddings,
            "image_projections": image_projections,
            "spectrum_embeddings": spectrum_embeddings,
            "spectrum_projections": spectrum_projections,
        }
