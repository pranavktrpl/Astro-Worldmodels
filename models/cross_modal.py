import torch
import torch.nn as nn
import timm

from models.resnet9 import MLP


def init_projector_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class CrossModalImageEncoder(nn.Module):
    def __init__(self, model_name, shared_dim, pretrained=False):
        super().__init__()
        self.model_name = model_name
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
            dynamic_img_pad=True,
        )
        self.embed_dim = getattr(self.backbone, "num_features", None)
        if self.embed_dim is None:
            raise ValueError(f"Could not infer num_features for {model_name}")

        self.shared_proj = MLP(
            in_channels=self.embed_dim,
            hidden_channels=[2 * self.embed_dim, 2 * self.embed_dim, shared_dim],
            norm_layer="batch_norm",
        )
        self.shared_proj.apply(init_projector_weights)

    def forward(self, views):
        if views.ndim != 5:
            raise ValueError(f"Expected image views [B,V,C,H,W], got {tuple(views.shape)}")
        batch_size, num_views = views.shape[:2]
        flat_embeddings = self.backbone(views.flatten(0, 1))
        flat_projections = self.shared_proj(flat_embeddings)
        embeddings = flat_embeddings.reshape(batch_size, num_views, -1).transpose(0, 1)
        projections = flat_projections.reshape(batch_size, num_views, -1).transpose(0, 1)
        return embeddings, projections


class CrossModalSpectrumEncoder(nn.Module):
    def __init__(
        self,
        shared_dim,
        patch_size=20,
        num_patches=389,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
        pooling="cls",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.pooling = pooling

        self.patch_embed = nn.Linear(patch_size, embed_dim)
        self.validity_embed = nn.Linear(patch_size, embed_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.shared_proj = MLP(
            in_channels=embed_dim,
            hidden_channels=[2 * embed_dim, 2 * embed_dim, shared_dim],
            norm_layer="batch_norm",
        )

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.zeros_(self.validity_embed.weight)
        self.shared_proj.apply(init_projector_weights)

    def _pool(self, hidden, token_attendable):
        if self.pooling == "cls":
            return hidden[:, 0]
        if self.pooling == "masked_mean":
            patch_hidden = hidden[:, 1:]
            weights = token_attendable[:, 1:].to(patch_hidden.dtype).unsqueeze(-1)
            return (patch_hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        raise ValueError(f"Unknown spectrum pooling mode: {self.pooling}")

    def forward(self, views, valid_pixels, jepa_masks):
        if views.ndim != 4:
            raise ValueError(
                f"Expected spectrum views [B,V,N,P], got {tuple(views.shape)}"
            )
        batch_size, num_views, num_patches, patch_size = views.shape
        if num_patches != self.num_patches or patch_size != self.patch_size:
            raise ValueError(
                f"Expected spectrum patches (*,*,{self.num_patches},{self.patch_size}), "
                f"got {tuple(views.shape)}"
            )
        if valid_pixels.shape != (batch_size, num_patches, patch_size):
            raise ValueError(f"Unexpected valid-pixel shape: {tuple(valid_pixels.shape)}")
        if jepa_masks.shape != (batch_size, num_views, num_patches):
            raise ValueError(f"Unexpected JEPA-mask shape: {tuple(jepa_masks.shape)}")

        flat_views = views.flatten(0, 1)
        flat_valid_pixels = (
            valid_pixels[:, None]
            .expand(batch_size, num_views, num_patches, patch_size)
            .reshape(batch_size * num_views, num_patches, patch_size)
            .bool()
        )
        flat_jepa_masks = jepa_masks.flatten(0, 1).bool()

        tokens = self.patch_embed(flat_views)
        tokens = tokens + self.validity_embed(
            flat_valid_pixels.to(tokens.dtype) - 1.0
        )
        tokens = torch.where(
            flat_jepa_masks.unsqueeze(-1),
            self.mask_token.expand_as(tokens),
            tokens,
        )

        cls_tokens = self.cls_token.expand(tokens.shape[0], -1, -1)
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
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        hidden = self.transformer(
            tokens,
            src_key_padding_mask=~token_attendable,
        )
        hidden = self.norm(hidden)

        flat_embeddings = self._pool(hidden, token_attendable)
        flat_projections = self.shared_proj(flat_embeddings)
        embeddings = flat_embeddings.reshape(batch_size, num_views, -1).transpose(0, 1)
        projections = flat_projections.reshape(batch_size, num_views, -1).transpose(0, 1)
        return embeddings, projections


class CrossModalScratchModel(nn.Module):
    def __init__(
        self,
        image_model_name,
        shared_dim,
        image_pretrained=False,
        spectra_patch_size=20,
        spectra_num_patches=389,
        spectra_embed_dim=768,
        spectra_depth=12,
        spectra_num_heads=12,
        spectra_mlp_ratio=4.0,
        spectra_dropout=0.0,
        spectra_pooling="cls",
    ):
        super().__init__()
        self.image_encoder = CrossModalImageEncoder(
            model_name=image_model_name,
            shared_dim=shared_dim,
            pretrained=image_pretrained,
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

    def forward(
        self,
        image_views,
        spectrum_views,
        spectrum_valid_pixels,
        spectrum_jepa_masks,
    ):
        image_embeddings, image_projections = self.image_encoder(image_views)
        spectrum_embeddings, spectrum_projections = self.spectrum_encoder(
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


def compute_cross_modal_lejepa_loss(
    image_projections,
    spectrum_projections,
    image_sigreg,
    spectrum_sigreg,
    lambd,
):
    if image_projections.ndim != 3 or spectrum_projections.ndim != 3:
        raise ValueError("Cross-modal projections must have shape [V,B,D]")
    if image_projections.shape[0] != 2 or spectrum_projections.shape[0] != 2:
        raise ValueError("The selected objective requires exactly two views per modality")
    if image_projections.shape[1:] != spectrum_projections.shape[1:]:
        raise ValueError(
            "Image and spectrum projections must share batch and feature dimensions"
        )
    if not 0.0 <= lambd <= 1.0:
        raise ValueError("SIGReg weight must be in [0, 1]")

    pairwise_differences = (
        image_projections[:, None] - spectrum_projections[None, :]
    )
    cross_invariance = pairwise_differences.square().mean()
    image_sigreg_loss = image_sigreg(image_projections)
    spectrum_sigreg_loss = spectrum_sigreg(spectrum_projections)
    sigreg = 0.5 * (image_sigreg_loss + spectrum_sigreg_loss)
    loss = (1.0 - lambd) * cross_invariance + lambd * sigreg

    return {
        "loss": loss,
        "cross_invariance": cross_invariance,
        "sigreg": sigreg,
        "image_sigreg": image_sigreg_loss,
        "spectrum_sigreg": spectrum_sigreg_loss,
    }
