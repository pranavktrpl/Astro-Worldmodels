import torch
import torch.nn as nn

from models.cross_modal_posttrained import CrossModalPosttrainedModel


class AstroCLIPAttentionPooler(nn.Module):
    """Frozen-token pooling head matching AstroCLIP's released architecture."""

    def __init__(
        self,
        input_dim,
        shared_dim=512,
        num_heads=4,
        dropout=0.1,
        residual_mlp=True,
    ):
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
            nn.Linear(4 * shared_dim, shared_dim),
            nn.Dropout(dropout),
        )
        self.residual_mlp = residual_mlp

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
        transformed = self.mlp(pooled)
        pooled = pooled + transformed if self.residual_mlp else transformed
        return pooled[:, 0]


class CrossModalPosttrainedCLIPModel(CrossModalPosttrainedModel):
    """Our frozen unimodal backbones with AstroCLIP alignment adapters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shared_dim = kwargs.get("shared_dim")
        if shared_dim is None and len(args) >= 2:
            shared_dim = args[1]
        pool_num_heads = kwargs.get("pool_num_heads", 4)
        pool_dropout = kwargs.get("pool_dropout", 0.1)

        self.image_encoder.shared_proj = AstroCLIPAttentionPooler(
            input_dim=self.image_encoder.embed_dim,
            shared_dim=shared_dim,
            num_heads=pool_num_heads,
            dropout=pool_dropout,
            residual_mlp=False,
        )
        self.spectrum_encoder.shared_proj = AstroCLIPAttentionPooler(
            input_dim=self.spectrum_encoder.embed_dim,
            shared_dim=shared_dim,
            num_heads=pool_num_heads,
            dropout=pool_dropout,
            residual_mlp=True,
        )
        self.freeze_backbones()
