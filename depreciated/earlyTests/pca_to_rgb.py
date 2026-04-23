#!/usr/bin/env python3
"""
PCA-to-RGB patch-feature maps (LeJEPA Fig-14 style) for CLIP ViT vision tower.

Default model = openai/clip-vit-large-patch14-336 (matches AstroLLaVA mm_vision_tower).

What it does:
  - Runs the CLIP vision encoder on an input image
  - Extracts last-layer patch embeddings (excluding CLS)
  - Performs PCA over patch embeddings for that image (no sklearn dependency)
  - Maps the top 3 principal components to R,G,B (per patch), reshapes to grid
  - Upsamples to image size and saves:
      * pca_rgb.png          (RGB visualization)
      * pca_rgb.npy          (raw RGB float in [0,1])
      * patch_embeds.npy     (raw patch embeddings)
      * input_{size}.png     (resized input)
  - Optionally saves the individual PC heatmaps as grayscale PNGs

Deps:
  pip install torch torchvision transformers pillow matplotlib numpy
"""

import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

from transformers import CLIPVisionModel, CLIPImageProcessor


def _to_numpy_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def force_eager_attention(model: CLIPVisionModel):
    """
    Bulletproof forcing of eager attention (kept identical style to your other script).
    Not strictly required here (we don't request attentions), but harmless and consistent.
    """
    try:
        model.set_attn_implementation("eager")
    except Exception:
        pass

    if hasattr(model, "vision_model"):
        try:
            model.vision_model.set_attn_implementation("eager")
        except Exception:
            pass

    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    if hasattr(model, "vision_model") and hasattr(model.vision_model, "config") and hasattr(model.vision_model.config, "_attn_implementation"):
        model.vision_model.config._attn_implementation = "eager"


@torch.no_grad()
def get_last_layer_patch_embeddings(model, pixel_values):
    """
    Returns:
      patch_embeds: (num_patches, hidden_dim) torch.float32 on CPU
      grid_size: int (sqrt(num_patches))
    """
    # We need hidden states to get last-layer token embeddings.
    out = model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
    last = out.hidden_states[-1][0]  # (tokens, dim) for batch[0]
    # tokens = 1 + num_patches (CLS + patches) for CLIP ViT
    patch = last[1:, :]              # (num_patches, dim)

    num_patches = patch.shape[0]
    grid_size = int(math.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"num_patches={num_patches} is not a square; can't reshape to grid")

    return patch.float().cpu(), grid_size


def pca_3_components(X: torch.Tensor):
    """
    X: (N, D) torch tensor on CPU
    Returns:
      Z: (N, 3) projected coordinates onto top-3 PCs
    Notes:
      - Uses SVD on centered data: Xc = U S V^T
      - PCs are rows of V^T; project by Xc @ V[:, :3]
    """
    # Center
    Xc = X - X.mean(dim=0, keepdim=True)  # (N, D)

    # SVD (economy)
    # Xc = U S Vh, where Vh: (D, D) (or (min(N,D), D) depending on backend)
    # For stability, use full_matrices=False
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)  # Vh: (K, D)
    V = Vh.transpose(0, 1)                                # (D, K)
    V3 = V[:, :3]                                         # (D, 3)
    Z = Xc @ V3                                           # (N, 3)
    return Z


def normalize_01(v: torch.Tensor):
    """
    Normalize each column independently to [0,1].
    v: (N, 3)
    """
    v_min = v.min(dim=0, keepdim=True).values
    v_max = v.max(dim=0, keepdim=True).values
    return (v - v_min) / (v_max - v_min + 1e-8)


def upsample_patch_rgb(rgb_patches_01: np.ndarray, grid_size: int, out_size: int):
    """
    rgb_patches_01: (num_patches, 3) float in [0,1]
    Returns rgb_map_01: (out_size, out_size, 3) float in [0,1]
    """
    # (gh, gw, 3)
    grid = rgb_patches_01.reshape(grid_size, grid_size, 3)
    # torch upsample expects NCHW
    t = torch.from_numpy(grid).permute(2, 0, 1).unsqueeze(0).float()  # (1,3,gh,gw)
    t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
    out = t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()           # (H,W,3)
    return out


def save_rgb_png(path, rgb_01: np.ndarray):
    """
    rgb_01: HxWx3 float in [0,1]
    """
    img = (np.clip(rgb_01, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img).save(path)


def save_gray_png(path, heat_01: np.ndarray):
    """
    heat_01: HxW float in [0,1]
    """
    img = (np.clip(heat_01, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--model", default="openai/clip-vit-large-patch14-336", help="CLIP vision tower id")
    ap.add_argument("--device", default=("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")))
    ap.add_argument("--outdir", default="pca_outputs")
    ap.add_argument("--save_pc_maps", action="store_true", help="Also save per-PC heatmaps (PC1/PC2/PC3)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load image
    img = Image.open(args.image).convert("RGB")

    # Load processor + model
    processor = CLIPImageProcessor.from_pretrained(args.model)
    model = CLIPVisionModel.from_pretrained(args.model, attn_implementation="eager")
    model = model.to(args.device).eval()
    force_eager_attention(model)

    # Debug
    attn_impl = getattr(model.config, "_attn_implementation", None)
    print(f"[debug] attention implementation: {attn_impl}")

    # Prepare inputs
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(args.device)

    # Output size aligned to model config (224 for B/32, 336 for L/14-336, etc.)
    out_size = int(model.config.image_size)

    # Get patch embeddings
    patch_embeds, grid_size = get_last_layer_patch_embeddings(model, pixel_values)  # (N,D), int

    # PCA -> (N,3)
    Z = pca_3_components(patch_embeds)     # (N,3)
    Z01 = normalize_01(Z).numpy()          # (N,3) in [0,1]

    # Upsample to image size
    rgb_map_01 = upsample_patch_rgb(Z01, grid_size, out_size)  # (H,W,3)

    # Save input resized (for reference)
    img_resized = img.resize((out_size, out_size))
    Image.fromarray(_to_numpy_rgb(img_resized)).save(os.path.join(args.outdir, f"input_{out_size}.png"))

    # Save outputs
    np.save(os.path.join(args.outdir, "patch_embeds.npy"), patch_embeds.numpy())
    np.save(os.path.join(args.outdir, "pca_rgb.npy"), rgb_map_01)
    save_rgb_png(os.path.join(args.outdir, "pca_rgb.png"), rgb_map_01)

    # Optionally save individual PC maps
    if args.save_pc_maps:
        # reshape per-patch PC values (before upsample) -> upsample each channel
        pc1 = Z01[:, 0].reshape(grid_size, grid_size)
        pc2 = Z01[:, 1].reshape(grid_size, grid_size)
        pc3 = Z01[:, 2].reshape(grid_size, grid_size)

        def upsample_scalar(grid_2d):
            t = torch.from_numpy(grid_2d).unsqueeze(0).unsqueeze(0).float()  # (1,1,gh,gw)
            t = F.interpolate(t, size=(out_size, out_size), mode="bilinear", align_corners=False)
            return t.squeeze().numpy()

        pc1_up = upsample_scalar(pc1)
        pc2_up = upsample_scalar(pc2)
        pc3_up = upsample_scalar(pc3)

        save_gray_png(os.path.join(args.outdir, "pc1_gray.png"), pc1_up)
        save_gray_png(os.path.join(args.outdir, "pc2_gray.png"), pc2_up)
        save_gray_png(os.path.join(args.outdir, "pc3_gray.png"), pc3_up)

    # Show
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.title(f"Input ({out_size}x{out_size})"); plt.axis("off"); plt.imshow(_to_numpy_rgb(img_resized))
    plt.subplot(1, 2, 2); plt.title("PCA-to-RGB (patch features)"); plt.axis("off"); plt.imshow(rgb_map_01)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
