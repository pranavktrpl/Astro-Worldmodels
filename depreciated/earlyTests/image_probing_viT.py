#!/usr/bin/env python3
"""
ViT Attention Rollout / CLS Attention heatmaps for CLIP ViT vision tower.

Produces (NO overlays):
  1) Attention rollout heatmap (CLS -> patches after composing layers)
  2) Last-layer CLS attention heatmap
  3) Optional thresholded binary mask (top-p% attention)
  4) Saves raw arrays as .npy for reproducibility

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


@torch.no_grad()
def get_attentions(model, pixel_values):
    """
    Returns list of attentions (len = num_layers).
    Each element shape: (B, heads, tokens, tokens)
    """
    out = model(pixel_values=pixel_values, output_attentions=True, return_dict=True)
    if out.attentions is None:
        raise RuntimeError(
            "attentions=None. This happens when the model is using SDPA attention.\n"
            "Fix: load with attn_implementation='eager' (this script tries hard to do that).\n"
            "If you STILL see this, check you're running the edited file/environment.\n"
        )
    return out.attentions


def force_eager_attention(model: CLIPVisionModel):
    """
    Bulletproof forcing of eager attention (covers different Transformers behaviors).
    """
    # Preferred public API (may exist depending on transformers version)
    try:
        model.set_attn_implementation("eager")
    except Exception:
        pass

    # CLIPVisionModel sometimes has nested .vision_model
    if hasattr(model, "vision_model"):
        try:
            model.vision_model.set_attn_implementation("eager")
        except Exception:
            pass

    # Last resort: set private config flag used by transformers internals
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    if hasattr(model, "vision_model") and hasattr(model.vision_model, "config") and hasattr(model.vision_model.config, "_attn_implementation"):
        model.vision_model.config._attn_implementation = "eager"


def attention_rollout(attentions, add_residual=True):
    """
    attentions: list of (B, H, T, T)
    Returns joint attention J of shape (T, T) for batch[0]
    """
    # Average heads -> (B, T, T)
    attn = [a.mean(dim=1) for a in attentions]  # list of (B, T, T)
    B, T, _ = attn[0].shape
    device = attn[0].device

    # Start with identity: joint attention from input to current
    J = torch.eye(T, device=device).unsqueeze(0).repeat(B, 1, 1)  # (B, T, T)

    for A in attn:
        if add_residual:
            A = A + torch.eye(T, device=device).unsqueeze(0)
        # Row-normalize
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
        # Compose
        J = torch.bmm(A, J)

    return J[0]  # (T, T)


def cls_to_heatmap(vec, grid_size, out_size):
    """
    vec: (num_patches,) attention scores for patches
    grid_size: int, e.g. 24 for 336/14, or 7 for 224/32
    out_size: int, e.g. 336 or 224
    Returns heatmap (out_size, out_size) float in [0,1]
    """
    h = vec.reshape(grid_size, grid_size).unsqueeze(0).unsqueeze(0)  # (1,1,gh,gw)
    h = F.interpolate(h, size=(out_size, out_size), mode="bilinear", align_corners=False)
    h = h.squeeze().clamp(min=0)
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)
    return h.detach().cpu().numpy()


def save_heatmap_png(path, heatmap, cmap="inferno"):
    """
    Saves a heatmap as a colored PNG (no overlay).
    """
    colored = (plt.get_cmap(cmap)(heatmap)[:, :, :3] * 255).astype(np.uint8)
    Image.fromarray(colored).save(path)


def save_heatmap_gray_png(path, heatmap):
    """
    Saves a heatmap as an 8-bit grayscale PNG.
    """
    gray = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(gray, mode="L").save(path)


def top_percent_mask(heatmap, top_p=0.15):
    """
    heatmap: HxW float [0,1]
    top_p: keep top p fraction of pixels (e.g. 0.15 keeps top 15%)
    """
    flat = heatmap.flatten()
    k = max(1, int((1 - top_p) * flat.size))
    thresh = np.partition(flat, k)[k]
    return (heatmap >= thresh).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--model", default="openai/clip-vit-large-patch14-336", help="CLIP vision tower id")
    ap.add_argument("--device", default=("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")))
    ap.add_argument("--outdir", default="attn_outputs")
    ap.add_argument("--make_mask", action="store_true", help="Also output thresholded binary mask")
    ap.add_argument("--mask_top_p", type=float, default=0.15, help="Top-p fraction for mask")
    ap.add_argument("--cmap", default="inferno", help="Matplotlib colormap for heatmap PNGs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load image
    img = Image.open(args.image).convert("RGB")

    # Load processor
    processor = CLIPImageProcessor.from_pretrained(args.model)

    # Load model WITH eager attention (critical for output_attentions)
    model = CLIPVisionModel.from_pretrained(args.model, attn_implementation="eager")
    model = model.to(args.device).eval()
    force_eager_attention(model)

    # Useful debug print
    attn_impl = getattr(model.config, "_attn_implementation", None)
    print(f"[debug] attention implementation: {attn_impl}")

    # Prepare inputs
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(args.device)

    # Get attentions (will raise a clear error if None)
    attentions = get_attentions(model, pixel_values)

    # tokens = 1 + num_patches (CLS + patches)
    B, Hh, T, _ = attentions[0].shape
    num_patches = T - 1
    grid_size = int(math.sqrt(num_patches))
    if grid_size * grid_size != num_patches:
        raise ValueError(f"num_patches={num_patches} is not a square; can't reshape to grid")

    # Use model-config image size (224 for B/32, 336 for L/14-336, etc.)
    out_size = int(model.config.image_size)

    # Resize for aligned visualization (NO overlays, just heatmaps)
    img_resized = img.resize((out_size, out_size))
    rgb = _to_numpy_rgb(img_resized)

    # 1) Last-layer CLS attention heatmap
    last = attentions[-1].mean(dim=1)[0]  # (T,T)
    cls_last = last[0, 1:]                # (num_patches,)
    heat_last = cls_to_heatmap(cls_last, grid_size, out_size)

    # 2) Attention rollout heatmap
    J = attention_rollout(attentions, add_residual=True)  # (T,T)
    cls_roll = J[0, 1:]
    heat_roll = cls_to_heatmap(cls_roll, grid_size, out_size)

    # Save input + raw arrays
    Image.fromarray(rgb).save(os.path.join(args.outdir, f"input_{out_size}.png"))
    np.save(os.path.join(args.outdir, "heat_lastlayer.npy"), heat_last)
    np.save(os.path.join(args.outdir, "heat_rollout.npy"), heat_roll)

    # Save heatmaps as images (colored + grayscale)
    save_heatmap_png(os.path.join(args.outdir, "heat_lastlayer.png"), heat_last, cmap=args.cmap)
    save_heatmap_png(os.path.join(args.outdir, "heat_rollout.png"), heat_roll, cmap=args.cmap)
    save_heatmap_gray_png(os.path.join(args.outdir, "heat_lastlayer_gray.png"), heat_last)
    save_heatmap_gray_png(os.path.join(args.outdir, "heat_rollout_gray.png"), heat_roll)

    # Optional mask (Fig-13-ish)
    if args.make_mask:
        mask = top_percent_mask(heat_last, top_p=args.mask_top_p) * 255
        Image.fromarray(mask.astype(np.uint8), mode="L").save(os.path.join(args.outdir, "mask_lastlayer_topP.png"))

    # Show (no overlays)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.title(f"Input ({out_size}x{out_size})"); plt.axis("off"); plt.imshow(rgb)
    plt.subplot(1, 3, 2); plt.title("Last-layer CLS heatmap"); plt.axis("off"); plt.imshow(heat_last, cmap=args.cmap)
    plt.subplot(1, 3, 3); plt.title("Attention rollout heatmap"); plt.axis("off"); plt.imshow(heat_roll, cmap=args.cmap)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
