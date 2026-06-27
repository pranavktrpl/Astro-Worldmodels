#!/usr/bin/env python3
"""Patch PCA and attention-rollout diagnostics for finalized Astro ViTs.

This script recreates the two old visual tests in ``depreciated/earlyTests``:

1. PCA-to-RGB from final patch tokens.
2. CLS attention heatmaps for every layer.
3. CLS last-layer attention and attention rollout heatmaps.

It intentionally lives as a standalone eval and does not import or modify the
deprecated scripts. Historical CLIP ViT outputs are copied into the output
directory as references.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# This environment's timm imports wandb from an optional utility module. The
# diagnostics do not use wandb, and importing the real package is slow here.
if "wandb" not in sys.modules:
    wandb_stub = types.ModuleType("wandb")
    wandb_stub.log = lambda *args, **kwargs: None
    sys.modules["wandb"] = wandb_stub

import timm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
DEFAULT_IMAGE = (
    REPO_ROOT / "depreciated/earlyTests/RedSpider_Webb_960_apod4feb.jpg"
)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    checkpoint: Path
    display_name: str


DEFAULT_MODELS = (
    ModelSpec(
        label="astro_vit_large_step_52000",
        checkpoint=REPO_ROOT
        / "checkpoints/VitLargePatch14_OfficialTrain5_Epoch5_2504/step_52000.pt",
        display_name="Astro ViT-L/14 step 52000",
    ),
    ModelSpec(
        label="astro_vit_small_step_21000",
        checkpoint=REPO_ROOT / "checkpoints/VitSmallPatch14_2204/step_21000.pt",
        display_name="Astro ViT-S/14 step 21000",
    ),
)


CLIP_REFERENCE_DIRS = (
    (
        "clip_vit_l14_336_pca_outputs",
        REPO_ROOT / "depreciated/earlyTests/pca_outputs",
    ),
    (
        "clip_vit_l14_336_attention_outputs",
        REPO_ROOT / "depreciated/earlyTests/astrollava_same_arch",
    ),
    (
        "clip_vit_legacy_attention_outputs",
        REPO_ROOT / "depreciated/earlyTests/attn_outputs",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PCA-to-RGB and attention-rollout diagnostics."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_IMAGE,
        help="Raw image to probe.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated diagnostics and copied CLIP references.",
    )
    parser.add_argument(
        "--output-size",
        type=int,
        default=336,
        help="Square image size used for the diagnostic forward pass.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for model inference.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.15,
        help="Top heatmap fraction to retain in binary masks.",
    )
    parser.add_argument(
        "--skip-copy-references",
        action="store_true",
        help="Do not copy historical CLIP ViT reference outputs.",
    )
    return parser.parse_args()


def torch_load(path: Path) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"map_location": "cpu", "weights_only": False}
    try:
        return torch.load(path, mmap=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def load_image(image_path: Path, size: int) -> tuple[Image.Image, Image.Image, torch.Tensor]:
    raw = Image.open(image_path).convert("RGB")
    resized = raw.resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return raw, resized, tensor


def load_backbone(spec: ModelSpec, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not spec.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {spec.checkpoint}")

    checkpoint = torch_load(spec.checkpoint)
    state = checkpoint["model"]
    cfg = dict(checkpoint.get("cfg") or {})
    model_name = str(cfg["model_name"])

    backbone = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True,
        dynamic_img_pad=True,
    )
    backbone_state = {
        key.removeprefix("backbone."): value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    backbone.load_state_dict(backbone_state, strict=True)
    backbone.to(device).eval()

    meta = {
        "label": spec.label,
        "display_name": spec.display_name,
        "checkpoint": str(spec.checkpoint.resolve()),
        "global_step": int(checkpoint.get("global_step", -1)),
        "model_name": model_name,
        "cfg": cfg,
        "num_blocks": len(getattr(backbone, "blocks", [])),
        "num_prefix_tokens": int(getattr(backbone, "num_prefix_tokens", 1)),
        "patch_size": list(getattr(backbone.patch_embed, "patch_size", ())),
        "num_features": int(getattr(backbone, "num_features", -1)),
    }

    del checkpoint, state, backbone_state
    gc.collect()
    return backbone, meta


def set_unfused_attention(backbone: torch.nn.Module) -> None:
    for block in getattr(backbone, "blocks", []):
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "fused_attn"):
            attn.fused_attn = False


@torch.inference_mode()
def forward_with_attention(
    backbone: torch.nn.Module,
    batch: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Return final tokens and mean-head attention matrices for each block."""
    set_unfused_attention(backbone)
    attention_mats: list[torch.Tensor] = []
    handles = []

    def make_hook():
        def hook(_module, _inputs, output):
            # output is attention after softmax/dropout: [B, heads, tokens, tokens].
            mean_heads = output.detach().float().mean(dim=1).squeeze(0).cpu()
            attention_mats.append(mean_heads)

        return hook

    for block in getattr(backbone, "blocks", []):
        attn = getattr(block, "attn", None)
        if attn is None or not hasattr(attn, "attn_drop"):
            raise RuntimeError("Could not find attn_drop on a ViT attention block.")
        handles.append(attn.attn_drop.register_forward_hook(make_hook()))

    try:
        tokens = backbone.forward_features(batch)
    finally:
        for handle in handles:
            handle.remove()

    if isinstance(tokens, dict):
        if "x" not in tokens:
            raise RuntimeError(f"Unexpected forward_features dict keys: {tokens.keys()}")
        tokens = tokens["x"]
    if tokens.ndim != 3:
        raise RuntimeError(f"Expected token tensor [B, T, D], got {tuple(tokens.shape)}")
    if len(attention_mats) != len(getattr(backbone, "blocks", [])):
        raise RuntimeError(
            f"Captured {len(attention_mats)} attention maps for "
            f"{len(getattr(backbone, 'blocks', []))} blocks."
        )
    return tokens.detach().float().cpu(), attention_mats


def square_grid(num_tokens: int) -> tuple[int, int]:
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Patch token count is not square: {num_tokens}")
    return side, side


def minmax(array: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    lo = float(np.nanmin(array))
    hi = float(np.nanmax(array))
    if hi - lo < eps:
        return np.zeros_like(array, dtype=np.float32)
    return (array - lo) / (hi - lo)


def pca_rgb_from_patches(patch_tokens: torch.Tensor) -> np.ndarray:
    centered = patch_tokens - patch_tokens.mean(dim=0, keepdim=True)
    _u, _s, vh = torch.linalg.svd(centered, full_matrices=False)
    rgb = centered @ vh[:3].T

    # PCA signs are arbitrary. Flip each component so its strongest loading is
    # positive, giving stable visual colors across reruns.
    for channel in range(rgb.shape[1]):
        col = rgb[:, channel]
        idx = int(torch.argmax(col.abs()).item())
        if col[idx] < 0:
            rgb[:, channel] = -col

    rgb_np = rgb.numpy()
    for channel in range(3):
        rgb_np[:, channel] = minmax(rgb_np[:, channel])
    return rgb_np.astype(np.float32)


def upsample_grid(array: np.ndarray, size: int, mode: str = "bilinear") -> np.ndarray:
    if array.ndim == 2:
        tensor = torch.from_numpy(array).float()[None, None]
        result = F.interpolate(
            tensor,
            size=(size, size),
            mode=mode,
            align_corners=False if mode in {"bilinear", "bicubic"} else None,
        )[0, 0]
        return result.numpy()
    if array.ndim == 3:
        tensor = torch.from_numpy(array).float().permute(2, 0, 1)[None]
        result = F.interpolate(
            tensor,
            size=(size, size),
            mode=mode,
            align_corners=False if mode in {"bilinear", "bicubic"} else None,
        )[0].permute(1, 2, 0)
        return result.numpy()
    raise ValueError(f"Expected 2D or 3D array, got shape {array.shape}")


def save_rgb(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = (np.clip(array, 0.0, 1.0) * 255).round().astype(np.uint8)
    Image.fromarray(image).save(path)


def save_gray(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = (np.clip(array, 0.0, 1.0) * 255).round().astype(np.uint8)
    Image.fromarray(image, mode="L").save(path)


def save_heatmap(array: np.ndarray, path: Path, cmap_name: str = "inferno") -> None:
    cmap = plt.get_cmap(cmap_name)
    colored = cmap(np.clip(array, 0.0, 1.0))[..., :3]
    save_rgb(colored, path)


def save_mask(array: np.ndarray, path: Path, top_p: float) -> None:
    threshold = float(np.quantile(array, max(0.0, min(1.0, 1.0 - top_p))))
    mask = (array >= threshold).astype(np.float32)
    save_gray(mask, path)


def rollout_attention(attention_mats: list[torch.Tensor]) -> torch.Tensor:
    if not attention_mats:
        raise ValueError("No attention matrices captured.")
    tokens = attention_mats[0].shape[-1]
    joint = torch.eye(tokens, dtype=torch.float32)
    eye = torch.eye(tokens, dtype=torch.float32)
    for attn in attention_mats:
        attn = attn.float()
        attn = attn + eye
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        joint = attn @ joint
    return joint


def save_layerwise_cls_attention(
    output_dir: Path,
    attention_mats: list[torch.Tensor],
    prefix_tokens: int,
    grid_h: int,
    grid_w: int,
    output_size: int,
) -> dict[str, Any]:
    layer_dir = output_dir / "cls_attention_by_layer"
    if layer_dir.exists():
        shutil.rmtree(layer_dir)
    layer_dir.mkdir(parents=True, exist_ok=True)

    heatmaps: list[np.ndarray] = []
    metadata: list[dict[str, str | int]] = []
    for idx, attn in enumerate(attention_mats, start=1):
        cls_grid = attn[0, prefix_tokens:].reshape(grid_h, grid_w).numpy()
        grid_heat = minmax(cls_grid)
        heat = minmax(upsample_grid(grid_heat, output_size))
        stem = f"layer_{idx:02d}"

        np.save(layer_dir / f"{stem}_grid.npy", grid_heat)
        np.save(layer_dir / f"{stem}_heat.npy", heat)
        save_heatmap(heat, layer_dir / f"{stem}_heat.png")
        save_gray(heat, layer_dir / f"{stem}_gray.png")
        heatmaps.append(heat)
        metadata.append(
            {
                "layer": idx,
                "heat_png": str(layer_dir / f"{stem}_heat.png"),
                "gray_png": str(layer_dir / f"{stem}_gray.png"),
                "heat_npy": str(layer_dir / f"{stem}_heat.npy"),
                "grid_npy": str(layer_dir / f"{stem}_grid.npy"),
            }
        )

    cols = min(6, len(heatmaps))
    rows = math.ceil(len(heatmaps) / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(2.2 * cols, 2.35 * rows),
        dpi=180,
        constrained_layout=True,
    )
    axes_array = np.asarray(axes).reshape(rows, cols)
    for idx, ax in enumerate(axes_array.flat):
        ax.set_xticks([])
        ax.set_yticks([])
        if idx < len(heatmaps):
            ax.imshow(heatmaps[idx], cmap="inferno", vmin=0.0, vmax=1.0)
            ax.set_title(f"Layer {idx + 1}", fontsize=8)
        else:
            ax.axis("off")

    contact_sheet = layer_dir / "cls_attention_layers_grid.png"
    fig.savefig(contact_sheet)
    plt.close(fig)
    shutil.copy2(contact_sheet, output_dir / "cls_attention_layers_grid.png")

    return {
        "layerwise_cls_attention_dir": str(layer_dir),
        "layerwise_cls_attention_contact_sheet": str(contact_sheet),
        "layerwise_cls_attention_png_count": len(heatmaps),
        "layerwise_cls_attention": metadata,
    }


def save_model_outputs(
    output_dir: Path,
    resized_input: Image.Image,
    tokens: torch.Tensor,
    attention_mats: list[torch.Tensor],
    prefix_tokens: int,
    output_size: int,
    top_p: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    resized_input.save(output_dir / f"input_{output_size}.png")

    patch_tokens = tokens[0, prefix_tokens:].contiguous()
    grid_h, grid_w = square_grid(patch_tokens.shape[0])

    np.save(output_dir / "patch_embeds.npy", patch_tokens.numpy())
    pca_grid = pca_rgb_from_patches(patch_tokens)
    pca_grid_image = pca_grid.reshape(grid_h, grid_w, 3)
    pca_rgb = minmax(upsample_grid(pca_grid_image, output_size))
    np.save(output_dir / "pca_rgb_grid.npy", pca_grid_image)
    np.save(output_dir / "pca_rgb.npy", pca_rgb)
    save_rgb(pca_rgb, output_dir / "pca_rgb.png")

    for channel in range(3):
        pc_grid = pca_grid[:, channel].reshape(grid_h, grid_w)
        pc_image = minmax(upsample_grid(pc_grid, output_size))
        save_gray(pc_image, output_dir / f"pc{channel + 1}_gray.png")

    layerwise = save_layerwise_cls_attention(
        output_dir,
        attention_mats,
        prefix_tokens,
        grid_h,
        grid_w,
        output_size,
    )

    last = attention_mats[-1]
    last_cls = last[0, prefix_tokens:].reshape(grid_h, grid_w).numpy()
    last_heat = minmax(upsample_grid(minmax(last_cls), output_size))
    np.save(output_dir / "heat_lastlayer_grid.npy", minmax(last_cls))
    np.save(output_dir / "heat_lastlayer.npy", last_heat)
    save_heatmap(last_heat, output_dir / "heat_lastlayer.png")
    save_gray(last_heat, output_dir / "heat_lastlayer_gray.png")
    save_mask(last_heat, output_dir / "mask_lastlayer_topP.png", top_p)

    rollout = rollout_attention(attention_mats)
    rollout_cls = rollout[0, prefix_tokens:].reshape(grid_h, grid_w).numpy()
    rollout_heat = minmax(upsample_grid(minmax(rollout_cls), output_size))
    np.save(output_dir / "heat_rollout_grid.npy", minmax(rollout_cls))
    np.save(output_dir / "heat_rollout.npy", rollout_heat)
    save_heatmap(rollout_heat, output_dir / "heat_rollout.png")
    save_gray(rollout_heat, output_dir / "heat_rollout_gray.png")

    return {
        "patch_grid": [grid_h, grid_w],
        "token_count": int(tokens.shape[1]),
        "patch_token_count": int(patch_tokens.shape[0]),
        "embedding_dim": int(patch_tokens.shape[1]),
        "attention_layers": len(attention_mats),
        "attention_shape": list(attention_mats[0].shape),
        **layerwise,
    }


def copy_reference_assets(output_dir: Path, image_path: Path) -> list[dict[str, str]]:
    copied: list[dict[str, str]] = []
    raw_dir = output_dir / "raw_input"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_dst = raw_dir / image_path.name
    shutil.copy2(image_path, raw_dst)
    copied.append({"kind": "raw_image", "source": str(image_path), "target": str(raw_dst)})

    ref_root = output_dir / "clip_vit_references"
    ref_root.mkdir(parents=True, exist_ok=True)
    for name, source in CLIP_REFERENCE_DIRS:
        if source.exists():
            target = ref_root / name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(source, target)
            copied.append({"kind": "clip_reference_dir", "source": str(source), "target": str(target)})
    return copied


def load_optional_image(path: Path, size: int) -> np.ndarray:
    if not path.exists():
        return np.ones((size, size, 3), dtype=np.float32)
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def make_comparison_grid(output_dir: Path, output_size: int) -> None:
    rows = [
        (
            "CLIP ViT-L/14-336 ref",
            output_dir / "clip_vit_references/clip_vit_l14_336_pca_outputs/input_336.png",
            output_dir / "clip_vit_references/clip_vit_l14_336_pca_outputs/pca_rgb.png",
            output_dir / "clip_vit_references/clip_vit_l14_336_attention_outputs/heat_lastlayer.png",
            output_dir / "clip_vit_references/clip_vit_l14_336_attention_outputs/heat_rollout.png",
        ),
        (
            "Astro ViT-L/14 step 52000",
            output_dir / "astro_vit_large_step_52000" / f"input_{output_size}.png",
            output_dir / "astro_vit_large_step_52000/pca_rgb.png",
            output_dir / "astro_vit_large_step_52000/heat_lastlayer.png",
            output_dir / "astro_vit_large_step_52000/heat_rollout.png",
        ),
        (
            "Astro ViT-S/14 step 21000",
            output_dir / "astro_vit_small_step_21000" / f"input_{output_size}.png",
            output_dir / "astro_vit_small_step_21000/pca_rgb.png",
            output_dir / "astro_vit_small_step_21000/heat_lastlayer.png",
            output_dir / "astro_vit_small_step_21000/heat_rollout.png",
        ),
    ]
    columns = ["Input", "PCA to RGB", "Last-layer CLS attn", "Rollout CLS attn"]

    fig, axes = plt.subplots(
        nrows=len(rows),
        ncols=len(columns),
        figsize=(13, 9),
        dpi=180,
        constrained_layout=True,
    )
    for row_idx, (row_label, *paths) in enumerate(rows):
        for col_idx, path in enumerate(paths):
            axes[row_idx, col_idx].imshow(load_optional_image(Path(path), output_size))
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(columns[col_idx], fontsize=10)
            if col_idx == 0:
                axes[row_idx, col_idx].set_ylabel(row_label, fontsize=9)
    fig.savefig(output_dir / "comparison_grid.png")
    plt.close(fig)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    raw_image, resized_input, input_tensor = load_image(args.image, args.output_size)
    references = []
    if not args.skip_copy_references:
        references = copy_reference_assets(output_dir, args.image)

    provenance: dict[str, Any] = {
        "image": str(args.image.resolve()),
        "raw_image_size": list(raw_image.size),
        "output_size": args.output_size,
        "device": str(device),
        "normalization": "RGB float in [0, 1]; no ImageNet/CLIP normalization",
        "reference_assets": references,
        "models": [],
    }

    for spec in DEFAULT_MODELS:
        print(f"\n=== {spec.display_name} ===", flush=True)
        print(f"Loading {spec.checkpoint}", flush=True)
        backbone, meta = load_backbone(spec, device)
        batch = input_tensor.to(device=device, dtype=torch.float32)
        print("Running forward pass with attention hooks...", flush=True)
        tokens, attention_mats = forward_with_attention(backbone, batch)
        model_dir = output_dir / spec.label
        diagnostics = save_model_outputs(
            model_dir,
            resized_input,
            tokens,
            attention_mats,
            prefix_tokens=meta["num_prefix_tokens"],
            output_size=args.output_size,
            top_p=args.top_p,
        )
        meta.update(diagnostics)
        provenance["models"].append(meta)
        write_json(model_dir / "metadata.json", meta)
        print(f"Saved {model_dir}", flush=True)

        del backbone, batch, tokens, attention_mats
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    make_comparison_grid(output_dir, args.output_size)
    write_json(output_dir / "provenance.json", provenance)
    print(f"\nDone. Main comparison: {output_dir / 'comparison_grid.png'}")


if __name__ == "__main__":
    main()
