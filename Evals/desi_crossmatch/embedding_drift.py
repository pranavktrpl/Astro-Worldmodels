#!/usr/bin/env python
"""Cosine drift between two cached embedding sets (same galaxies, two models).

    python Evals/desi_crossmatch/embedding_drift.py \
        results/astro_vit_small_step_21000 results/vit_small_adapted_xmatch

Mean cosine >= ~0.99 means the adaptation barely moved the representation
(null result says nothing yet); ~0.8-0.95 means it genuinely moved.
"""

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent


def load(arg: str) -> np.ndarray:
    path = Path(arg)
    if not path.is_absolute() and not path.exists():
        path = SCRIPT_DIR / arg
    return np.load(path / "embeddings.npy")


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    base, adapted = load(sys.argv[1]), load(sys.argv[2])
    if base.shape != adapted.shape:
        sys.exit(f"shape mismatch: {base.shape} vs {adapted.shape}")
    a = base / np.linalg.norm(base, axis=1, keepdims=True).clip(1e-12)
    b = adapted / np.linalg.norm(adapted, axis=1, keepdims=True).clip(1e-12)
    cos = (a * b).sum(axis=1)
    print(
        f"{base.shape[0]} galaxies, dim {base.shape[1]}: "
        f"mean cosine {cos.mean():.4f}, median {np.median(cos):.4f}, "
        f"5th pct {np.percentile(cos, 5):.4f}, min {cos.min():.4f}"
    )


if __name__ == "__main__":
    main()
