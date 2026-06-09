#!/usr/bin/env python3
"""Regenerate checkpoint loss plots from cached JSON metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_loss_curves import (  # noqa: E402
    DEFAULT_IGNORED_DIRECTORIES,
    write_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=REPO_ROOT / "checkpoints",
    )
    parser.add_argument("--checkpoint-dir", type=Path, action="append", default=[])
    args = parser.parse_args()

    directories = (
        [path.resolve() for path in args.checkpoint_dir]
        if args.checkpoint_dir
        else sorted(
            path
            for path in args.checkpoint_root.resolve().iterdir()
            if path.is_dir() and path.name not in DEFAULT_IGNORED_DIRECTORIES
        )
    )
    for directory in directories:
        metrics_path = directory / "loss_evolution_metrics.json"
        if not metrics_path.exists():
            continue
        records = json.loads(metrics_path.read_text())
        write_outputs(directory, records)
        print(f"Updated plots in {directory}")


if __name__ == "__main__":
    main()
