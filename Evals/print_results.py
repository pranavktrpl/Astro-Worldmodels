#!/usr/bin/env python
"""Print a compact test-R2 summary of every redshift-probe result on disk.

    python Evals/print_results.py            # all results
    python Evals/print_results.py adapted    # only labels containing 'adapted'
"""

import glob
import json
import sys
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent


def main() -> None:
    needle = sys.argv[1] if len(sys.argv) > 1 else ""
    pattern = str(EVALS_DIR / "*" / "results" / "*" / "metrics.json")
    matched = False
    for path in sorted(glob.glob(pattern)):
        if needle and needle not in path:
            continue
        data = json.loads(Path(path).read_text())
        summary = data.get("summary")
        if not summary or "ridge" not in summary:
            continue  # not a redshift-probe result (e.g. matched-heads)
        matched = True
        suite = Path(path).parts[-4]
        label = Path(path).parent.name
        print(f"{suite} / {label}")
        for head in ("ridge", "knn", "mlp"):
            stats = summary[head]["test"]["r2"]
            line = f"  {head:5s} test R2 {stats['mean']:.4f} ± {stats['std']:.4f}"
            clipped = summary[head].get("test_clipped", {}).get("r2")
            if clipped:
                line += f" | clipped {clipped['mean']:.4f} ± {clipped['std']:.4f}"
            print(line)
    if not matched:
        print(f"no redshift-probe metrics.json found matching '{needle}'")


if __name__ == "__main__":
    main()
