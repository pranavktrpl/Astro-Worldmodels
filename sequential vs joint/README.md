# Sequential versus Joint JEPA Study

This directory is a self-contained analysis of the completed frozen-backbone
sequential alignment and from-scratch joint cross-modal JEPA runs.

Start with [STUDY_REPORT.md](STUDY_REPORT.md). It contains the full experimental
definition, results, interpretation, limitations, and paper-oriented next steps.

## Contents

- `study.py`: validated retrieval, geometry, rank, mapping, and probe routines.
- `analysis.py`: equal-capacity shared/private and retention analysis.
- `morphology.py`: Galaxy10 private-information experiment.
- `make_outputs.py`: reproducible CSV and figure generation.
- `results/`: machine-readable metrics.
- `tables/`: compact CSV summaries.
- `figures/`: report figures.
- `artifacts/`: labels, exact neighbors, PCA fits, manifest, and morphology embeddings.

Original checkpoints and the large 168,280-row embedding caches remain outside
this directory and are only read by path. No training weights are modified.

## Environment

All evaluations were run with:

```bash
source /mnt/ssd-cluster/pranav/.cluster/bashrc
conda activate /mnt/ssd-cluster/pranav/conda/envs/lejepa-og
```

The fixed-checkpoint analysis is complete. Experiments listed in report section
12 require new training and are intentionally identified as future work.

