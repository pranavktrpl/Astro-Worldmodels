# JEPA versus CLIP Study

This directory is a self-contained analysis bundle comparing the completed
307K-pair scratch JEPA+SIGReg and scratch CLIP representations.

Read [`STUDY_REPORT.md`](STUDY_REPORT.md) for the complete scientific result.

## Reproduction

Use the project environment:

```bash
source /mnt/ssd-cluster/pranav/.cluster/bashrc
conda activate /mnt/ssd-cluster/pranav/conda/envs/lejepa-og
cd /mnt/ssd-cluster/pranav/Astro-Worldmodels
```

Prepare the fixed manifest and labels:

```bash
python "jepa vs clip/study.py" prepare
```

The remaining subcommands are:

```text
retrieval --model {jepa,clip}
neighbors --space {J_I,J_S,C_I,C_S}
geometry
mappings --model {jepa,clip}
probes --space {J_I,J_S,C_I,C_S}
decoder-transfer --model {jepa,clip}
rank-geometry --space {J_I,J_S,C_I,C_S}
```

After the result JSON files exist, regenerate tables and figures with:

```bash
python "jepa vs clip/make_figures.py"
```

GPU-heavy subcommands accept `--device`. Independent models/spaces can be run
in parallel. Source embeddings and datasets are never modified.
