# Baseline Train Runs Snapshot

Created on 2026-08-10 as a preservation point before starting the next round of
vision and spectra pre-training improvements.

This directory copies the current baseline training and loading code used for
the project runs completed so far. It is meant as a lightweight code snapshot,
not a copy of checkpoint tensors or datasets.

## Included Files

- `train-vision.py`: current galaxy-image LeJEPA training entry point.
- `train-spectra.py`: current DESI spectra LeJEPA training entry point.
- `data/dataloaders.py`: shared iterable dataset and DDP/worker sharding logic.
- `data/galaxies_source.py`: Hugging Face galaxy-image streaming source.
- `data/desiSpectra_source.py`: Hugging Face spectra streaming source.
- `data/AstroTransforms.py`: galaxy-image multi-crop, blur, and noise transforms.
- `data/SpectraTransforms.py`: spectra patchification, crop, and mask transforms.
- `configs/config.py`: active ViT-L image baseline configuration.
- `configs/config-vitL-galaxy-images.py`: saved ViT-L image config variant.
- `configs/train_resnet9.py`: historical ResNet9 baseline config.
- `models/resnet9.py`: shared MLP projector and historical ResNet9 backbone.

Future training changes should happen outside this snapshot.
