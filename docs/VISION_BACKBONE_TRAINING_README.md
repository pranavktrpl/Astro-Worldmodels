# Vision Backbone Training README

This document describes the galaxy-image LeJEPA training path used by
`train-vision.py`. It is intentionally implementation-level: the goal is that a
future run can be debugged from the exact tensor contracts and control flow,
not just from the high-level method name.

## What The Vision Pipeline Trains

The vision training script trains a self-supervised galaxy image backbone. It
does not use morphology labels during pretraining. Each source image is turned
into multiple galaxy-preserving views, those views pass through one shared
image encoder, and the resulting projected features are optimized with the
LeJEPA objective:

1. make all views of the same galaxy agree with the mean of the global views;
2. prevent representational collapse with a sliced Epps-Pulley Gaussian
   regularizer, called SIGReg in this repo.

The resulting backbone is the part intended for downstream use. The projector
is an objective head and is not the main downstream representation.

## Main Files

| File | Role |
|---|---|
| `train-vision.py` | Main DDP training script for galaxy image backbones. |
| `configs/config.py` | Active image training config used by `train-vision.py`. |
| `data/galaxies_source.py` | Hugging Face streaming source for `Smith42/galaxies`. |
| `data/dataloaders.py` | Iterable dataset, rank/worker sharding, multi-crop yield path. |
| `data/AstroTransforms.py` | Astronomy-specific image multi-crop augmentation code. |
| `models/resnet9.py` | Defines the shared `MLP` projector used by image and spectra runs. |
| `lejepa/` | Local LeJEPA/SIGReg implementation. |

## Data Source

The active image source is:

```python
dataset_name = "Smith42/galaxies"
columns = ["image_crop"]
split = "train"
streaming = True
```

The `GalaxiesSource` wrapper is intentionally tiny:

```python
load_dataset(dataset, columns=columns, split=split, streaming=True)
```

It does not shard or batch by itself. Sharding is handled inside
`MyDataset.__iter__`, because the correct shard count depends on both:

- number of DDP ranks / GPUs;
- number of PyTorch DataLoader workers per rank.

## DDP And Streaming Sharding

`MyDataset` is an `IterableDataset`. Every rank constructs its own instance.
Inside `__iter__`, the code computes:

```python
num_workers = worker_info.num_workers if worker_info else 1
worker_id = worker_info.id if worker_info else 0
num_shards = num_workers * world_size
shard_id = rank * num_workers + worker_id
shard_dataset = stream.shard(num_shards=num_shards, index=shard_id)
```

So with 4 GPUs and 1 worker per GPU:

```text
num_shards = 1 * 4 = 4
rank 0 -> shard 0
rank 1 -> shard 1
rank 2 -> shard 2
rank 3 -> shard 3
```

With 4 GPUs and 2 workers per GPU:

```text
num_shards = 2 * 4 = 8
rank 0 workers -> shards 0, 1
rank 1 workers -> shards 2, 3
rank 2 workers -> shards 4, 5
rank 3 workers -> shards 6, 7
```

This is the reason sharding does not live in `GalaxiesSource`: the data source
does not know how many DataLoader worker subprocesses will exist.

## Image Multi-Crop Transform

The image augmentation class is `AstroMultiCropTransform` in
`data/AstroTransforms.py`.

For each source image:

```python
image = image.convert("RGB")
global_crops = [global1, global2]
local_crops = [local_0, ..., local_{Vl-1}]
```

Default view counts:

```python
Vg = 2
Vl = 8
V = Vg + Vl = 10
```

### Global Crops

Both global crops use the same geometric transform:

```python
RandomResizedCrop(
    size=140,
    scale=(0.947, 0.947),
    ratio=(1.0, 1.0),
)
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)
RandomRotation(degrees=(0, 180))
```

Then the two global views get different observational perturbation strengths:

```python
global1: GaussianBlur(p=1.0) + GaussianNoise(p=1.0)
global2: GaussianBlur(p=0.1) + GaussianNoise(p=0.1)
```

The crop scale is currently a fixed value, not a range. The crop location,
orientation, blur draw, and noise draw are stochastic.

### Local Crops

Local crops use:

```python
RandomResizedCrop(
    size=56,
    scale=(0.394, 0.394),
    ratio=(1.0, 1.0),
)
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.5)
RandomRotation(degrees=(0, 180))
GaussianBlur(p=0.5)
GaussianNoise(p=0.5)
```

Again, the retained area fraction is fixed by the code, while crop placement
and other augmentations are random.

### Batch Tensor Contract

Each dataset sample yields:

```python
{
    "global_crops": Tensor[Vg, 3, 140, 140],
    "local_crops":  Tensor[Vl, 3, 56, 56] or None,
}
```

After PyTorch collation:

```text
global_crops: [B, Vg, 3, 140, 140]
local_crops:  [B, Vl, 3,  56,  56]
```

The training loop moves these to the local GPU and calls:

```python
emb, proj = model(global_crops, local_crops)
```

## Model Architecture

The image model is `TimmEncoder` inside `train-vision.py`.

```python
self.backbone = timm.create_model(
    model_name,
    pretrained=cfg.pretrained_backbone,
    num_classes=0,
    dynamic_img_size=True,
    dynamic_img_pad=True,
)
```

Important details:

- `num_classes=0` removes the classifier and returns pooled embeddings.
- `dynamic_img_size=True` lets the same ViT process both `140 x 140` global
  crops and `56 x 56` local crops.
- The backbone output dimension is inferred from `backbone.num_features`.

The projector is:

```python
MLP(
    in_channels=embed_dim,
    hidden_channels=[2 * embed_dim, 2 * embed_dim, proj_dim],
    norm_layer="batch_norm",
)
```

For the active ViT-L config:

```python
model_name = "vit_large_patch14_dinov2.lvd142m"
proj_dim = 64
pretrained_backbone = False
```

Despite the DINOv2 model name, the backbone is initialized from scratch unless
`pretrained_backbone=True`.

## View Encoding Contract

`TimmEncoder._encode_views` receives:

```text
x: [B, V, C, H, W]
```

It flattens the batch and view dimensions:

```python
flat = x.flatten(0, 1)         # [B*V, C, H, W]
flat_emb = backbone(flat)      # [B*V, D]
flat_proj = proj(flat_emb)     # [B*V, P]
```

Then it restores the view-first contract required by the LeJEPA loss:

```python
emb  = flat_emb.reshape(B, V, -1).transpose(0, 1)   # [V, B, D]
proj = flat_proj.reshape(B, V, -1).transpose(0, 1)  # [V, B, P]
```

The full forward pass encodes global views first, then local views, then
concatenates along the view dimension:

```text
emb:  [Vg + Vl, B, embed_dim]
proj: [Vg + Vl, B, proj_dim]
```

This shape is shared by the spectra model too, so the same loss function can be
used.

## LeJEPA Loss

The loss function is:

```python
loss, sim, sigreg = compute_lejepa_loss(
    proj=proj,
    sigreg_fn=sigreg_fn,
    lambd=cfg.lambd,
    num_global_views=cfg.Vg,
)
```

The projected tensor has shape:

```text
proj: [V, B, P]
```

The first `Vg` views are global:

```python
global_proj = proj[:num_global_views]  # [Vg, B, P]
centers = global_proj.mean(0)          # [B, P]
```

The similarity term moves every view toward the global-view center:

```python
sim = (centers.unsqueeze(0) - proj).square().mean()
```

The regularizer applies SIGReg independently to every view:

```python
sigreg = torch.stack([sigreg_fn(proj[v]) for v in range(proj.size(0))]).mean()
```

Final loss:

```python
loss = (1 - lambd) * sim + lambd * sigreg
```

With the active config:

```python
lambd = 0.05
sigreg_num_points = 17
sigreg_num_slices = 1024
```

SIGReg internally uses distributed collectives. This means all ranks must enter
the loss in the same order with compatible tensors.

## Optimizer, Scheduler, And AMP

Optimizer:

```python
AdamW(
    model.parameters(),
    lr=5e-4,
    weight_decay=5e-2,
)
```

Scheduler:

1. `LinearLR` warmup for `warmup_steps=1000`, starting at 1 percent of LR.
2. `CosineAnnealingLR` from warmup end to `total_steps`, ending at `min_lr`.
3. Combined through `SequentialLR`.

AMP:

```python
amp_dtype = torch.bfloat16
GradScaler(enabled=False)
```

The model forward and loss are inside:

```python
with autocast("cuda", dtype=amp_dtype):
    ...
```

## Step Accounting

The image config currently computes:

```python
steps_per_epoch = cfg.safe_samples_per_rank_per_epoch // cfg.bs
total_steps = cfg.epochs * cfg.steps_per_epoch
```

With the active image config:

```python
bs = 192
epochs = 5
safe_samples_per_rank_per_epoch = 21976 * 96
```

So:

```text
safe_samples_per_rank_per_epoch = 2,109,696
steps_per_epoch = 2,109,696 // 192 = 10,988
total_steps = 5 * 10,988 = 54,940
```

Global examples per optimizer step are:

```text
global_batch = bs * world_size
```

For 4 GPUs:

```text
global_batch = 192 * 4 = 768 source images per step
```

Each source image yields 10 views, so per optimizer step the backbone processes:

```text
768 source images * 10 views = 7,680 image crops
```

The `safe_samples_per_rank_per_epoch` name matters: it is not the full dataset
size. It is the planned number of source examples each rank should consume per
epoch before the training loop stops.

## DDP Setup

Launch command:

```bash
cd /mnt/ssd-cluster/pranav/Astro-Worldmodels
source /mnt/ssd-cluster/pranav/.cluster/bashrc

CUDA_VISIBLE_DEVICES=0,1,2,3 \
/mnt/ssd-cluster/pranav/conda/envs/lejepa-og/bin/torchrun \
  --standalone \
  --nproc_per_node=4 \
  train-vision.py
```

DDP setup:

```python
dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
```

The 60-minute timeout is a guardrail around slow/stuck collectives. It is not a
fix for rank desynchronization. If one rank exits the dataloader and another
rank enters SIGReg, NCCL will still eventually time out.

## Checkpointing

Checkpoints contain:

```python
{
    "model": model_state_dict,
    "optimizer": optimizer_state_dict,
    "scheduler": scheduler_state_dict,
    "scaler": scaler_state_dict,
    "epoch": epoch,
    "global_step": global_step,
    "cfg": asdict(cfg),
    "python_rng": random.getstate(),
    "torch_rng": torch.get_rng_state(),
    "cuda_rng": torch.cuda.get_rng_state_all(),
}
```

Periodic checkpoints are written every:

```python
ckpt_every = 4000
```

plus one `last_epoch_{epoch}.pt` checkpoint at epoch end and `complete.pt` at
the end of training.

Resume path is controlled by:

```python
cfg.resume_path
```

For image training, the checked-in active config resumes from:

```text
./checkpoints/VitLargePatch14_OfficialTrain5_Epoch5_2504/step_12000.pt
```

Change this to `None` for a fresh run.

## Logging And Collapse Monitoring

W&B is initialized only on rank 0. Logged quantities include:

- `train/loss`
- `train/sim`
- `train/sigreg`
- `train/emb_mean_std`
- `train/emb_min_std`
- `train/proj_mean_std`
- `train/proj_min_std`
- `train/lr`
- `train/epoch`
- `train/step`

The collapse stats are computed by flattening `[V, B, D]` into `[B*V, D]` and
measuring per-dimension standard deviation. If `emb_min_std` or
`proj_min_std` falls near zero for many dimensions, the representation is
collapsing or becoming low-rank.

## Known Sharp Edges

1. **HF streaming shard imbalance.** Iterable HF sharding is not guaranteed to
   produce exactly equal rank lengths. `drop_last=True` only drops incomplete
   batches within a rank; it does not force all ranks to have the same number
   of batches.
2. **SIGReg uses collectives.** If ranks desynchronize before the loss, the
   process can hang in an all-reduce until the NCCL timeout.
3. **Image crop comments are slightly stale.** Some comments mention `144` and
   `60`, but the active transform sizes are `140` and `56`.
4. **The crop scale values are fixed.** The image code currently uses
   `scale=(0.947, 0.947)` and `scale=(0.394, 0.394)`, so crop size is fixed
   while location/orientation/noise remain stochastic.
5. **Training is intentionally stream-based.** The run does not materialize the
   image dataset locally. This saves disk but makes the training loop sensitive
   to network and HF streaming behavior.

## How To Read A Vision Checkpoint

The downstream feature encoder is the backbone inside the checkpoint:

```python
state = torch.load(path, map_location="cpu")
model_state = state["model"]
```

For downstream work, use the backbone embedding, not the projector output. The
projector is trained to support LeJEPA/SIGReg and can be useful diagnostically,
but the backbone is the intended reusable representation.

