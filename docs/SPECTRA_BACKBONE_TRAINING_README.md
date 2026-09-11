# Spectra Backbone Training README

> Historical v1 document. The preserved implementation is under
> `baseline_train_runs_2026-08-10/`. The live DESI v2 entry point is
> `train-spectra-v2.py`; its design and implementation reference are maintained
> in `docs/SPECTRA_V2_DECISION_LOG.md`.

This document describes the DESI spectra LeJEPA training path built around
`train-spectra.py`. It is the spectra analogue of the galaxy image training
pipeline, but the data representation, crop mechanism, and encoder are
different enough that they deserve their own precise notes.

## What The Spectra Pipeline Trains

The spectra pipeline trains a self-supervised backbone over DESI galaxy spectra.
Each input spectrum is converted into an ordered sequence of fixed-length flux
patches. Multi-view training is created by keeping contiguous wavelength
regions and replacing the rest of the patch positions with PAD patches. A
Transformer encoder maps every cropped view into an embedding, and the same
LeJEPA loss used for images trains the model.

The model is not a classifier. It does not predict redshift or morphology
during pretraining. It learns a reusable spectral representation from unlabeled
flux sequences.

## Main Files

| File | Role |
|---|---|
| `train-spectra.py` | Main DDP training script for spectra. |
| `data/desiSpectra_source.py` | Hugging Face streaming wrapper for DESI-like spectra datasets. |
| `data/dataloaders.py` | Adds `DesiSpectraDataset`, DDP/worker sharding, and sample transform calls. |
| `data/SpectraTransforms.py` | Lightweight spectra patchification and crop/PAD transform. |
| `models/resnet9.py` | Provides the shared `MLP` projector. |
| `lejepa/` | Local LeJEPA/SIGReg implementation. |

## Current Dataset

The active spectra config in `train-spectra.py` points at:

```python
spectra_dataset_name = "UniverseTBD/mmu_desi_edr_sv3"
train_num_spectra = 1126441
```

This replaced the smaller initial source:

```python
"MultimodalUniverse/desi"  # 100,000 rows
```

The current dataset has:

```text
split: train
rows:  1,126,441
```

Actual parquet samples were checked directly. For each tested row:

```text
spectrum["flux"]      length 7781
spectrum["ivar"]      length 7781
spectrum["lsf_sigma"] length 7781
spectrum["lambda"]    length 7781
spectrum["mask"]      length 7781
```

The training pipeline uses only:

```python
spectrum["flux"]
```

for now. The other arrays are scientifically meaningful but not part of the
current v1 pretraining input.

## Why Only Flux Is Used

The spectrum object contains:

- `flux`: measured spectral intensity per wavelength bin;
- `lambda`: wavelength coordinate grid;
- `ivar`: inverse variance, a per-bin reliability/weight;
- `mask`: bad-pixel or unusable-bin flags;
- `lsf_sigma`: line-spread-function width / instrumental resolution context.

We checked that `lambda` is the same across sampled rows, so it is a fixed
coordinate grid rather than sample-specific signal. For the first clean spectra
backbone, the input is therefore the flux signal only. The model gets patch
positions through learned positional embeddings.

## Data Source Wrapper

`DesiSpectraSource` is deliberately thin:

```python
dataset = load_dataset(
    self.dataset,
    columns=self.columns,
    split=self.split,
    streaming=True,
)
```

As with image training, the source does not shard or batch. DDP rank sharding is
handled later inside the iterable dataset because it depends on rank count and
DataLoader worker count.

The default source still says `"MultimodalUniverse/desi"` in
`data/desiSpectra_source.py`, but `train-spectra.py` overrides it through
`SpectraTrainConfig.spectra_dataset_name`.

## Spectra Dataset And DDP Sharding

`DesiSpectraDataset` in `data/dataloaders.py` is an `IterableDataset`.

Inside `__iter__`, it loads the HF stream, optionally shuffles, sets the epoch,
and computes:

```python
num_workers = worker_info.num_workers if worker_info else 1
worker_id = worker_info.id if worker_info else 0
num_shards = num_workers * world_size
shard_id = rank * num_workers + worker_id
shard_dataset = stream.shard(num_shards=num_shards, index=shard_id)
```

For 4 GPUs and 1 worker per GPU:

```text
rank 0 -> shard 0
rank 1 -> shard 1
rank 2 -> shard 2
rank 3 -> shard 3
```

For 4 GPUs and 2 workers per GPU:

```text
rank 0 -> shards 0, 1
rank 1 -> shards 2, 3
rank 2 -> shards 4, 5
rank 3 -> shards 6, 7
```

Each row is converted by:

```python
flux = np.asarray(sample["spectrum"]["flux"], dtype=np.float32)[None, :]
```

The extra singleton channel is harmless because the transform squeezes the
input before patchifying.

## Patchification

Every DESI flux array has length:

```text
7781
```

The spectra transform uses:

```python
patch_size = 20
```

Patchification does:

```text
7781 values
-> use floor division by 20
-> 389 full patches
-> 389 * 20 = 7780 covered values
-> final 1 flux value is dropped
```

The base ordered representation is:

```text
spectrum_patches: [389, 20]
```

This order is sacred. The model uses learned absolute positional embeddings,
so patch positions must not be compacted or shuffled.

## Spectra Crops

Spectra crops are not pixel crops. They are patch-level contiguous keep masks.

The transform samples a retained fraction, selects one contiguous span of patch
indices, and replaces all other patch positions with a PAD vector.

Current config:

```python
spectra_global_scale = (0.90, 1.0)
spectra_local_scale  = (0.35, 0.50)
```

So each global crop keeps a random contiguous region covering 90 to 100 percent
of the 389 patch positions. Each local crop keeps 35 to 50 percent.

Examples:

```text
global retained patches: 371, 388, 369, ...
local retained patches:  140, 111, 84, 107, ...
```

The crop shape never shrinks:

```text
global_crops: [Vg, 389, 20]
local_crops:  [Vl, 389, 20]
```

Only the values at cropped-out patch positions are replaced.

## PAD Is Not Zero Detection

Cropped patch positions are numerically filled with:

```python
pad_value = 0.0
```

But zero can be a real flux value. Therefore the model must never infer padding
from the crop values.

The transform also returns masks:

```text
global_masks: [Vg, 389]
local_masks:  [Vl, 389]
```

Mask convention:

```text
1 / True  = real retained patch
0 / False = cropped-out PAD patch
```

After batching:

```text
global_crops: [B, Vg, 389, 20]
local_crops:  [B, Vl, 389, 20]
global_masks: [B, Vg, 389]
local_masks:  [B, Vl, 389]
```

The model receives masks explicitly. PAD handling is mask-based, not
value-based.

## Spectrum Transformer Encoder

The spectra model is `SpectrumTransformerEncoder` in `train-spectra.py`.

Current config:

```python
spectra_embed_dim = 768
spectra_depth = 12
spectra_num_heads = 12
spectra_mlp_ratio = 4.0
spectra_dropout = 0.0
spectra_pooling = "cls"
proj_dim = 64
```

Parameter count printed in the run:

```text
spectra encoder params: 85,373,184
projector params:       3,646,528
total params:           89,019,712
```

### Patch Embedding

Every patch has shape:

```text
[20]
```

The patch embedding layer is:

```python
nn.Linear(20, embed_dim)
```

so patch tokens become:

```text
[B*V, 389, embed_dim]
```

### Learned PAD Token

The model owns:

```python
self.pad_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
```

After patch embedding:

```python
tokens = torch.where(flat_masks.unsqueeze(-1), tokens, pad_tokens)
```

This means:

- real patch positions keep their flux-derived embedding;
- cropped positions get the learned PAD token;
- zero-valued real flux patches remain real if `mask=True`.

### CLS Token And Position Embeddings

The model prepends a CLS token:

```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
```

The full token sequence is:

```text
[CLS] + 389 patch tokens = 390 tokens
```

Therefore position embeddings have shape:

```python
self.pos_embed = nn.Parameter(torch.zeros(1, 390, embed_dim))
```

CLS is never padding.

### Transformer Encoder

The encoder uses PyTorch's built-in transformer:

```python
nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=num_heads,
    dim_feedforward=int(embed_dim * spectra_mlp_ratio),
    activation="gelu",
    batch_first=True,
    norm_first=True,
)
```

With the current values:

```text
dim_feedforward = 768 * 4 = 3072
```

So each block has a feed-forward MLP of:

```text
768 -> 3072 -> 768
```

No causal mask is used. This is an encoder, not an autoregressive model.

### Attention Padding Mask

PyTorch expects:

```text
src_key_padding_mask=True means ignore this token
```

Our crop masks use:

```text
True means real patch
```

So the model builds:

```python
token_mask = [CLS mask] + flat_masks
key_padding_mask = ~token_mask
```

This is the key correctness point. PAD tokens are ignored by attention.

### Pooling

Default pooling:

```python
emb = hidden[:, 0]
```

This is CLS pooling. The model can also use:

```python
spectra_pooling = "masked_mean"
```

Masked mean pooling averages only real patch tokens:

```python
weights = token_mask[:, 1:]
emb = (patch_hidden * weights).sum(dim=1) / weights.sum(dim=1)
```

PAD tokens never contribute to masked mean pooling.

### Projector

The spectra model reuses the same projector style as image training:

```python
MLP(
    in_channels=embed_dim,
    hidden_channels=[2 * embed_dim, 2 * embed_dim, proj_dim],
    norm_layer="batch_norm",
)
```

For `embed_dim=768` and `proj_dim=64`:

```text
768 -> 1536 -> 1536 -> 64
```

## Output Contract

`SpectrumTransformerEncoder.forward` returns the same contract as
`TimmEncoder`:

```text
emb:  [Vg + Vl, B, embed_dim]
proj: [Vg + Vl, B, proj_dim]
```

This lets spectra training reuse the exact same loss function:

```python
loss, sim, sigreg = compute_lejepa_loss(
    proj=proj,
    sigreg_fn=sigreg_fn,
    lambd=cfg.lambd,
    num_global_views=cfg.Vg,
)
```

The model smoke test confirmed:

```text
emb  (5, 2, 32)
proj (5, 2, 7)
```

for a toy model with `Vg=2`, `Vl=3`, `B=2`, `embed_dim=32`, and `proj_dim=7`.

## LeJEPA Loss

The spectra run uses the same loss as image training.

Given:

```text
proj: [V, B, P]
```

the first `Vg` projections are the global crops:

```python
global_proj = proj[:num_global_views]
centers = global_proj.mean(0)
```

All views are moved toward the global center:

```python
sim = (centers.unsqueeze(0) - proj).square().mean()
```

SIGReg is applied view by view:

```python
sigreg = torch.stack([sigreg_fn(proj[v]) for v in range(proj.size(0))]).mean()
```

Final loss:

```python
loss = (1 - lambd) * sim + lambd * sigreg
```

Current regularization settings:

```python
lambd = 0.05
sigreg_num_points = 17
sigreg_num_slices = 1024
```

SIGReg performs distributed all-reduces internally, so every rank must enter
the loss in the same order. This is why coordinated early-stop is needed.

## Step Accounting

Current spectra config:

```python
train_num_spectra = 1126441
world_size = 4
bs = 16
epochs = 5
```

The script computes:

```python
safe_samples_per_rank_per_epoch = train_num_spectra // world_size
steps_per_epoch = safe_samples_per_rank_per_epoch // bs
total_steps = epochs * steps_per_epoch
```

Numerically:

```text
safe_samples_per_rank_per_epoch = 1,126,441 // 4 = 281,610
steps_per_epoch = 281,610 // 16 = 17,600
total_steps = 5 * 17,600 = 88,000
```

Global batch per optimizer step:

```text
global_batch = bs * world_size = 16 * 4 = 64 spectra
```

Planned spectra per epoch:

```text
17,600 steps * 64 spectra/step = 1,126,400 spectra
```

This is almost exactly one pass over the dataset, with 41 examples left over by
integer division.

Important: `steps_per_epoch` is the planned maximum. HF streaming shards can be
uneven, so the actual epoch can stop earlier if one rank exhausts its shard.

## Coordinated Early Stop

HF streaming over many parquet files does not guarantee every DDP rank has
exactly the same number of batches. `drop_last=True` only drops incomplete
batches within each rank; it does not make all ranks end together.

The failure mode we saw was:

1. one rank exhausted its stream shard or moved toward epoch-end;
2. other ranks still had a batch and entered `compute_lejepa_loss`;
3. SIGReg called distributed `all_reduce`;
4. not all ranks entered the same collective;
5. NCCL timed out after 60 minutes.

The training loop now uses an explicit iterator:

```python
train_iter = iter(train_loader)

for step in range(cfg.steps_per_epoch):
    try:
        batch = next(train_iter)
        has_batch = torch.tensor(1, device=device)
    except StopIteration:
        batch = None
        has_batch = torch.tensor(0, device=device)

    dist.all_reduce(has_batch, op=dist.ReduceOp.MIN)

    if has_batch.item() == 0:
        break

    # safe to run model/loss/backward
```

If any rank has no next batch, the `MIN` reduction becomes zero and every rank
breaks before forward/loss collectives. This is the correct fix for uneven HF
stream shard tails.

## Optimizer, Scheduler, AMP, DDP

The spectra script intentionally keeps the same training regimen as the vision
script:

```python
AdamW(lr=5e-4, weight_decay=5e-2)
LinearLR warmup for 1000 steps
CosineAnnealingLR to min_lr=1e-6
amp_dtype = "bf16"
```

DDP:

```python
dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
torch.cuda.set_device(local_rank)
model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    broadcast_buffers=False,
    gradient_as_bucket_view=True,
    static_graph=True,
)
```

`static_graph=True` is reasonable here because:

- `Vg` and `Vl` are fixed;
- every crop tensor keeps full `[389, 20]` shape;
- PAD changes values/masks, not tensor shapes.

## Launch Command

From repo root:

```bash
cd /mnt/ssd-cluster/pranav/Astro-Worldmodels
source /mnt/ssd-cluster/pranav/.cluster/bashrc

CUDA_VISIBLE_DEVICES=0,1,2,3 \
/mnt/ssd-cluster/pranav/conda/envs/lejepa-og/bin/torchrun \
  --standalone \
  --nproc_per_node=4 \
  train-spectra.py
```

The `.cluster/bashrc` sets persistent HF/W&B cache paths and loads `.env`.
`HF_TOKEN` is already present in `.env`; the recent NCCL failure was not an auth
problem. It was rank desynchronization at the streamed dataset tail.

## Checkpointing

The checkpoint payload is the same as image training:

```python
{
    "model": ...,
    "optimizer": ...,
    "scheduler": ...,
    "scaler": ...,
    "epoch": epoch,
    "global_step": global_step,
    "cfg": asdict(cfg),
    "python_rng": ...,
    "torch_rng": ...,
    "cuda_rng": ...,
}
```

Current spectra output path:

```python
save_dir = "./checkpoints/SPECTRA_run4_bs16_2806_Epoch5_utbd_desi"
```

Current run name:

```python
run_name = "SPECTRA_run4_bs16_2806_Epoch5_utbd_desi"
```

Change `resume_path` from `None` to a checkpoint path to resume.

## Runtime Failure Modes And Fixes

### 1. NCCL timeout inside SIGReg all-reduce

Symptom:

```text
Watchdog caught collective operation timeout
all_reduce from lejepa/univariate/epps_pulley.py
compute_lejepa_loss
```

Cause:

```text
Ranks entered different collectives because one streaming shard ended before another.
```

Fix:

```text
Coordinated early-stop before forward/loss. Implemented in train-spectra.py.
```

### 2. Slow HF streaming initialization

The large dataset resolves hundreds of parquet files. Startup may print several
`Resolving data files` bars. That is normal.

### 3. Jagged loss

Spectra training can be noisier than image training because:

- the dataset is smaller than the image stream;
- global batch is currently `64`, not hundreds;
- crops can expose very different wavelength regions;
- raw flux scales can vary widely.

Possible stabilizers:

- robust per-spectrum flux normalization;
- larger per-GPU batch size if memory allows;
- `spectra_pooling = "masked_mean"`;
- lower LR, e.g. `2e-4`;
- gradient clipping.

These are not currently part of the main script unless explicitly added.

## Current Open Design Choices

1. **Flux normalization.** The current input is raw flux. This is simple but
   may make training noisier. A robust median/IQR normalization per spectrum is
   a strong candidate for the next iteration.
2. **Use of `ivar` and `mask`.** The current model ignores DESI inverse
   variance and bad-bin masks. A future version could use them as auxiliary
   channels or attention weights.
3. **PAD token versus zero patch.** The dataloader emits zero PAD patches for a
   stable tensor representation, but the model replaces padded positions using
   the explicit mask and a learned `pad_token`.
4. **Pooling.** CLS pooling is the default. Masked mean pooling may stabilize
   early training.
5. **Epoch exactness.** Because HF streaming shards can be uneven, one epoch is
   best understood as "up to the planned number of synchronized DDP steps",
   not a mathematically exact full pass.
