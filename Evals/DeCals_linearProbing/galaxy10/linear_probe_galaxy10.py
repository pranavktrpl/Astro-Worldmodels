import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

import timm
from torchvision import transforms


GALAXY10_CLASS_NAMES = [
    "disturbed",
    "merging",
    "round_smooth",
    "in_between_round_smooth",
    "cigar_shaped_smooth",
    "barred_spiral",
    "unbarred_tight_spiral",
    "unbarred_loose_spiral",
    "edge_on_without_bulge",
    "edge_on_with_bulge",
]


class Galaxy10H5Dataset(Dataset):
    """Lazy-opening HDF5 dataset so it works with DataLoader workers safely."""

    def __init__(self, h5_path: str, indices: np.ndarray, transform=None):
        self.h5_path = str(h5_path)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self._h5 = None
        self._images = None
        self._labels = None

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            self._images = self._h5["images"]
            self._labels = self._h5["ans"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i: int):
        self._ensure_open()
        idx = int(self.indices[i])
        image = self._images[idx]
        label = int(self._labels[idx])

        # Galaxy10_DECals.h5 stores HWC images.
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class LinearProbeHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class Standardize(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std.clamp_min(1e-6)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(path: str, device: torch.device):
    return torch.load(path, map_location=device)


def infer_model_name(ckpt: Dict, cli_model_name: Optional[str]) -> str:
    if cli_model_name:
        return cli_model_name
    cfg = ckpt.get("cfg", {})
    model_name = cfg.get("model_name")
    if model_name is None:
        raise ValueError(
            "Could not infer model_name from checkpoint['cfg']. "
            "Pass --model-name explicitly."
        )
    return model_name


def infer_input_size(ckpt: Dict, model, cli_input_size: Optional[int]) -> int:
    if cli_input_size is not None:
        return int(cli_input_size)

    cfg = ckpt.get("cfg", {})
    for key in ["global_size", "img_size", "image_size", "crop_size"]:
        if key in cfg and cfg[key] is not None:
            value = cfg[key]
            if isinstance(value, (list, tuple)):
                return int(value[-1])
            return int(value)

    data_cfg = timm.data.resolve_model_data_config(model)
    input_size = data_cfg.get("input_size", (3, 224, 224))
    return int(input_size[-1])


def build_backbone(model_name: str):
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,
        dynamic_img_size=True,
        dynamic_img_pad=True,
    )
    return model


def load_backbone_weights(backbone: nn.Module, ckpt: Dict):
    state = ckpt["model"]
    backbone_state = {}
    for key, value in state.items():
        if key.startswith("backbone."):
            backbone_state[key[len("backbone."):]] = value

    if not backbone_state:
        raise ValueError(
            "No 'backbone.' keys found in checkpoint['model']; "
            "this script expects the checkpoint format produced by your LeJEPA train.py."
        )

    missing, unexpected = backbone.load_state_dict(backbone_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Backbone state mismatch. missing={missing}, unexpected={unexpected}"
        )


def build_eval_transform(model, input_size: int):
    data_cfg = timm.data.resolve_model_data_config(model)
    mean = data_cfg.get("mean", (0.485, 0.456, 0.406))
    std = data_cfg.get("std", (0.229, 0.224, 0.225))

    return transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def read_labels(h5_path: str) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        labels = np.asarray(f["ans"][:], dtype=np.int64)
    return labels


def stratified_split_indices(
    labels: np.ndarray,
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-8:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []

    num_classes = int(labels.max()) + 1
    for c in range(num_classes):
        cls_idx = np.flatnonzero(labels == c)
        rng.shuffle(cls_idx)

        n = len(cls_idx)
        n_train = int(np.floor(train_frac * n))
        n_val = int(np.floor(val_frac * n))
        n_test = n - n_train - n_val

        # Keep all splits non-empty if possible.
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = max(1, n_test)
            while n_train + n_val + n_test > n:
                if n_train >= n_val and n_train >= n_test and n_train > 1:
                    n_train -= 1
                elif n_val >= n_test and n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    break
            while n_train + n_val + n_test < n:
                n_train += 1
        else:
            if n == 2:
                n_train, n_val, n_test = 1, 0, 1
            elif n == 1:
                n_train, n_val, n_test = 1, 0, 0

        train_idx.append(cls_idx[:n_train])
        val_idx.append(cls_idx[n_train:n_train + n_val])
        test_idx.append(cls_idx[n_train + n_val:n_train + n_val + n_test])

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


@torch.no_grad()
def extract_embeddings(
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
) -> Tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    all_embs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    use_amp = amp_dtype is not None and device.type == "cuda"
    autocast_ctx = torch.autocast if use_amp else None

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        if use_amp:
            with autocast_ctx(device_type="cuda", dtype=amp_dtype):
                embs = backbone(images)
        else:
            embs = backbone(images)

        all_embs.append(embs.float().cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_embs, dim=0), torch.cat(all_labels, dim=0)


def compute_standardizer(train_embs: torch.Tensor) -> Standardize:
    mean = train_embs.mean(dim=0, keepdim=True)
    std = train_embs.std(dim=0, keepdim=True)
    return Standardize(mean, std)


def make_feature_loader(
    embs: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = TensorDataset(embs, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def confusion_matrix_numpy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def metrics_from_confusion_matrix(cm: np.ndarray) -> Dict:
    total = cm.sum()
    correct = np.trace(cm)
    top1_acc = float(correct / total) if total > 0 else 0.0

    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    tp = np.diag(cm).astype(np.float64)

    per_class_acc = np.divide(
        tp,
        row_sums,
        out=np.zeros_like(tp, dtype=np.float64),
        where=row_sums > 0,
    )

    precision = np.divide(
        tp,
        col_sums,
        out=np.zeros_like(tp, dtype=np.float64),
        where=col_sums > 0,
    )
    recall = np.divide(
        tp,
        row_sums,
        out=np.zeros_like(tp, dtype=np.float64),
        where=row_sums > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    macro_f1 = float(f1.mean()) if len(f1) > 0 else 0.0

    return {
        "top1_acc": top1_acc,
        "macro_f1": macro_f1,
        "per_class_acc": per_class_acc.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support": row_sums.tolist(),
    }


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    standardizer: Standardize,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits = []
    all_labels = []
    standardizer = standardizer.to(device)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        x = standardizer(x)
        logits = model(x)
        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def evaluate_probe(
    model: nn.Module,
    standardizer: Standardize,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Dict:
    logits, labels = predict_logits(model, standardizer, loader, device)
    preds = logits.argmax(dim=1).numpy()
    labels_np = labels.numpy()
    cm = confusion_matrix_numpy(labels_np, preds, num_classes=num_classes)
    metrics = metrics_from_confusion_matrix(cm)
    metrics["confusion_matrix"] = cm.tolist()
    return metrics


def train_linear_probe(
    train_embs: torch.Tensor,
    train_labels: torch.Tensor,
    val_embs: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
    probe_epochs: int,
    probe_batch_size: int,
    lr_candidates: Sequence[float],
    wd_candidates: Sequence[float],
) -> Tuple[nn.Module, Standardize, Dict]:
    input_dim = train_embs.shape[1]
    standardizer = compute_standardizer(train_embs)

    train_loader = make_feature_loader(train_embs, train_labels, probe_batch_size, shuffle=True)
    val_loader = make_feature_loader(val_embs, val_labels, probe_batch_size, shuffle=False)

    best_overall = None

    for lr in lr_candidates:
        for wd in wd_candidates:
            model = LinearProbeHead(input_dim, num_classes).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            criterion = nn.CrossEntropyLoss()
            current_standardizer = Standardize(standardizer.mean.clone(), standardizer.std.clone()).to(device)

            best_state = None
            best_val = -float("inf")
            best_epoch = -1

            for epoch in range(probe_epochs):
                model.train()
                for x, y in train_loader:
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    x = current_standardizer(x)

                    logits = model(x)
                    loss = criterion(logits, y)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                val_metrics = evaluate_probe(
                    model=model,
                    standardizer=current_standardizer,
                    loader=val_loader,
                    device=device,
                    num_classes=num_classes,
                )
                val_score = val_metrics["macro_f1"]
                if val_score > best_val:
                    best_val = val_score
                    best_epoch = epoch
                    best_state = {
                        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                        "standardizer_mean": current_standardizer.mean.detach().cpu().clone(),
                        "standardizer_std": current_standardizer.std.detach().cpu().clone(),
                        "val_metrics": val_metrics,
                    }

            if best_state is None:
                raise RuntimeError("Probe training produced no best state.")

            candidate = {
                "lr": lr,
                "wd": wd,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_val,
                "state": best_state,
            }

            if best_overall is None or candidate["best_val_macro_f1"] > best_overall["best_val_macro_f1"]:
                best_overall = candidate

    best_model = LinearProbeHead(input_dim, num_classes)
    best_model.load_state_dict(best_overall["state"]["model"])
    best_model = best_model.to(device)

    best_standardizer = Standardize(
        best_overall["state"]["standardizer_mean"],
        best_overall["state"]["standardizer_std"],
    )

    best_info = {
        "lr": best_overall["lr"],
        "wd": best_overall["wd"],
        "best_epoch": best_overall["best_epoch"],
        "val_metrics": best_overall["state"]["val_metrics"],
    }
    return best_model, best_standardizer, best_info


def save_confusion_matrix_csv(cm: List[List[int]], out_path: Path):
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + GALAXY10_CLASS_NAMES)
        for class_name, row in zip(GALAXY10_CLASS_NAMES, cm):
            writer.writerow([class_name] + row)


def pretty_print_metrics(name: str, metrics: Dict):
    print(f"\n===== {name} =====")
    print(f"top1_acc  : {metrics['top1_acc']:.4f}")
    print(f"macro_f1  : {metrics['macro_f1']:.4f}")
    print("per_class_acc:")
    for idx, (class_name, acc, support) in enumerate(
        zip(GALAXY10_CLASS_NAMES, metrics["per_class_acc"], metrics["support"])
    ):
        print(f"  {idx:2d} {class_name:28s} acc={acc:.4f}  n={support}")
    print("confusion_matrix:")
    cm = np.array(metrics["confusion_matrix"], dtype=np.int64)
    print(cm)


def main():
    parser = argparse.ArgumentParser(description="Linear probe Galaxy10 DECaLS using frozen LeJEPA backbone embeddings.")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to LeJEPA checkpoint, e.g. step_20000.pt")
    parser.add_argument("--galaxy10-h5", type=str, required=True, help="Path to Galaxy10_DECals.h5")
    parser.add_argument("--out-dir", type=str, default="probe_outputs/galaxy10")
    parser.add_argument("--model-name", type=str, default=None, help="Optional override if checkpoint cfg is missing model_name")
    parser.add_argument("--input-size", type=int, default=None, help="Optional override for probe image size")
    parser.add_argument("--batch-size", type=int, default=128, help="Backbone embedding extraction batch size")
    parser.add_argument("--probe-batch-size", type=int, default=512, help="Linear probe batch size on frozen embeddings")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers for HDF5 image loading")
    parser.add_argument("--probe-epochs", type=int, default=100)
    parser.add_argument("--probe-lrs", type=float, nargs="+", default=[1e-2, 3e-3, 1e-3])
    parser.add_argument("--probe-wds", type=float, nargs="+", default=[0.0, 1e-4, 1e-3])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["none", "bf16", "fp16"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    ckpt = load_checkpoint(args.ckpt_path, device=device)
    model_name = infer_model_name(ckpt, args.model_name)

    backbone = build_backbone(model_name)
    load_backbone_weights(backbone, ckpt)
    input_size = infer_input_size(ckpt, backbone, args.input_size)
    transform = build_eval_transform(backbone, input_size)

    backbone = backbone.to(device)
    backbone.eval()

    amp_dtype = None
    if device.type == "cuda":
        if args.amp_dtype == "bf16":
            amp_dtype = torch.bfloat16
        elif args.amp_dtype == "fp16":
            amp_dtype = torch.float16

    labels = read_labels(args.galaxy10_h5)
    train_idx, val_idx, test_idx = stratified_split_indices(labels=labels, seed=args.seed)

    print(f"Loaded checkpoint: {args.ckpt_path}")
    print(f"Using model_name   : {model_name}")
    print(f"Using input_size   : {input_size}")
    print(f"Dataset size       : {len(labels)}")
    print(f"Split sizes        : train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    train_ds = Galaxy10H5Dataset(args.galaxy10_h5, train_idx, transform=transform)
    val_ds = Galaxy10H5Dataset(args.galaxy10_h5, val_idx, transform=transform)
    test_ds = Galaxy10H5Dataset(args.galaxy10_h5, test_idx, transform=transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    print("Extracting train embeddings...")
    train_embs, train_labels = extract_embeddings(backbone, train_loader, device, amp_dtype)
    print("Extracting val embeddings...")
    val_embs, val_labels = extract_embeddings(backbone, val_loader, device, amp_dtype)
    print("Extracting test embeddings...")
    test_embs, test_labels = extract_embeddings(backbone, test_loader, device, amp_dtype)

    print(f"Embedding dim      : {train_embs.shape[1]}")

    probe_model, standardizer, best_probe_info = train_linear_probe(
        train_embs=train_embs,
        train_labels=train_labels,
        val_embs=val_embs,
        val_labels=val_labels,
        num_classes=len(GALAXY10_CLASS_NAMES),
        device=device,
        probe_epochs=args.probe_epochs,
        probe_batch_size=args.probe_batch_size,
        lr_candidates=args.probe_lrs,
        wd_candidates=args.probe_wds,
    )

    train_feature_loader = make_feature_loader(train_embs, train_labels, args.probe_batch_size, shuffle=False)
    val_feature_loader = make_feature_loader(val_embs, val_labels, args.probe_batch_size, shuffle=False)
    test_feature_loader = make_feature_loader(test_embs, test_labels, args.probe_batch_size, shuffle=False)

    train_metrics = evaluate_probe(probe_model, standardizer, train_feature_loader, device, len(GALAXY10_CLASS_NAMES))
    val_metrics = evaluate_probe(probe_model, standardizer, val_feature_loader, device, len(GALAXY10_CLASS_NAMES))
    test_metrics = evaluate_probe(probe_model, standardizer, test_feature_loader, device, len(GALAXY10_CLASS_NAMES))

    pretty_print_metrics("TRAIN", train_metrics)
    pretty_print_metrics("VAL", val_metrics)
    pretty_print_metrics("TEST", test_metrics)

    results = {
        "checkpoint_path": args.ckpt_path,
        "galaxy10_h5": args.galaxy10_h5,
        "model_name": model_name,
        "input_size": input_size,
        "seed": args.seed,
        "split_sizes": {
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
        },
        "best_probe": best_probe_info,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with (out_dir / "metrics.json").open("w") as f:
        json.dump(results, f, indent=2)

    save_confusion_matrix_csv(test_metrics["confusion_matrix"], out_dir / "test_confusion_matrix.csv")

    torch.save(
        {
            "probe_model": probe_model.state_dict(),
            "standardizer_mean": standardizer.mean.cpu(),
            "standardizer_std": standardizer.std.cpu(),
            "best_probe_info": best_probe_info,
            "model_name": model_name,
            "input_size": input_size,
            "class_names": GALAXY10_CLASS_NAMES,
        },
        out_dir / "linear_probe.pt",
    )

    print(f"\nSaved metrics to          : {out_dir / 'metrics.json'}")
    print(f"Saved test confusion CSV : {out_dir / 'test_confusion_matrix.csv'}")
    print(f"Saved linear probe state : {out_dir / 'linear_probe.pt'}")


if __name__ == "__main__":
    main()
