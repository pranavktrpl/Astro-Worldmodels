#!/usr/bin/env python3
"""Shared/private and representation-retention analysis."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
import study

ROOT = Path(__file__).resolve().parent
A, R = ROOT / "artifacts", ROOT / "results"
PROPERTIES = ("stellar_mass", "metallicity", "age", "ssfr")
STATES = {
    "U_I": ("S_I", "raw"), "U_S": ("S_S", "raw"),
    "S_I": ("S_I", "projected"), "S_S": ("S_S", "projected"),
    "J_I_raw": ("J_I", "raw"), "J_S_raw": ("J_S", "raw"),
    "J_I": ("J_I", "projected"), "J_S": ("J_S", "projected"),
}
REGIMES = {
    "independent": ("U_I", "U_S"),
    "sequential": ("S_I", "S_S"),
    "joint": ("J_I", "J_S"),
}
L2 = (1e-4, 1e-2, 1.0, 1e2, 1e4)


def raw(name):
    root, kind = STATES[name]
    return study.embedding(root, kind)


def pca(name, device):
    labels = np.load(A / "labels.npz")
    idx = np.flatnonzero(~labels["is_test"])
    x = np.asarray(raw(name)[idx], dtype=np.float32)
    mean = x.mean(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(x.std(0, dtype=np.float64).astype(np.float32), 1e-6)
    cov = torch.zeros((x.shape[1], x.shape[1]), device=device)
    for start in range(0, len(x), 4096):
        z = torch.from_numpy((x[start:start + 4096] - mean) / std).to(device)
        cov += z.T @ z
    eig, vec = torch.linalg.eigh(cov / (len(x) - 1))
    np.savez_compressed(A / f"pca_{name}.npz", mean=mean, std=std,
                        basis=vec[:, -256:].cpu().numpy(),
                        eigenvalues=eig.cpu().numpy(), fit_rows=len(idx))
    print(f"{name}: {x.shape[1]} -> 256 PCA", flush=True)


def values(name, idx):
    x = np.asarray(raw(name)[idx], dtype=np.float32)
    if name in ("U_I", "U_S"):
        fit = np.load(A / f"pca_{name}.npz")
        x = ((x - fit["mean"]) / fit["std"]) @ fit["basis"]
    return x


def standardize(fit, *rest):
    mean = fit.mean(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(fit.std(0, dtype=np.float64).astype(np.float32), 1e-6)
    return tuple((x - mean) / std for x in (fit,) + rest)


def r2(pred, target):
    return float(1 - np.square(pred - target).sum() /
                 max(np.square(target - target.mean(0)).sum(), 1e-20))


def choose_map(xfit, yfit, xval, yval, device):
    x, y = torch.from_numpy(xfit).to(device), torch.from_numpy(yfit).to(device)
    xv = torch.from_numpy(xval).to(device)
    trials, best = [], None
    for l2 in L2:
        gram = x.T @ x
        gram.diagonal().add_(l2)
        w = torch.linalg.solve(gram, x.T @ y)
        score = r2((xv @ w).cpu().numpy(), yval)
        trials.append({"l2": l2, "validation_global_r2": score})
        if best is None or score > best[0]:
            best = (score, l2)
    return best[1], trials


def refit(x, y, l2, device):
    x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    gram = x.T @ x
    gram.diagonal().add_(l2)
    return torch.linalg.solve(gram, x.T @ y).cpu().numpy()


def probes(train_x, test_x, train_idx, test_idx, labels, device):
    targets = {"redshift": labels["redshift"]}
    targets.update({n: labels["properties"][:, i] for i, n in enumerate(PROPERTIES)})
    output = {}
    for name, target in targets.items():
        tr, te = np.isfinite(target[train_idx]), np.isfinite(target[test_idx])
        output[name] = study.ridge_probe(
            train_x[tr], target[train_idx][tr],
            test_x[te], target[test_idx][te], device)
    return output


def decompose(regime, device):
    labels = np.load(A / "labels.npz")
    train = np.flatnonzero(~labels["is_test"])
    order = np.random.default_rng(2026).permutation(train)
    nfit, nval = int(.70 * len(order)), int(.10 * len(order))
    idx = [order[:nfit], order[nfit:nfit+nval], order[nfit+nval:],
           np.flatnonzero(labels["is_test"])]
    ni, ns = REGIMES[regime]
    image = standardize(*[values(ni, i) for i in idx])
    spectrum = standardize(*[values(ns, i) for i in idx])
    output = {
        "regime": regime, "states": [ni, ns],
        "definition": "shared(target)=ridge prediction from the other modality; private(target)=target-shared(target)",
        "warning": "Operational linear-predictability decomposition, not a unique information-theoretic split.",
        "split": dict(map_fit=len(idx[0]), map_validation=len(idx[1]),
                      probe_train=len(idx[2]), final_test=len(idx[3])),
        "directions": {}
    }
    for direction, source, target in (
        ("image_to_spectrum", image, spectrum),
        ("spectrum_to_image", spectrum, image),
    ):
        selected, trials = choose_map(source[0], target[0], source[1], target[1], device)
        w = refit(np.concatenate(source[:2]), np.concatenate(target[:2]), selected, device)
        shared_train, shared_test = source[2] @ w, source[3] @ w
        private_train, private_test = target[2] - shared_train, target[3] - shared_test
        output["directions"][direction] = {
            "selected_l2": selected, "candidates": trials,
            "test_global_r2": r2(shared_test, target[3]),
            "residual_variance_fraction": float(np.var(private_test) / np.var(target[3])),
            "probes": {
                "total": probes(target[2], target[3], idx[2], idx[3], labels, device),
                "shared": probes(shared_train, shared_test, idx[2], idx[3], labels, device),
                "private": probes(private_train, private_test, idx[2], idx[3], labels, device),
            },
        }
        print(regime, direction, flush=True)
    output["cca"] = study.cca_summary(
        np.concatenate(image[:2]).astype(np.float64),
        np.concatenate(spectrum[:2]).astype(np.float64),
        image[3], spectrum[3])
    study.atomic_json(R / f"decomposition_{regime}.json", output)


@torch.inference_mode()
def neighbors(name, device):
    labels = np.load(A / "labels.npz")
    idx = np.flatnonzero(labels["is_test"])
    x = torch.from_numpy(values(name, idx)).to(device)
    x = torch.nn.functional.normalize(x, dim=1)
    result = np.empty((len(x), 100), dtype=np.int32)
    for start in range(0, len(x), 1024):
        stop = min(len(x), start + 1024)
        score = x[start:stop] @ x.T
        score[torch.arange(stop-start, device=device),
              torch.arange(start, stop, device=device)] = -float("inf")
        result[start:stop] = torch.topk(score, k=100, dim=1).indices.cpu().numpy()
    np.save(A / f"neighbors_{name}_test.npy", result)
    print("neighbors", name, flush=True)


def cka(x, y):
    x, y = x - x.mean(0), y - y.mean(0)
    return float(np.square(x.T @ y).sum() /
                 np.sqrt(np.square(x.T @ x).sum() * np.square(y.T @ y).sum()))


def retention():
    labels = np.load(A / "labels.npz")
    test = np.flatnonzero(labels["is_test"])
    pairs = (
        ("independent_to_sequential_image", "U_I", "S_I"),
        ("independent_to_sequential_spectrum", "U_S", "S_S"),
        ("independent_to_joint_raw_image", "U_I", "J_I_raw"),
        ("independent_to_joint_raw_spectrum", "U_S", "J_S_raw"),
        ("sequential_to_joint_image", "S_I", "J_I"),
        ("sequential_to_joint_spectrum", "S_S", "J_S"),
    )
    rng = np.random.default_rng(42)
    a, b = rng.integers(0, len(test), (2, 300_000))
    b[a == b] = (b[a == b] + 1) % len(test)
    cache = {name: values(name, test) for name in STATES}
    near = {name: np.load(A / f"neighbors_{name}_test.npy") for name in STATES}
    output = {"comparisons": {}}
    for label, left, right in pairs:
        x, y = cache[left], cache[right]
        xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        yn = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
        sx = np.einsum("ij,ij->i", xn[a], xn[b])
        sy = np.einsum("ij,ij->i", yn[a], yn[b])
        output["comparisons"][label] = {
            "states": [left, right], "linear_cka_test": cka(x, y),
            "pairwise_cosine_spearman": float(spearmanr(sx, sy).statistic),
            "neighbor_overlap": {str(k): study.row_overlap(near[left], near[right], k)
                                 for k in (10, 50, 100)}
        }
    study.atomic_json(R / "retention.json", output)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pca"); p.add_argument("--state", choices=("U_I", "U_S")); p.add_argument("--device")
    n = sub.add_parser("neighbors"); n.add_argument("--state", choices=STATES); n.add_argument("--device")
    d = sub.add_parser("decompose"); d.add_argument("--regime", choices=REGIMES); d.add_argument("--device")
    sub.add_parser("retention")
    args = parser.parse_args()
    if args.cmd == "pca": pca(args.state, torch.device(args.device))
    elif args.cmd == "neighbors": neighbors(args.state, torch.device(args.device))
    elif args.cmd == "decompose": decompose(args.regime, torch.device(args.device))
    else: retention()


if __name__ == "__main__":
    main()

