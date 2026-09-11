#!/usr/bin/env python3
"""Reproducible sequential-vs-joint cross-modal JEPA representation study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.linalg import eigh
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
DATA_DIR = Path("/mnt/datasets/pranav/astroclip")
PROVABGS = Path(
    "/mnt/datasets/utbd_pranav/catalogs/desi_provabgs/"
    "mmu_desi_provabgs/dataset"
)
CACHE = Path("/mnt/datasets/utbd_pranav/astrojepa_eval_cache")

SPACES = {
    "S_I": CACHE / "crossmodal_posttrained_307k_last_image",
    "S_S": CACHE / "crossmodal_posttrained_307k_last_spectrum",
    "J_I": CACHE / "crossmodal_scratch_307k_last_image",
    "J_S": CACHE / "crossmodal_scratch_307k_last_spectrum",
}
MODELS = {"sequential": ("S_I", "S_S"), "joint": ("J_I", "J_S")}
PROPERTY_NAMES = ("stellar_mass", "metallicity", "age", "ssfr")
L2_GRID = (1e-4, 1e-2, 1.0, 1e2, 1e4)


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedding(name: str, kind: str = "projected") -> np.ndarray:
    return np.load(SPACES[name] / f"{kind}.npy", mmap_mode="r")


def split_files() -> tuple[list[Path], list[Path]]:
    train = sorted((DATA_DIR / "data").glob("train-*.parquet"))
    test = sorted((DATA_DIR / "data").glob("test-*.parquet"))
    if len(train) != 120 or len(test) != 26:
        raise RuntimeError(f"Expected 120/26 shards, found {len(train)}/{len(test)}")
    return train, test


def prepare() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    manifest = {"created_unix": time.time(), "spaces": {}}
    expected_shape = None
    for name, root in SPACES.items():
        meta_path = root / "metadata.json"
        metadata = json.loads(meta_path.read_text())
        shapes = {}
        for kind in ("raw", "projected"):
            array = embedding(name, kind)
            shapes[kind] = list(array.shape)
            if expected_shape is None:
                expected_shape = array.shape[0]
            if array.shape[0] != expected_shape:
                raise RuntimeError(f"Row mismatch for {name}/{kind}: {array.shape}")
        manifest["spaces"][name] = {
            "root": str(root),
            "metadata_sha256": sha256(meta_path),
            "checkpoint": metadata["checkpoint"],
            "global_step": metadata["global_step"],
            "saved_epoch": metadata["saved_epoch"],
            "shapes": shapes,
        }

    train_files, test_files = split_files()
    redshifts, targetids, is_test = [], [], []
    for test_flag, files in ((False, train_files), (True, test_files)):
        for path in files:
            table = pq.read_table(path, columns=["redshift", "targetid"])
            n = len(table)
            redshifts.append(table["redshift"].to_numpy().astype(np.float64))
            targetids.append(table["targetid"].to_numpy().astype(np.int64))
            is_test.append(np.full(n, test_flag, dtype=bool))
    redshifts = np.concatenate(redshifts)
    targetids = np.concatenate(targetids)
    is_test = np.concatenate(is_test)
    if len(redshifts) != expected_shape:
        raise RuntimeError(f"Label rows {len(redshifts)} != embedding rows {expected_shape}")

    catalog = {}
    columns = ["object_id", "LOG_MSTAR", "Z_MW", "TAGE_MW", "AVG_SFR"]
    shards = sorted(PROVABGS.rglob("Npix=*.parquet"))
    for index, path in enumerate(shards, 1):
        table = pq.read_table(path, columns=columns)
        ids = table["object_id"].to_numpy().astype(np.int64)
        values = np.column_stack(
            [table[name].to_numpy().astype(np.float64) for name in columns[1:]]
        )
        for object_id, value in zip(ids, values):
            catalog.setdefault(int(object_id), value)
        if index % 10 == 0:
            print(f"PROVABGS {index}/{len(shards)}", flush=True)
    raw = np.full((len(targetids), 4), np.nan)
    matched = np.zeros(len(targetids), dtype=bool)
    for index, targetid in enumerate(targetids):
        value = catalog.get(int(targetid))
        if value is not None:
            raw[index] = value
            matched[index] = True
    log_mass, metallicity, age, sfr = raw.T
    valid = (
        matched
        & np.isfinite(raw).all(axis=1)
        & (log_mass > 0)
        & (metallicity > 0)
        & (age > 0)
        & (sfr > 0)
    )
    properties = np.full_like(raw, np.nan)
    properties[valid, 0] = log_mass[valid]
    properties[valid, 1] = np.log10(metallicity[valid])
    properties[valid, 2] = age[valid]
    properties[valid, 3] = np.log10(sfr[valid]) - log_mass[valid]
    np.savez_compressed(
        ARTIFACTS / "labels.npz",
        redshift=redshifts,
        targetid=targetids,
        is_test=is_test,
        properties=properties,
        property_valid=valid,
    )
    manifest["dataset"] = {
        "data_dir": str(DATA_DIR),
        "provabgs": str(PROVABGS),
        "rows": int(len(redshifts)),
        "train_rows": int((~is_test).sum()),
        "test_rows": int(is_test.sum()),
        "property_rows": int(valid.sum()),
        "property_train_rows": int((valid & ~is_test).sum()),
        "property_test_rows": int((valid & is_test).sum()),
        "row_order": "sorted train shards followed by sorted test shards",
    }
    atomic_json(ARTIFACTS / "manifest.json", manifest)
    print(json.dumps(manifest["dataset"], indent=2), flush=True)


def device_from_arg(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda:0")
    return device


def normalized_tensor(array: np.ndarray, indices: np.ndarray, device) -> torch.Tensor:
    values = np.asarray(array[indices], dtype=np.float32)
    tensor = torch.from_numpy(values).to(device)
    return torch.nn.functional.normalize(tensor, dim=1)


@torch.inference_mode()
def retrieval_direction(query, keys, chunk: int = 1024) -> dict:
    ranks = np.empty(len(query), dtype=np.int32)
    matched = np.empty(len(query), dtype=np.float32)
    for start in range(0, len(query), chunk):
        stop = min(len(query), start + chunk)
        scores = query[start:stop] @ keys.T
        local = torch.arange(stop - start, device=query.device)
        truth = torch.arange(start, stop, device=query.device)
        true_scores = scores[local, truth]
        ranks[start:stop] = (
            1 + (scores > true_scores[:, None]).sum(dim=1)
        ).cpu().numpy()
        matched[start:stop] = true_scores.cpu().numpy()
        if start % (chunk * 10) == 0:
            print(f"retrieval {stop}/{len(query)}", flush=True)
    rng = np.random.default_rng(42)
    permutation = rng.permutation(len(query))
    fixed = permutation == np.arange(len(query))
    permutation[fixed] = np.roll(permutation[fixed], 1)
    random_cos = (query * keys[torch.from_numpy(permutation).to(query.device)]).sum(1)
    return {
        "recall_at_1": float((ranks <= 1).mean()),
        "recall_at_5": float((ranks <= 5).mean()),
        "recall_at_10": float((ranks <= 10).mean()),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(ranks.mean()),
        "mrr": float((1.0 / ranks).mean()),
        "matched_cosine_mean": float(matched.mean()),
        "matched_cosine_std": float(matched.std()),
        "random_cosine_mean": float(random_cos.mean().cpu()),
        "matched_minus_random": float(matched.mean() - random_cos.mean().cpu()),
        "num_queries": int(len(query)),
    }


def retrieval(model: str, device: torch.device) -> None:
    image_name, spectrum_name = MODELS[model]
    labels = np.load(ARTIFACTS / "labels.npz")
    all_index = np.arange(len(labels["is_test"]))
    test_index = np.flatnonzero(labels["is_test"])
    output = {"model": model, "representation": "projected", "scopes": {}}
    for scope, index in (("all_168280", all_index), ("test_29697", test_index)):
        print(f"{model} {scope}: loading", flush=True)
        image = normalized_tensor(embedding(image_name), index, device)
        spectrum = normalized_tensor(embedding(spectrum_name), index, device)
        output["scopes"][scope] = {
            "image_to_spectrum": retrieval_direction(image, spectrum),
            "spectrum_to_image": retrieval_direction(spectrum, image),
        }
        del image, spectrum
        torch.cuda.empty_cache()
    atomic_json(RESULTS / f"retrieval_{model}.json", output)


@torch.inference_mode()
def neighbors(name: str, device: torch.device, k: int = 100) -> None:
    labels = np.load(ARTIFACTS / "labels.npz")
    index = np.flatnonzero(labels["is_test"])
    values = normalized_tensor(embedding(name), index, device)
    result = np.empty((len(values), k), dtype=np.int32)
    chunk = 1024
    for start in range(0, len(values), chunk):
        stop = min(len(values), start + chunk)
        scores = values[start:stop] @ values.T
        local = torch.arange(stop - start, device=device)
        scores[local, torch.arange(start, stop, device=device)] = -float("inf")
        result[start:stop] = torch.topk(scores, k=k, dim=1).indices.cpu().numpy()
        if start % (chunk * 5) == 0:
            print(f"{name} neighbors {stop}/{len(values)}", flush=True)
    np.save(ARTIFACTS / f"neighbors_{name}_test.npy", result)


@torch.inference_mode()
def linear_cka(x_array, y_array, index, device, chunk=8192) -> float:
    x_mean = torch.from_numpy(np.asarray(x_array[index]).mean(0).astype(np.float32)).to(device)
    y_mean = torch.from_numpy(np.asarray(y_array[index]).mean(0).astype(np.float32)).to(device)
    cross = torch.zeros((x_array.shape[1], y_array.shape[1]), device=device)
    xx = torch.zeros((x_array.shape[1], x_array.shape[1]), device=device)
    yy = torch.zeros((y_array.shape[1], y_array.shape[1]), device=device)
    for start in range(0, len(index), chunk):
        selected = index[start : start + chunk]
        x = torch.from_numpy(np.asarray(x_array[selected], dtype=np.float32)).to(device) - x_mean
        y = torch.from_numpy(np.asarray(y_array[selected], dtype=np.float32)).to(device) - y_mean
        cross += x.T @ y
        xx += x.T @ x
        yy += y.T @ y
    numerator = cross.square().sum()
    denominator = xx.square().sum().sqrt() * yy.square().sum().sqrt()
    return float((numerator / denominator.clamp_min(1e-20)).cpu())


def row_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    total = 0
    chunk = 512
    for start in range(0, len(a), chunk):
        aa = a[start : start + chunk, :k]
        bb = b[start : start + chunk, :k]
        total += int((aa[:, :, None] == bb[:, None, :]).any(axis=2).sum())
    return total / (len(a) * k)


def geometry(device: torch.device, pair_samples: int = 500_000) -> None:
    labels = np.load(ARTIFACTS / "labels.npz")
    all_index = np.arange(len(labels["is_test"]))
    test_index = np.flatnonzero(labels["is_test"])
    names = list(SPACES)
    cka = {}
    for kind, index in (("projected_full", all_index), ("raw_test", test_index)):
        matrix = np.eye(len(names), dtype=float)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                print(f"CKA {kind} {names[i]} {names[j]}", flush=True)
                value = linear_cka(
                    embedding(names[i], kind.split("_")[0]),
                    embedding(names[j], kind.split("_")[0]),
                    index,
                    device,
                )
                matrix[i, j] = matrix[j, i] = value
        cka[kind] = matrix.tolist()

    rng = np.random.default_rng(42)
    first = rng.integers(0, len(all_index), size=pair_samples)
    second = rng.integers(0, len(all_index), size=pair_samples)
    same = first == second
    second[same] = (second[same] + 1) % len(all_index)
    similarities = []
    for name in names:
        array = embedding(name)
        a = np.asarray(array[first], dtype=np.float32)
        b = np.asarray(array[second], dtype=np.float32)
        a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
        similarities.append(np.einsum("ij,ij->i", a, b))
    distance_spearman = np.eye(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            value = spearmanr(similarities[i], similarities[j]).statistic
            distance_spearman[i, j] = distance_spearman[j, i] = value

    neighborhood = {}
    arrays = {name: np.load(ARTIFACTS / f"neighbors_{name}_test.npy") for name in names}
    for k in (10, 50, 100):
        matrix = np.eye(len(names))
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                matrix[i, j] = matrix[j, i] = row_overlap(arrays[names[i]], arrays[names[j]], k)
        neighborhood[str(k)] = matrix.tolist()
    atomic_json(
        RESULTS / "geometry.json",
        {
            "space_order": names,
            "cka": cka,
            "distance_spearman_projected": distance_spearman.tolist(),
            "distance_pair_samples": pair_samples,
            "neighborhood_overlap_test": neighborhood,
            "neighborhood_candidate_rows": int(len(test_index)),
        },
    )


@torch.inference_mode()
def rank_geometry(name: str, device: torch.device) -> None:
    labels = np.load(ARTIFACTS / "labels.npz")
    index = np.flatnonzero(~labels["is_test"])
    output = {"space": name, "representations": {}}
    for kind in ("raw", "projected"):
        array = embedding(name, kind)
        values = np.asarray(array[index], dtype=np.float32)
        mean = values.mean(0)
        std = np.maximum(values.std(0), 1e-6)
        covariance = torch.zeros((values.shape[1], values.shape[1]), device=device)
        for start in range(0, len(values), 4096):
            x = torch.from_numpy((values[start : start + 4096] - mean) / std).to(device)
            covariance += x.T @ x
        eigenvalues = torch.linalg.eigvalsh(covariance / (len(values) - 1)).clamp_min(0).cpu().numpy()
        total = eigenvalues.sum()
        probabilities = eigenvalues / max(total, 1e-20)
        nonzero = probabilities > 0
        output["representations"][kind] = {
            "feature_dim": int(values.shape[1]),
            "effective_rank": float(np.exp(-(probabilities[nonzero] * np.log(probabilities[nonzero])).sum())),
            "participation_ratio": float(total**2 / np.square(eigenvalues).sum()),
            "largest_eigenvalue_share": float(eigenvalues[-1] / total),
            "eigenvalues_ascending": eigenvalues.tolist(),
            "rows": int(len(values)),
            "definition": "eigenvalues of train-standardized feature correlation matrix",
        }
        print(f"rank geometry {name} {kind}", flush=True)
    atomic_json(RESULTS / f"rank_geometry_{name}.json", output)


def standardize(train: np.ndarray, test: np.ndarray):
    mean = train.mean(0, dtype=np.float64).astype(np.float32)
    std = train.std(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return (train - mean) / std, (test - mean) / std, mean, std


def global_r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    residual = (pred - target).square().sum()
    denominator = (target - target.mean(0)).square().sum()
    return float((1 - residual / denominator.clamp_min(1e-20)).cpu())


@torch.inference_mode()
def tensor_retrieval(query, keys) -> dict:
    query = torch.nn.functional.normalize(query.float(), dim=1)
    keys = torch.nn.functional.normalize(keys.float(), dim=1)
    return retrieval_direction(query, keys, chunk=1024)


def fit_ridge_map(train_x, train_y, test_x, test_y, device):
    xtr = torch.from_numpy(train_x).float().to(device)
    ytr = torch.from_numpy(train_y).float().to(device)
    xte = torch.from_numpy(test_x).float().to(device)
    yte = torch.from_numpy(test_y).float().to(device)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(train_x))
    nval = int(0.1 * len(order))
    val = torch.from_numpy(order[:nval]).to(device)
    fit = torch.from_numpy(order[nval:]).to(device)
    candidates = []
    best = None
    for l2 in L2_GRID:
        gram = xtr[fit].T @ xtr[fit]
        gram.diagonal().add_(l2)
        weights = torch.linalg.solve(gram, xtr[fit].T @ ytr[fit])
        score = global_r2(xtr[val] @ weights, ytr[val])
        candidates.append({"l2": l2, "validation_global_r2": score})
        if best is None or score > best[0]:
            best = (score, l2)
    gram = xtr.T @ xtr
    gram.diagonal().add_(best[1])
    weights = torch.linalg.solve(gram, xtr.T @ ytr)
    prediction = xte @ weights
    residual = prediction - yte
    variance = (yte - yte.mean(0)).square().sum(0).clamp_min(1e-12)
    per_dim_r2 = 1 - residual.square().sum(0) / variance
    cosine = torch.nn.functional.cosine_similarity(prediction, yte).mean()
    return {
        "selected_l2": best[1],
        "candidates": candidates,
        "global_r2": global_r2(prediction, yte),
        "uniform_mean_dimension_r2": float(per_dim_r2.mean().cpu()),
        "explained_variance_weighted": float(
            (1 - residual.var(0, unbiased=False).sum() / yte.var(0, unbiased=False).sum()).cpu()
        ),
        "mean_cosine": float(cosine.cpu()),
        "predicted_retrieval": tensor_retrieval(prediction, yte),
    }


def cca_summary(train_x, train_y, test_x, test_y, regularization=1e-3):
    n = len(train_x)
    cxx = train_x.T @ train_x / (n - 1)
    cyy = train_y.T @ train_y / (n - 1)
    cxy = train_x.T @ train_y / (n - 1)
    cxx += np.eye(cxx.shape[0]) * regularization
    cyy += np.eye(cyy.shape[0]) * regularization
    ex, ux = eigh(cxx)
    ey, uy = eigh(cyy)
    wx = (ux * np.maximum(ex, 1e-10) ** -0.5) @ ux.T
    wy = (uy * np.maximum(ey, 1e-10) ** -0.5) @ uy.T
    u, train_corr, vt = np.linalg.svd(wx @ cxy @ wy, full_matrices=False)
    ax, ay = wx @ u, wy @ vt.T
    tx, ty = test_x @ ax, test_y @ ay
    test_corr = np.array(
        [np.corrcoef(tx[:, i], ty[:, i])[0, 1] for i in range(tx.shape[1])]
    )
    return {
        "regularization": regularization,
        "train_top_canonical_correlation": float(train_corr[0]),
        "test_top1": float(test_corr[0]),
        "test_top10_mean": float(test_corr[:10].mean()),
        "test_top50_mean": float(test_corr[:50].mean()),
        "test_all_mean": float(test_corr.mean()),
    }


def mappings(model: str, device: torch.device) -> None:
    image_name, spectrum_name = MODELS[model]
    labels = np.load(ARTIFACTS / "labels.npz")
    train_index = np.flatnonzero(~labels["is_test"])
    test_index = np.flatnonzero(labels["is_test"])
    ix = np.asarray(embedding(image_name)[train_index], dtype=np.float32)
    iy = np.asarray(embedding(image_name)[test_index], dtype=np.float32)
    sx = np.asarray(embedding(spectrum_name)[train_index], dtype=np.float32)
    sy = np.asarray(embedding(spectrum_name)[test_index], dtype=np.float32)
    ix, iy, image_mean, image_std = standardize(ix, iy)
    sx, sy, spectrum_mean, spectrum_std = standardize(sx, sy)

    cross = ix.T @ sx
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    rotation = u @ vt
    np.save(ARTIFACTS / f"procrustes_{model}_image_to_spectrum.npy", rotation)
    i_test = torch.from_numpy(iy).float().to(device)
    s_test = torch.from_numpy(sy).float().to(device)
    rot = torch.from_numpy(rotation).float().to(device)
    procrustes = {
        "image_to_spectrum": {
            "before": tensor_retrieval(i_test, s_test),
            "after": tensor_retrieval(i_test @ rot, s_test),
        },
        "spectrum_to_image": {
            "before": tensor_retrieval(s_test, i_test),
            "after": tensor_retrieval(s_test @ rot.T, i_test),
        },
        "heldout_rows": int(len(test_index)),
        "fit_rows": int(len(train_index)),
        "preprocessing": "per-modality train mean/std before orthogonal fit",
    }
    ridge = {
        "image_to_spectrum": fit_ridge_map(ix, sx, iy, sy, device),
        "spectrum_to_image": fit_ridge_map(sx, ix, sy, iy, device),
    }
    cca = cca_summary(ix.astype(np.float64), sx.astype(np.float64), iy, sy)
    atomic_json(
        RESULTS / f"mappings_{model}.json",
        {"model": model, "procrustes": procrustes, "ridge": ridge, "cca": cca},
    )


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    residual = prediction - target
    denominator = np.square(target - target.mean()).sum()
    return {
        "r2": float(1 - np.square(residual).sum() / max(denominator, 1e-12)),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
    }


def ridge_probe(train_x, train_y, test_x, test_y, device, seeds=(42, 43, 44)):
    xtr = torch.from_numpy(train_x).float().to(device)
    xte = torch.from_numpy(test_x).float().to(device)
    repeats = []
    for seed in seeds:
        order = np.random.default_rng(seed).permutation(len(train_y))
        nval = max(1, int(0.1 * len(order)))
        validation, fit = order[:nval], order[nval:]
        fit_index = torch.from_numpy(fit).to(device)
        val_index = torch.from_numpy(validation).to(device)
        yfit = torch.from_numpy(train_y[fit]).double().to(device)
        ymean = yfit.mean()
        best = None
        for l2 in L2_GRID:
            x = xtr[fit_index].double()
            gram = x.T @ x
            gram.diagonal().add_(l2)
            weights = torch.linalg.solve(gram, x.T @ (yfit - ymean))
            pred = (xtr[val_index].double() @ weights + ymean).cpu().numpy()
            score = regression_metrics(pred, train_y[validation])["r2"]
            if best is None or score > best[0]:
                best = (score, l2, weights, ymean)
        pred = (xte.double() @ best[2] + best[3]).cpu().numpy()
        repeats.append({"seed": seed, "selected_l2": best[1], **regression_metrics(pred, test_y)})
    summary = {}
    for metric in ("r2", "mae", "rmse"):
        values = [repeat[metric] for repeat in repeats]
        summary[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    return {"repeats": repeats, "summary": summary}


def probes(name: str, device: torch.device) -> None:
    labels = np.load(ARTIFACTS / "labels.npz")
    train_mask = ~labels["is_test"]
    test_mask = labels["is_test"]
    property_mask = labels["property_valid"]
    output = {"space": name, "representations": {}}
    for kind in ("raw", "projected"):
        array = embedding(name, kind)
        train = np.asarray(array[train_mask], dtype=np.float32)
        test = np.asarray(array[test_mask], dtype=np.float32)
        train, test, _, _ = standardize(train, test)
        result = {
            "redshift": ridge_probe(
                train,
                labels["redshift"][train_mask],
                test,
                labels["redshift"][test_mask],
                device,
            ),
            "properties": {},
        }
        train_global = np.flatnonzero(train_mask)
        test_global = np.flatnonzero(test_mask)
        train_lookup = np.full(len(train_mask), -1, dtype=np.int64)
        test_lookup = np.full(len(test_mask), -1, dtype=np.int64)
        train_lookup[train_global] = np.arange(len(train_global))
        test_lookup[test_global] = np.arange(len(test_global))
        property_train_global = np.flatnonzero(property_mask & train_mask)
        property_test_global = np.flatnonzero(property_mask & test_mask)
        property_train = train[train_lookup[property_train_global]]
        property_test = test[test_lookup[property_test_global]]
        for column, target in enumerate(PROPERTY_NAMES):
            print(f"{name} {kind} {target}", flush=True)
            result["properties"][target] = ridge_probe(
                property_train,
                labels["properties"][property_train_global, column],
                property_test,
                labels["properties"][property_test_global, column],
                device,
            )
        output["representations"][kind] = result
        del train, test
        torch.cuda.empty_cache()
    atomic_json(RESULTS / f"probes_{name}.json", output)


def fit_scalar_decoder(train_x, train_y, device):
    xtr = torch.from_numpy(train_x).float().to(device)
    ytr = torch.from_numpy(train_y).double().to(device)
    order = np.random.default_rng(42).permutation(len(train_y))
    nval = max(1, int(0.1 * len(order)))
    validation = torch.from_numpy(order[:nval]).to(device)
    fit = torch.from_numpy(order[nval:]).to(device)
    ymean = ytr[fit].mean()
    best = None
    for l2 in L2_GRID:
        x = xtr[fit].double()
        gram = x.T @ x
        gram.diagonal().add_(l2)
        weights = torch.linalg.solve(gram, x.T @ (ytr[fit] - ymean))
        prediction = (xtr[validation].double() @ weights + ymean).cpu().numpy()
        score = regression_metrics(prediction, train_y[order[:nval]])["r2"]
        if best is None or score > best[0]:
            best = (score, l2, weights, ymean)
    return best[2], float(best[3]), best[1]


def apply_scalar_decoder(features, weights, ymean, target, device):
    values = torch.from_numpy(features).double().to(device)
    prediction = (values @ weights + ymean).cpu().numpy()
    return regression_metrics(prediction, target)


def decoder_transfer(model: str, device: torch.device) -> None:
    image_name, spectrum_name = MODELS[model]
    labels = np.load(ARTIFACTS / "labels.npz")
    train_mask = ~labels["is_test"]
    test_mask = labels["is_test"]
    train_index = np.flatnonzero(train_mask)
    test_index = np.flatnonzero(test_mask)
    image_train = np.asarray(embedding(image_name)[train_index], dtype=np.float32)
    image_test = np.asarray(embedding(image_name)[test_index], dtype=np.float32)
    spectrum_train = np.asarray(embedding(spectrum_name)[train_index], dtype=np.float32)
    spectrum_test = np.asarray(embedding(spectrum_name)[test_index], dtype=np.float32)
    image_train, image_test, _, _ = standardize(image_train, image_test)
    spectrum_train, spectrum_test, _, _ = standardize(spectrum_train, spectrum_test)
    rotation = np.load(ARTIFACTS / f"procrustes_{model}_image_to_spectrum.npy")

    targets = {"redshift": labels["redshift"]}
    for column, name in enumerate(PROPERTY_NAMES):
        targets[name] = labels["properties"][:, column]
    output = {
        "model": model,
        "representation": "projected",
        "coordinate_preprocessing": "per-modality train mean/std",
        "targets": {},
    }
    for name, values in targets.items():
        valid = np.isfinite(values)
        train_global = np.flatnonzero(valid & train_mask)
        test_global = np.flatnonzero(valid & test_mask)
        train_local = np.searchsorted(train_index, train_global)
        test_local = np.searchsorted(test_index, test_global)
        ytrain, ytest = values[train_global], values[test_global]
        image_weights, image_mean, image_l2 = fit_scalar_decoder(
            image_train[train_local], ytrain, device
        )
        spectrum_weights, spectrum_mean, spectrum_l2 = fit_scalar_decoder(
            spectrum_train[train_local], ytrain, device
        )
        output["targets"][name] = {
            "image_decoder": {
                "selected_l2": image_l2,
                "own_image": apply_scalar_decoder(
                    image_test[test_local], image_weights, image_mean, ytest, device
                ),
                "direct_spectrum": apply_scalar_decoder(
                    spectrum_test[test_local], image_weights, image_mean, ytest, device
                ),
                "procrustes_spectrum_to_image": apply_scalar_decoder(
                    spectrum_test[test_local] @ rotation.T,
                    image_weights,
                    image_mean,
                    ytest,
                    device,
                ),
            },
            "spectrum_decoder": {
                "selected_l2": spectrum_l2,
                "own_spectrum": apply_scalar_decoder(
                    spectrum_test[test_local], spectrum_weights, spectrum_mean, ytest, device
                ),
                "direct_image": apply_scalar_decoder(
                    image_test[test_local], spectrum_weights, spectrum_mean, ytest, device
                ),
                "procrustes_image_to_spectrum": apply_scalar_decoder(
                    image_test[test_local] @ rotation,
                    spectrum_weights,
                    spectrum_mean,
                    ytest,
                    device,
                ),
            },
        }
        print(f"{model} decoder transfer {name}", flush=True)
    atomic_json(RESULTS / f"decoder_transfer_{model}.json", output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    retrieve = sub.add_parser("retrieval")
    retrieve.add_argument("--model", choices=MODELS, required=True)
    retrieve.add_argument("--device", default="cuda:0")
    near = sub.add_parser("neighbors")
    near.add_argument("--space", choices=SPACES, required=True)
    near.add_argument("--device", default="cuda:0")
    geom = sub.add_parser("geometry")
    geom.add_argument("--device", default="cuda:0")
    mapping = sub.add_parser("mappings")
    mapping.add_argument("--model", choices=MODELS, required=True)
    mapping.add_argument("--device", default="cuda:0")
    probe = sub.add_parser("probes")
    probe.add_argument("--space", choices=SPACES, required=True)
    probe.add_argument("--device", default="cuda:0")
    transfer = sub.add_parser("decoder-transfer")
    transfer.add_argument("--model", choices=MODELS, required=True)
    transfer.add_argument("--device", default="cuda:0")
    rank = sub.add_parser("rank-geometry")
    rank.add_argument("--space", choices=SPACES, required=True)
    rank.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "retrieval":
        retrieval(args.model, device_from_arg(args.device))
    elif args.command == "neighbors":
        neighbors(args.space, device_from_arg(args.device))
    elif args.command == "geometry":
        geometry(device_from_arg(args.device))
    elif args.command == "mappings":
        mappings(args.model, device_from_arg(args.device))
    elif args.command == "probes":
        probes(args.space, device_from_arg(args.device))
    elif args.command == "decoder-transfer":
        decoder_transfer(args.model, device_from_arg(args.device))
    elif args.command == "rank-geometry":
        rank_geometry(args.space, device_from_arg(args.device))


if __name__ == "__main__":
    main()
