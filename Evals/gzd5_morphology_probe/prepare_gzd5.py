#!/usr/bin/env python3
"""Prepare public GZD-5 catalogs, debiased labels, and images."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
GZD5_DIR = DATA_DIR / "gz_decals_5"
TRASH_DIR = SCRIPT_DIR / "trash"

TRAIN_CATALOG_URL = "https://dl.dropboxusercontent.com/s/1tuehajonhgv8a2/decals_dr5_ortho_train_catalog.parquet"
TEST_CATALOG_URL = "https://dl.dropboxusercontent.com/s/3vo6hjlnbqgzuxz/decals_dr5_ortho_test_catalog.parquet"
IMAGE_TARBALL_URL = "https://dl.dropboxusercontent.com/s/bs6jp0mkgiekhww/decals_dr5_images.tar.gz"
VOLUNTEER_URL = "https://zenodo.org/records/4573248/files/gz_decals_volunteers_5.csv?download=1"

IMAGE_TARBALL_MD5 = "1347de4c8df4ec579d5a58241c1f280b"
SPOTCHECK_IMAGE = GZD5_DIR / "images/J073/J073013.60+242930.0.jpg"

QUESTIONS = {
    "smooth": {
        "prefix": "smooth-or-featured",
        "answers": ["smooth", "featured-or-disk", "artifact"],
    },
    "disk-edge-on": {"prefix": "disk-edge-on", "answers": ["yes", "no"]},
    "spiral-arms": {"prefix": "has-spiral-arms", "answers": ["yes", "no"]},
    "bar": {"prefix": "bar", "answers": ["strong", "weak", "no"]},
    "bulge-size": {
        "prefix": "bulge-size",
        "answers": ["dominant", "large", "moderate", "small", "none"],
    },
    "how-rounded": {
        "prefix": "how-rounded",
        "answers": ["round", "in-between", "cigar-shaped"],
    },
    "edge-on-bulge": {
        "prefix": "edge-on-bulge",
        "answers": ["boxy", "none", "rounded"],
    },
    "spiral-winding": {
        "prefix": "spiral-winding",
        "answers": ["tight", "medium", "loose"],
    },
    "spiral-arm-count": {
        "prefix": "spiral-arm-count",
        "answers": ["1", "2", "3", "4", "more-than-4", "cant-tell"],
    },
    "merging": {
        "prefix": "merging",
        "answers": ["none", "minor-disturbance", "major-disturbance", "merger"],
    },
}


def download_file(url: str, path: Path, timeout: int = 180) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        print(f"exists: {path} ({path.stat().st_size} bytes)")
        return
    print(f"downloading: {url}")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
        tmp.replace(path)
    print(f"saved: {path} ({path.stat().st_size} bytes)")


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def move_to_trash(path: Path, reason: str) -> Path:
    TRASH_DIR.mkdir(parents=True, exist_ok=True)
    target = TRASH_DIR / f"{path.name}.{reason}"
    counter = 1
    while target.exists():
        target = TRASH_DIR / f"{path.name}.{reason}.{counter}"
        counter += 1
    shutil.move(str(path), str(target))
    return target


def ensure_images(download_images: bool) -> None:
    tarball = GZD5_DIR / "decals_dr5_images.tar.gz"
    if SPOTCHECK_IMAGE.exists():
        print(f"images already extracted: {SPOTCHECK_IMAGE}")
        return
    if not download_images:
        print("images missing; rerun without --skip-images to download/extract them")
        return

    if tarball.exists():
        actual = md5(tarball)
        if actual != IMAGE_TARBALL_MD5:
            trashed = move_to_trash(tarball, f"bad-md5-{actual}")
            print(f"moved corrupt tarball to {trashed}")
    if not tarball.exists():
        print("downloading image tarball with curl -C - for resume support")
        subprocess.run(
            [
                "curl",
                "-L",
                "-C",
                "-",
                IMAGE_TARBALL_URL,
                "-o",
                str(tarball),
            ],
            check=True,
        )
    actual = md5(tarball)
    if actual != IMAGE_TARBALL_MD5:
        trashed = move_to_trash(tarball, f"bad-md5-{actual}")
        raise RuntimeError(f"Image tarball failed md5 and was moved to {trashed}")

    print(f"extracting {tarball}")
    subprocess.run(["tar", "-xzf", str(tarball), "-C", str(GZD5_DIR)], check=True)
    if not SPOTCHECK_IMAGE.exists():
        raise FileNotFoundError(f"Extraction finished but spotcheck missing: {SPOTCHECK_IMAGE}")


def target_columns() -> list[str]:
    cols = ["iauname", "smooth-or-featured_total-votes"]
    for spec in QUESTIONS.values():
        prefix = spec["prefix"]
        cols.append(f"{prefix}_total-votes")
        cols.extend(f"{prefix}_{answer}_debiased" for answer in spec["answers"])
    return sorted(set(cols))


def merge_labels(split: str, volunteer: pd.DataFrame) -> pd.DataFrame:
    catalog_path = GZD5_DIR / f"decals_dr5_ortho_{split}_catalog.parquet"
    catalog = pd.read_parquet(catalog_path)
    merged = catalog.merge(
        volunteer,
        left_on="id_str",
        right_on="iauname",
        how="left",
        suffixes=("", "_volunteer"),
    )
    missing = int(merged["smooth-or-featured_total-votes"].isna().sum())
    if missing:
        raise RuntimeError(f"{split} merge has {missing} missing volunteer label rows")
    merged["file_loc"] = merged.apply(
        lambda row: str(GZD5_DIR / "images" / row["subfolder"] / row["filename"]),
        axis=1,
    )
    # AstroCLIP removes galaxies with fewer than 3 votes on the root question.
    merged = merged[merged["smooth-or-featured_total-votes"] >= 3].reset_index(drop=True)
    output = DATA_DIR / f"merged_{split}.parquet"
    merged.to_parquet(output, index=False)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-images", action="store_true", help="Only prepare catalogs/labels.")
    args = parser.parse_args()

    GZD5_DIR.mkdir(parents=True, exist_ok=True)
    download_file(TRAIN_CATALOG_URL, GZD5_DIR / "decals_dr5_ortho_train_catalog.parquet")
    download_file(TEST_CATALOG_URL, GZD5_DIR / "decals_dr5_ortho_test_catalog.parquet")
    download_file(VOLUNTEER_URL, DATA_DIR / "gz_decals_volunteers_5.csv")

    volunteer = pd.read_csv(DATA_DIR / "gz_decals_volunteers_5.csv", usecols=target_columns())
    train = merge_labels("train", volunteer)
    test = merge_labels("test", volunteer)
    ensure_images(download_images=not args.skip_images)

    metadata = {
        "train_rows_after_smooth_vote_filter": int(len(train)),
        "test_rows_after_smooth_vote_filter": int(len(test)),
        "total_rows_after_smooth_vote_filter": int(len(train) + len(test)),
        "raw_train_catalog_rows": int(pd.read_parquet(GZD5_DIR / "decals_dr5_ortho_train_catalog.parquet").shape[0]),
        "raw_test_catalog_rows": int(pd.read_parquet(GZD5_DIR / "decals_dr5_ortho_test_catalog.parquet").shape[0]),
        "volunteer_csv_rows": int(len(volunteer)),
        "questions": QUESTIONS,
        "uses_official_debiased_labels": True,
        "astroclip_notebook_filter": "smooth-or-featured_total-votes >= 3; per-question test counts > 34 during MLP eval",
        "image_spotcheck_exists": SPOTCHECK_IMAGE.exists(),
    }
    (DATA_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

