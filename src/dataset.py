"""
Load the generated crop dataset (per data/generated_dataset/splits.csv)
into CNN-ready NumPy arrays.

Crops are 129x129 uint16 (raw camera intensity, not 0-65535 full range --
varies by frame/exposure). Each crop is:
  1. per-crop min-max normalized to [0, 1] (matches the same per-image
     contrast handling segmentation.py uses -- keeps a dim frame and a
     bright frame comparable instead of the model partly learning
     "exposure level" as a proxy feature)
  2. downsampled to INPUT_SIZE x INPUT_SIZE with anti-aliasing. Full 129x129
     would make the from-scratch conv layers slow for no real benefit --
     the neck/bud shape signal survives well below that resolution.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import tifffile
from skimage.transform import resize

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "generated_dataset"
SPLITS_CSV = DATA_DIR / "splits.csv"

INPUT_SIZE = 48  # bumped from 32: small early-stage buds were shrinking to a
                 # handful of pixels and nearly vanishing at 32x32 (see
                 # docs/CNN_DESIGN.md false-negative analysis)


def _load_crop(rel_path: str) -> np.ndarray:
    img = tifffile.imread(DATA_DIR / rel_path).astype(np.float64)
    lo, hi = img.min(), img.max()
    img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    img = resize(img, (INPUT_SIZE, INPUT_SIZE), anti_aliasing=True, preserve_range=True)
    return img.astype(np.float64)


def load_split(name: str):
    """Returns (X, y, meta) for split `name` in {'train','val','test'}.
    X: (N, 1, INPUT_SIZE, INPUT_SIZE) float64 in [0,1]
    y: (N,) float64 in {0., 1.}, label = modern (convexity-defect) verdict
    meta: list of (frame, cell_label) for traceability
    """
    rows = [r for r in csv.DictReader(open(SPLITS_CSV)) if r["split"] == name]
    X = np.zeros((len(rows), 1, INPUT_SIZE, INPUT_SIZE))
    y = np.zeros(len(rows))
    meta = []
    for i, r in enumerate(rows):
        X[i, 0] = _load_crop(r["crop_path"])
        y[i] = 1.0 if r["modern_budding"] == "True" else 0.0
        meta.append((int(r["frame"]), int(r["cell_label"])))
    return X, y, meta


AUX_FEATURE_NAMES = ["area", "eccentricity", "solidity", "extent", "aspect_ratio",
                       "n_defects", "max_defect_depth_ratio"]


def load_split_hybrid(name: str):
    """Like load_split, but also returns the raw (unstandardized) hand-
    engineered shape features from aux_features.csv, joined by
    (frame, cell_label). Standardization must use train-set statistics
    only -- done by the caller (see scripts/train_cnn_hybrid.py) so this
    function stays a pure data-loading step with no train/val/test leakage
    baked in."""
    aux_by_key = {}
    for r in csv.DictReader(open(DATA_DIR / "aux_features.csv")):
        key = (int(r["frame"]), int(r["cell_label"]))
        aux_by_key[key] = np.array([float(r[name]) for name in AUX_FEATURE_NAMES])

    rows = [r for r in csv.DictReader(open(SPLITS_CSV)) if r["split"] == name]
    X = np.zeros((len(rows), 1, INPUT_SIZE, INPUT_SIZE))
    aux = np.zeros((len(rows), len(AUX_FEATURE_NAMES)))
    y = np.zeros(len(rows))
    meta = []
    missing = 0
    for i, r in enumerate(rows):
        key = (int(r["frame"]), int(r["cell_label"]))
        X[i, 0] = _load_crop(r["crop_path"])
        y[i] = 1.0 if r["modern_budding"] == "True" else 0.0
        if key in aux_by_key:
            aux[i] = aux_by_key[key]
        else:
            missing += 1
        meta.append(key)
    if missing:
        print(f"WARNING: {missing}/{len(rows)} rows in split '{name}' had no matching "
              f"aux_features.csv entry (left as zeros) -- check aux_features.csv is up to date")
    return X, aux, y, meta


def augment_batch(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random 90-degree rotation + random horizontal/vertical flip, applied
    per-sample. Yeast cells have no canonical orientation under the
    microscope, so this is free extra training variety, not distortion --
    and it directly targets the false-positive failure mode where the
    model may have been keying on an incidental brightness asymmetry (a
    shadow on one particular side) rather than true cell shape."""
    Xa = X.copy()
    for i in range(len(Xa)):
        k = rng.integers(0, 4)
        img = np.rot90(Xa[i, 0], k)
        if rng.random() < 0.5:
            img = np.fliplr(img)
        if rng.random() < 0.5:
            img = np.flipud(img)
        Xa[i, 0] = img
    return Xa


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        X, y, meta = load_split(split)
        print(f"{split:5s}: X {X.shape} y {y.shape}  budding={y.mean():.1%}  "
              f"pixel range [{X.min():.2f}, {X.max():.2f}]")
