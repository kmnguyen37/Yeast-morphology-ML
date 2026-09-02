"""
Modern classical budding classifier: a concave-neck rule on convexity-defect
depth, instead of the original zero-crossing-of-local-slope heuristic.

A budding cell is two roughly-convex lobes (mother + daughter) joined at a
narrow neck. That neck is exactly what a convexity defect measures: how far
the actual boundary caves in from its convex hull. One clean, interpretable
threshold on defect depth (relative to cell size) replaces boundary
smoothing + a 6-point sliding-window slope fit + an arbitrary zero-crossing
count.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.measure import label, regionprops

from features import CELL_AREA_CUTOFF, CROP_HALF, _crop, clean_mask
from lobe_analysis import analyze_lobes
from shape_features import ShapeFeatures, compute_shape_features

DEFECT_DEPTH_THRESHOLD = 0.15  # fraction of equivalent radius; tuned below


@dataclass
class ModernCellResult:
    label: int
    centroid_xy: tuple[float, float]
    features: ShapeFeatures
    budding: bool
    crop: np.ndarray | None
    blob_type: str = "single"
    included: bool = True


def classify_by_convexity(features: ShapeFeatures, threshold: float = DEFECT_DEPTH_THRESHOLD) -> bool:
    return features.max_defect_depth_ratio >= threshold


def analyze_frame_modern(
    mask: np.ndarray, original_image: np.ndarray, threshold: float = DEFECT_DEPTH_THRESHOLD
) -> list[ModernCellResult]:
    """Same blob-typing/inclusion policy as features.analyze_frame -- see
    that function's docstring. single -> budding=False, no shape-feature
    call needed; budding_pair -> the convexity-defect classifier; multicell
    / ambiguous -> excluded."""
    mask = clean_mask(mask, area_cutoff=CELL_AREA_CUTOFF)
    labeled = label(mask)
    results: list[ModernCellResult] = []

    for region in regionprops(labeled):
        region_mask = labeled == region.label
        lobes = analyze_lobes(region_mask)

        budding, included = False, True
        feats = compute_shape_features(region_mask)  # still computed: useful even for 'single' (e.g. eccentricity)
        if feats is None:
            continue
        if lobes.label == "single":
            budding = False
        elif lobes.label == "budding_pair":
            budding = classify_by_convexity(feats, threshold)
        else:
            included = False

        cy, cx = region.centroid
        crop = _crop(original_image, cx, cy, half=CROP_HALF)

        results.append(
            ModernCellResult(
                label=region.label,
                centroid_xy=(cx, cy),
                features=feats,
                budding=budding,
                crop=crop,
                blob_type=lobes.label,
                included=included,
            )
        )
    return results
