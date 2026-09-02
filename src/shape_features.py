"""
Shape descriptors for a single segmented cell, used by the modern classical
classifier (see modern_classifier.py). Convexity-defect analysis is the
standard CV technique for detecting a concave "neck" between two joined
blobs -- exactly what a budding mother-daughter pair looks like -- and is
far less sensitive to noise/parameter choices than a hand-tuned curvature
heuristic.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.measure import regionprops


@dataclass
class ShapeFeatures:
    area: float
    eccentricity: float
    solidity: float           # area / convex_hull_area -- low = concave shape
    extent: float              # area / bounding_box_area
    aspect_ratio: float        # major_axis / minor_axis
    n_defects: int              # count of "significant" convexity defects
    max_defect_depth_ratio: float  # deepest defect, normalized by sqrt(area/pi) (~cell radius)


def _region_mask_to_contour(region_mask: np.ndarray) -> np.ndarray | None:
    mask_u8 = (region_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def compute_shape_features(region_mask: np.ndarray, depth_significance: float = 0.05) -> ShapeFeatures | None:
    """region_mask: boolean mask containing exactly one connected cell
    (e.g. `labeled == region.label`). depth_significance: minimum defect
    depth, as a fraction of the cell's equivalent radius, to count as a
    "real" concavity rather than boundary-tracing/pixelation noise."""
    contour = _region_mask_to_contour(region_mask)
    if contour is None or len(contour) < 5:
        return None

    props = regionprops(region_mask.astype(np.uint8))
    if not props:
        return None
    p = props[0]

    equiv_radius = np.sqrt(p.area / np.pi)

    hull_indices = cv2.convexHull(contour, returnPoints=False)
    n_defects = 0
    max_depth_ratio = 0.0
    if hull_indices is not None and len(hull_indices) > 3:
        hull_indices = np.sort(hull_indices, axis=0)
        try:
            defects = cv2.convexityDefects(contour, hull_indices)
        except cv2.error:
            defects = None
        if defects is not None:
            for i in range(defects.shape[0]):
                depth = defects[i, 0, 3] / 256.0  # fixed-point depth in pixels
                depth_ratio = depth / equiv_radius if equiv_radius > 0 else 0.0
                if depth_ratio > max_depth_ratio:
                    max_depth_ratio = depth_ratio
                if depth_ratio >= depth_significance:
                    n_defects += 1

    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = p.area / hull_area if hull_area > 0 else 1.0

    minor = p.axis_minor_length if p.axis_minor_length > 0 else 1e-6
    aspect_ratio = p.axis_major_length / minor

    return ShapeFeatures(
        area=p.area,
        eccentricity=p.eccentricity,
        solidity=solidity,
        extent=p.extent,
        aspect_ratio=aspect_ratio,
        n_defects=n_defects,
        max_defect_depth_ratio=max_depth_ratio,
    )
