"""
Distance-transform peak counting: decide how many roughly-circular "lobes"
a segmented blob actually contains, so we can tell a single cell from a
budding mother-daughter pair from a genuine multi-cell clump -- without
cutting the blob apart (which would destroy the neck shape both
classifiers rely on).

This is the marker-finding half of watershed segmentation, used here purely
for counting/classification, not for splitting.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

# Minimum distance (px) between two peaks for them to count as separate
# lobes, rather than the same lobe's local max spilling over. Should be
# smaller than a typical bud's radius.
MIN_PEAK_DISTANCE = 6
# A candidate peak must reach at least this fraction of the tallest peak's
# height to count as a real lobe rather than a rounding bump.
MIN_RELATIVE_HEIGHT = 0.35


@dataclass
class LobeInfo:
    n_lobes: int
    peak_heights: list[float]
    label: str  # 'single' | 'budding_pair' | 'multicell' | 'ambiguous'


def analyze_lobes(region_mask: np.ndarray) -> LobeInfo:
    dist = ndi.distance_transform_edt(region_mask)
    coords = peak_local_max(dist, min_distance=MIN_PEAK_DISTANCE, labels=region_mask)
    if len(coords) == 0:
        return LobeInfo(0, [], "ambiguous")

    heights = sorted((dist[r, c] for r, c in coords), reverse=True)
    tallest = heights[0]
    real_peaks = [h for h in heights if h >= tallest * MIN_RELATIVE_HEIGHT]
    n = len(real_peaks)

    if n <= 1:
        label = "single"
    elif n == 2:
        # budding pairs are asymmetric (mother notably bigger than bud);
        # two peaks of near-equal height more likely means two separate
        # same-size cells just touching, not a real mother-daughter pair.
        ratio = real_peaks[1] / real_peaks[0]
        label = "budding_pair" if ratio < 0.85 else "ambiguous"
    else:
        label = "multicell"

    return LobeInfo(n_lobes=n, peak_heights=real_peaks, label=label)
