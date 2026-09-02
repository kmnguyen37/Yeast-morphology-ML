"""
Python port of create_yeast_training_sets.m: per-cell region analysis,
boundary-curvature-based budding classification, and 128x128 crop
extraction.

Faithfully reproduces the original algorithm, INCLUDING one quirk found
while porting it: the original script rotates each cell's boundary to
align with the fitted ellipse's major axis (`xnew`/`ynew`), but that
rotated boundary is never actually used in the classification step below
-- the zero-crossing count runs on the smoothed but *unrotated* boundary
coordinates. We reproduce that behavior here (see `classify_budding`)
rather than silently "fixing" it, so results match the original output;
the discrepancy from the thesis text (which describes rotation-then-plot)
is documented in docs/PLAN.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter
from skimage.measure import find_contours, label, regionprops
from skimage.segmentation import clear_border
from skimage.morphology import remove_small_objects

from ellipse_fit import fit_ellipse
from lobe_analysis import analyze_lobes

# Raised from the original 200: at 200, thin segmentation-seam artifacts
# (e.g. a 203px sliver between two touching cells) were slipping through
# and getting misread as real cells -- see WEEK1_FINDINGS.md / the frame-14
# "cell 11" case. 350 sits just below the smallest genuine single cell
# observed in the sample (389px) while excluding the artifact cluster.
CELL_AREA_CUTOFF = 350
CROP_HALF = 64          # xBit/2, yBit/2 in the original -> 129x129 crop
RUN_P = 6               # sliding-window size for the local-slope fit
SAVGOL_WINDOW = 27
SAVGOL_POLYORDER = 2
ZERO_CROSSING_THRESHOLD = 4


@dataclass
class CellResult:
    label: int
    centroid_xy: tuple[float, float]
    area: float
    major_axis_length: float
    minor_axis_length: float
    eccentricity: float
    zero_crossings: int
    budding: bool
    crop: np.ndarray | None
    blob_type: str = "single"   # 'single' | 'budding_pair' | 'multicell' | 'ambiguous'
    included: bool = True        # False for multicell/ambiguous -- excluded from training/eval,
                                  # same as the original data's separate `multicell` folder


def clean_mask(mask: np.ndarray, area_cutoff: int = CELL_AREA_CUTOFF) -> np.ndarray:
    """imclearborder + bwareaopen."""
    mask = clear_border(mask)
    mask = remove_small_objects(mask, min_size=area_cutoff)
    return mask


def _zero_crossings(signal: np.ndarray) -> int:
    s = np.sign(signal)
    s[s == 0] = 1  # match dsp.ZeroCrossingDetector treating exact 0 as no independent crossing
    return int(np.sum(np.diff(s) != 0))


def classify_budding(boundary_xy: np.ndarray, ellipse) -> tuple[int, bool]:
    """boundary_xy: Nx2 array of (x, y) points, in the traced boundary
    order. Returns (zero_crossing_count, is_budding)."""
    x, y = boundary_xy[:, 0].astype(np.float64), boundary_xy[:, 1].astype(np.float64)

    n = len(x)
    window = min(SAVGOL_WINDOW, n if n % 2 == 1 else n - 1)
    if window >= SAVGOL_POLYORDER + 2 and window >= 3:
        x = savgol_filter(x, window, SAVGOL_POLYORDER, mode="interp")
        y = savgol_filter(y, window, SAVGOL_POLYORDER, mode="interp")

    n_steps = len(x) - RUN_P
    if n_steps < 2:
        return 0, False

    deriv = np.empty(n_steps)
    for i in range(n_steps):
        xpol = x[i:i + RUN_P]
        ypol = y[i:i + RUN_P]
        if np.ptp(xpol) < 1e-9:
            deriv[i] = 0.0
        else:
            slope, _ = np.polyfit(xpol, ypol, 1)
            deriv[i] = slope

    z = _zero_crossings(deriv)
    return z, z > ZERO_CROSSING_THRESHOLD


def _crop(image: np.ndarray, cx: float, cy: float, half: int = CROP_HALF) -> np.ndarray | None:
    h, w = image.shape[:2]
    x0, x1 = int(np.floor(cx - half)), int(np.floor(cx + half))
    y0, y1 = int(np.floor(cy - half)), int(np.floor(cy + half))
    if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
        return None
    return image[y0:y1 + 1, x0:x1 + 1]


def analyze_frame(mask: np.ndarray, original_image: np.ndarray) -> list[CellResult]:
    """Run the full per-cell pipeline on one segmented frame.

    Each blob is first typed by lobe count (single / budding_pair /
    multicell / ambiguous, see lobe_analysis.py):
    - single: no neck to analyze -- budding=False by construction, the
      zero-crossing classifier isn't run (nothing for it to find).
    - budding_pair: the real target case -- runs the full boundary/ellipse/
      zero-crossing classifier as before.
    - multicell / ambiguous: excluded (included=False), same as the
      original data's separate `multicell` folder. Not classified because
      a single budding/non-budding label doesn't meaningfully describe a
      3+-cell clump or two same-size cells that just happen to touch.
    """
    mask = clean_mask(mask)
    labeled = label(mask)
    results: list[CellResult] = []

    for region in regionprops(labeled):
        region_mask = labeled == region.label
        contours = find_contours(region_mask, level=0.5)
        if not contours:
            continue
        boundary_rc = max(contours, key=len)  # (row, col) = (y, x)
        boundary_xy = boundary_rc[:, ::-1]     # -> (x, y), matches MATLAB data(:,2), data(:,1)

        lobes = analyze_lobes(region_mask)

        z, budding, included = 0, False, True
        if lobes.label == "single":
            z, budding = 0, False
        elif lobes.label == "budding_pair":
            ellipse = fit_ellipse(boundary_xy[:, 0], boundary_xy[:, 1])
            if ellipse is not None and ellipse.status == "":
                z, budding = classify_budding(boundary_xy, ellipse)
        else:  # multicell or ambiguous
            included = False

        cy, cx = region.centroid  # skimage centroid is (row, col)
        crop = _crop(original_image, cx, cy)

        results.append(
            CellResult(
                label=region.label,
                centroid_xy=(cx, cy),
                area=region.area,
                major_axis_length=region.axis_major_length,
                minor_axis_length=region.axis_minor_length,
                eccentricity=region.eccentricity,
                zero_crossings=z,
                budding=budding,
                crop=crop,
                blob_type=lobes.label,
                included=included,
            )
        )
    return results
