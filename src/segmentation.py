"""
Classical yeast-cell segmentation.

Python port of the original ImageJ macro (fill_images.txt):

    run("8-bit")
    run("Enhance Contrast", "saturated=0.5")
    run("Find Edges")
    run("Invert")
    setAutoThreshold("Otsu")
    run("Convert to Mask")
    run("Fill Holes")
    run("Erode")
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sobel, threshold_otsu
from skimage.morphology import erosion, closing, disk


def to_8bit(img: np.ndarray) -> np.ndarray:
    """ImageJ's 'Image > Type > 8-bit': linear scale from the image's own
    min/max to 0-255 (ImageJ's default display-range behavior when no
    explicit min/max has been set)."""
    img = img.astype(np.float64)
    lo, hi = img.min(), img.max()
    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint8)
    scaled = (img - lo) / (hi - lo) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def enhance_contrast(img: np.ndarray, saturated: float = 0.5) -> np.ndarray:
    """ImageJ 'Enhance Contrast' with a given % saturated pixels: clip the
    saturated/2 percentile at each tail, then stretch to the full 0-255
    range (ImageJ normalizes when 'Normalize' is implied by later steps;
    here we replicate the equalized-range stretch)."""
    lo_pct = saturated / 2
    hi_pct = 100 - saturated / 2
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    if hi <= lo:
        return img
    stretched = (img.astype(np.float64) - lo) / (hi - lo) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)


def find_edges(img: np.ndarray) -> np.ndarray:
    """ImageJ 'Find Edges' = Sobel edge filter."""
    edges = sobel(img.astype(np.float64) / 255.0)
    edges = edges / edges.max() * 255.0 if edges.max() > 0 else edges
    return edges.astype(np.uint8)


def invert(img: np.ndarray) -> np.ndarray:
    return 255 - img


def segment(
    raw: np.ndarray,
    saturated: float = 0.5,
    erode_iterations: int = 1,
    close_radius: int = 1,
) -> np.ndarray:
    """Full segmentation pipeline. Returns a boolean mask (True = cell).

    close_radius: a morphological closing (dilate-then-erode) applied right
    after thresholding, before fill_holes. Fixes a specific failure mode our
    Sobel-based edge detector has that the original ImageJ macro apparently
    doesn't: a real optical ridge at the neck between a budding mother and
    daughter cell gets picked up as its own closed edge loop, splitting what
    should be one connected blob into two. Closing bridges that thin gap
    without meaningfully distorting a real single cell's outer boundary.
    Set close_radius=0 to disable and reproduce the earlier (splitting)
    behavior.
    """
    img8 = to_8bit(raw)
    img8 = enhance_contrast(img8, saturated=saturated)
    edges = find_edges(img8)
    inverted = invert(edges)

    thresh = threshold_otsu(inverted)
    # ImageJ's Otsu on an *inverted* edge image + "Convert to Mask" keeps
    # pixels *below* threshold as foreground on this kind of image (edges
    # of cells become dark after inversion where contrast was strongest).
    mask = inverted < thresh

    if close_radius > 0:
        mask = closing(mask, disk(close_radius))

    mask = ndi.binary_fill_holes(mask)

    selem = disk(1)  # ImageJ's default Erode uses a 3x3 (radius-1) neighborhood
    for _ in range(erode_iterations):
        mask = erosion(mask, selem)

    return mask
