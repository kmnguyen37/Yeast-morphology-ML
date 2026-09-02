# Week 1 findings — segmentation + classical classifier port

Built and validated against a 15-frame sample (Set1, frames 12-26) staged from
the original data: `src/segmentation.py`, `src/ellipse_fit.py`, `src/features.py`.

## Segmentation (ImageJ macro -> Python)

Ported `fill_images.txt` (8-bit -> contrast enhance -> Find Edges -> invert ->
Otsu -> fill holes -> erode) to scikit-image/scipy. On frame 12: **IoU 0.79**
against the original ImageJ-generated mask; visually the two masks pick out
the same cells in the same positions (see `data/sample/seg_compare.png`).
The gap is mostly edge-thickness/erosion differences — ImageJ's exact Sobel
and erosion implementations aren't identically reproducible from outside,
which is expected and worth stating plainly rather than chasing a perfect
match.

## Classical classifier (MATLAB -> Python)

Ported `fit_ellipse.m` and the region-analysis/classification loop from
`create_yeast_training_sets.m`: clear border, drop cells under 200px, fit an
ellipse per cell, smooth the boundary (Savitzky-Golay, window 27), compute
local slope over a 6-point sliding window, count zero-crossings, classify
budding if count > 4, crop a 129x129 patch per cell.

**Found and preserved a quirk in the original code**: the script computes a
rotated version of each cell's boundary (`xnew`/`ynew`, meant to align the
ellipse's major axis to the x-axis, matching the thesis text's description),
but the classification loop actually runs on the smoothed *unrotated*
boundary — the rotated version is computed and never used. We reproduced
this behavior exactly rather than "fixing" it, so our output matches the
original algorithm's actual behavior, not its documented intent. Worth a
callout in the write-up: this is the kind of discrepancy a fresh
implementation surfaces that's easy to miss in the original.

On the 15-frame sample: 229 cells detected, 115 classified budding / 114
non-budding — in the same ballpark as the original Set1 output (659
budding / 417 non-budding across all ~100 frames, so roughly 61/39 vs. our
50/50 on this smaller sample; not identical, expected given segmentation
differences compound into which cells get detected at all).

**Visual comparison** (`data/sample/crop_compare.png`) of our port's output
against the original crops for the same class is the more interesting
result: our port reproduces the *same characteristic error mode* as the
original -- both misclassify some touching mother-daughter pairs as
non-budding and occasionally flag a single round cell as budding. That's
actually the right validation outcome at this stage: it shows the port is
behaviorally faithful to the original zero-crossing heuristic, including
its weaknesses, not just its correct answers. Those weaknesses are exactly
what Week 2's CNN comparison should be evaluated against.

## Known limitation to flag in the write-up

Found one corrupted/mislabeled file in the original data during validation:
`YPD-0/Set1/budding/bud102.tif` is actually a `.DS_Store`-type file, not a
TIFF. Worth a data-quality note and a validation step in the final pipeline
that filters non-image files before use.

## Next

- Week 2: rebuild the CNN in Python on the existing training_set/testing_set
  crops is explicitly OUT per Iris's direction -- instead, generate our own
  training/testing set by running the full Python pipeline (segmentation +
  classification + crop) across the raw frames, and validate *that* output
  against what she already has, the same way we validated Week 1.
- Cellpose benchmark against the classical segmentation.
- Proper train/val/test split + precision/recall/F1/confusion matrix once
  a from-scratch dataset exists.
