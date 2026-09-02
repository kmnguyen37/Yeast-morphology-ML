# Yeast Morphology ML

A from-scratch Python rebuild of one piece of Khanh Mai Nguyen's (Iris Johnson) Ph.D. dissertation,
*Microorganisms in Extreme Environmental Conditions* (Physics/Biophysics, University
of Arkansas, 2023): classifying budding vs. non-budding *S. cerevisiae* cells from
phase-contrast microscopy images.

The original pipeline used three separate tools — an ImageJ macro for segmentation,
a MATLAB script for per-cell shape analysis and classification, and a Keras CNN. This
project consolidates all of it into a single Python pipeline, built and validated one
piece at a time, with every classifier evaluated against a real held-out test set
(the original work didn't need one) and a from-scratch neural network built without
any ML framework.

## Why this project

Most portfolio projects use public toy datasets with a known right answer. This one
starts from real research data and an existing "reference" pipeline, so every new
piece of code could be checked against something concrete: does the Python port of
the ImageJ macro produce the same masks? Does the ported MATLAB classifier reproduce
the same characteristic mistakes as the original? When the model changes, does the
change actually fix a real, inspectable error, or just move the number?

## Pipeline

```
raw phase-contrast frame
        │
        ▼
  segmentation (src/segmentation.py)        8-bit → contrast stretch → Sobel edges →
        │                                   Otsu threshold → morphological closing →
        │                                   fill holes → erode
        ▼
  blob typing (src/lobe_analysis.py)        distance-transform peak counting →
        │                                   single / budding_pair / multicell / ambiguous
        ▼
  per-cell classification, two ways, run head-to-head on every budding_pair blob:
        │
        ├─ classical (src/features.py)      ellipse fit → boundary smoothing →
        │                                   sliding-window slope → zero-crossing count
        │                                   (faithful port of the original MATLAB method)
        │
        └─ modern (src/shape_features.py +  convexity-defect depth on the boundary
           src/modern_classifier.py)        vs. its convex hull — one interpretable
                                             threshold on neck depth
        ▼
  129×129 crop saved per classifiable cell → data/generated_dataset/
        ▼
  from-scratch NumPy CNN (src/cnn.py)       trained on the crops (see below)
```

Everything is original code end to end. No pretrained models, and — since the network
access this project's build environment had didn't reach PyPI, so nothing outside the
Python standard scientific stack (numpy/scipy/scikit-image/opencv) could actually be
installed — no TensorFlow, PyTorch, or Cellpose either. That constraint pushed the
last stage toward writing the CNN's forward *and backward* passes by hand instead of
calling a framework, which turned into the most interesting part of the project (see
`docs/CNN_FINDINGS.md`).

## What's a genuinely fresh dataset, not a reused one

By design, `data/generated_dataset/` is **not** the original thesis's training/testing
folders. It's built by running this repo's own segmentation + blob-typing pipeline
across all 100 raw frames of Set1 (`scripts/build_dataset.py`), so every crop and
every label in this repo traces back to code in this repo, not to a black-box folder
inherited from the original project. Across the full run: 1,281 classifiable cells
(770 single, 511 budding pairs; 65 clear multicell clumps and 75 ambiguous same-size
touching pairs excluded — see **Known limitations** below), with the classical and
modern classifiers agreeing on 97.9% of them.

## A real bug, found and fixed

Early validation surfaced a genuine segmentation bug: the Sobel-based edge detector
sometimes traces the real optical ridge at a budding neck as its own closed loop,
splitting one mother-daughter blob into two separate regions after hole-filling. Root-
caused by direct comparison against the original ImageJ masks (which stayed correctly
connected), then fixed with a small morphological closing step before hole-filling.
Trade-off, measured directly: closing improves neck-preservation but costs a small
amount of overall mask IoU against the ImageJ ground truth (0.798 → 0.776 across a
15-frame validation sample) — a real accuracy-vs-correctness trade, made explicitly
and documented rather than silently tuned away. Details in `docs/WEEK1_FINDINGS.md`.

## Classical vs. modern classical

The original method counts zero-crossings in the local slope of a smoothed cell
boundary — a working heuristic, but with an arbitrary threshold and two documented
failure modes: it misses budding pairs where the bud is small, and it occasionally
flags boundary noise on a single round cell as budding. The **modern** classifier
replaces that with one number that has a real geometric meaning: how deep the
boundary's biggest concavity is, relative to cell size, via OpenCV convexity defects.
Head-to-head on real disagreement cases, it fixes 2 of 3 (a missed budding pair, a
false-positive single cell) and introduces one known miss of its own (a bud tucked
very close to the mother, not enough concavity to clear the threshold). Full analysis
in `docs/WEEK1B_FINDINGS.md`.

## The from-scratch CNN

Three iterations, each built to fix a specific, inspected failure mode rather than
just re-tuning hyperparameters and hoping. Held-out **test set** results (197 crops
from 15 frames never seen in any form during training — the split is by *frame*, not
by crop, so no field-of-view leaks between train and test):

| version | input | test accuracy | precision | recall | F1 | test errors |
|---|---|---|---|---|---|---|
| v1 | 32×32 pixels | 88.3% | 84.0% | 87.2% | 85.5% | 23 / 197 |
| v2 | 48×48 pixels + rotation/flip augmentation | 90.4% | 86.4% | 89.7% | 88.1% | 19 / 197 |
| **v3 (final)** | 48×48 pixels **+ 7 hand-engineered shape features** | **94.4%** | **93.5%** | **92.3%** | **92.9%** | **11 / 197** |

Every layer's backward pass (both conv layers, both dense layers, ReLU, max-pool, the
BCE loss) is verified against numerical gradient checking before any training run —
finite-difference and analytic gradients agree to ~1e-9–1e-10 relative error across
the board (`docs/gradcheck_output.txt`). v3's jump comes from a real diagnosis, not a
bigger model: reviewing every v2 misclassification crop by eye showed the errors
weren't random — false negatives were almost all faint, tiny early-stage buds, and
false positives were almost all elongated or unevenly-lit single cells being read as
"bud-like" from pixels alone. v3 concatenates the same 7 geometric features the
modern classical classifier uses (area, eccentricity, solidity, extent, aspect ratio,
defect count, max defect depth) onto the flattened conv output before the dense
layers — giving the network direct access to the one thing a pixels-only model
structurally can't see: boundary concavity. Full build log, architecture reasoning,
and before/after misclassification galleries in `docs/CNN_FINDINGS.md`.

## Known limitations

- **Multicell clumps (3+ touching cells) are excluded, not decomposed.** ~5% of
  detected blobs. Splitting them (watershed + a pairwise merge-decision heuristic)
  is a well-scoped v2 feature, deliberately deferred until the core single/
  budding_pair pipeline was solid — see `docs/PLAN.md`.
- **The remaining CNN errors are almost entirely one kind of cell**: a handful of
  pixels' worth of bud, at or near the practical resolution floor of a 129×129 source
  crop. This looks like a real ceiling for a pixel-based approach on this crop size,
  not a tuning gap.
- Segmentation isn't a pixel-perfect match to the original ImageJ output (see the
  IoU trade-off above) — expected, since ImageJ's exact Sobel/erode implementations
  aren't independently reproducible, and documented rather than hidden.

## Running it

```bash
pip install -r requirements.txt

python scripts/split_dataset.py          # frame-level train/val/test split
python scripts/compute_aux_features.py   # hand-engineered shape features (for v3)
python scripts/gradcheck.py              # verify the from-scratch backprop before training
python scripts/train_cnn.py              # baseline pixel-only CNN (v1/v2 architecture)
python scripts/train_cnn_hybrid.py       # final hybrid CNN (v3)
python scripts/visualize_misclassified.py --predictions results/hybrid_cnn/test_predictions.csv --out /tmp/misclassified.png
```

`scripts/build_dataset.py` regenerates `data/generated_dataset/` from raw frames in
`data/sample/Set1/` (not included in this repo — original unpublished research data).

## Project layout

```
src/            pipeline code: segmentation, ellipse fitting, both classifiers,
                shape features, blob typing, the from-scratch CNN, data loading
scripts/        build/split/train/evaluate/visualize entry points
data/           generated_dataset/ — the fresh crop dataset + manifest + splits
results/        baseline_cnn/ and hybrid_cnn/ — trained weights, test predictions,
                training logs, misclassified-crop galleries
docs/           the actual build log: findings, trade-offs, and the CNN write-up,
                written as the work happened rather than after the fact
```

## License

Code is MIT-licensed (see `LICENSE`). The crop images under `data/generated_dataset/`
are derived from unpublished research data and are included here for portfolio/
reproducibility purposes, not under an open license of their own.

## Acknowledgments

Built on top of unpublished data and methodology from Khanh Mai Nguyen's Ph.D.
dissertation (University of Arkansas, 2023) — the original ImageJ macro and MATLAB
scripts are the reference this project's Python pipeline was validated against, not
included here as they aren't this project's code.
