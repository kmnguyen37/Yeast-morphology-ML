# The from-scratch CNN: build log

## Why from scratch

The plan was to use Cellpose for segmentation and a standard framework (TensorFlow or
PyTorch) for the classifier. Neither could actually be installed: this build
environment's network access didn't reach PyPI, in either the cloud workspace or the
local shell on the Mac this project syncs to — confirmed by direct `pip install`
failures in both places, not assumed. Rather than block on that, the decision (made
explicitly, not as a fallback) was to implement the CNN's forward *and backward*
passes by hand in NumPy — convolutions, pooling, dense layers, the loss, and an Adam
optimizer, no autograd — and keep segmentation classical rather than depend on a
pretrained model at all.

## Architecture

```
Conv(1→8, 5×5) → ReLU → MaxPool(2×2)
Conv(8→16, 3×3) → ReLU → MaxPool(2×2)
Flatten [→ concat 7 shape features, v3 only]
Dense(→32) → ReLU → Dense(→1) → sigmoid (binary cross-entropy loss)
```

Deliberately small: with ~1,190 labeled crops total (not ImageNet scale), two conv
layers is enough to pick up "is there a neck" without seriously overfitting. Convolution
uses an im2col/col2im formulation (`numpy.lib.stride_tricks.as_strided` for zero-copy
sliding windows) rather than a python loop over every pixel, since raw-python conv
would be too slow to iterate with in pure NumPy.

## Verifying the backward pass before trusting it

A hand-derived backward pass is exactly the kind of code that can look like it's
learning (loss goes down) while actually being wrong (a sign error nets out over many
steps, or a bug only shows up on some input shapes). Before any real training run,
every layer was checked against numerical gradient checking: perturb one parameter by
±1e-4, measure the change in loss, compare that finite-difference estimate to what
`backward()` computed analytically.

```
Conv2D(3->4,k3)      d/dx   max rel error: 1.60e-09  OK
Conv2D(3->4,k3)      d/dW   max rel error: 5.90e-11  OK
MaxPool2D            d/dx   max rel error: 5.83e-09  OK
Dense(10->4)         d/dW   max rel error: 9.36e-11  OK
ReLU                 d/dx   max rel error: 3.17e-11  OK
bce_with_logits      d/dlogits max rel error: 1.97e-10  OK
--- end-to-end, full model, both YeastCNN and HybridYeastCNN ---
conv1.W / conv2.W / fc1.W / fc2.W  all  ~1e-9 - 1e-10 rel error  OK
```

(Full output: `docs/gradcheck_output.txt`, reproducible via `scripts/gradcheck.py`.)
Every check passed at ~1e-9–1e-10 relative error — well below the ~1e-4 threshold
that would flag a real bug — before a single training epoch ran.

## Data split: by frame, not by crop

`scripts/split_dataset.py` splits the 100 raw frames 70/15/15 into train/val/test,
*then* takes every crop from each frame's cells into that frame's split — not a
random per-crop split. Reasoning: crops from the same frame share lighting,
background texture, and often a similar size range of cells. A random per-crop split
would let, say, a mother cell's frame be in train while a visually similar cell one
frame later ends up in test — leakage that inflates validation accuracy without the
model actually having learned to generalize to a new field of view. Splitting whole
frames avoids that: every val/test crop comes from a frame the model never saw in any
form. Of 200 random seeds tried, the one producing the most balanced budding rate
across splits was kept (39.2–39.6% budding in every split, vs. 39.5% overall).

## v1 → v2 → v3

**v1** (32×32 input, no augmentation): 88.3% test accuracy, F1 85.5%, 23/197 errors.
Baseline: does a from-scratch CNN work at all on this data. It did.

**v2** (48×48 input + random 90°-rotation/flip augmentation during training): 90.4%
accuracy, F1 88.1%, 19/197 errors. Two changes, both motivated by reviewing every v1
misclassification crop by eye rather than guessing: small early-stage buds were
shrinking to a handful of pixels and nearly vanishing at 32×32 (more resolution should
help), and yeast cells have no canonical orientation under the microscope so rotation/
flip is free extra training variety, not distortion.

**v3 — final** (48×48 pixels + 7 hand-engineered shape features, concatenated onto the
flattened conv output before the dense layers): **94.4% accuracy, F1 92.9%, 11/197
errors**. Reviewing v2's remaining 19 misclassifications showed two clear, *specific*
failure modes, not random noise:

- **False negatives** (8 of 19): almost all a small, faint bud tucked against a much
  larger mother cell — visible in the crop but reduced to a handful of ambiguous
  pixels after downsampling.
- **False positives** (11 of 19): almost all elongated or unevenly-lit single cells.
  A pixels-only model has no direct signal for "is there a concave neck" — it can
  only approximate that from brightness patterns, which an elongated or shadowed
  single cell can accidentally resemble.

Both failure modes point at the same missing ingredient: geometric information the
conv layers never get to see directly, but that `src/shape_features.py` already
computes precisely from the segmented boundary (the same 7 features the *modern*
classical classifier uses: area, eccentricity, solidity, extent, aspect ratio, defect
count, and max convexity-defect depth). v3 (`HybridYeastCNN` in `src/cnn.py`)
concatenates those 7 numbers (standardized using train-set mean/std only, to avoid
leaking val/test statistics into training) onto the flattened conv features before
`fc1`, so the network has both signals and learns how to weigh them. It is not a
fallback to hand-engineered features instead of a CNN — the conv trunk is unchanged
and still does the pixel-level work; this only adds what pixels alone can't provide.

Note: the 7 shape features are rotation/flip invariant by construction (eccentricity,
solidity, etc. don't change if you rotate the cell), so — unlike the pixel input —
they're **not** augmented; only the pixel branch gets the random rotation/flip each
epoch.

### What v3 fixed, and what's still hard

Re-reviewing the misclassifications after the hybrid model: false positives roughly
halved (11 → 5), consistent with the model now having direct access to the concavity
signal it was missing. False negatives dropped from 8 to 6, but **the same 6 cells**
persisted across all three versions — the same tiny, faint, early-stage buds every
time, now with lower-confidence (0.24–0.48) rather than confidently-wrong predictions.
That consistency across three otherwise-different models is itself informative: it
looks like a real ceiling set by how much bud is actually resolvable in a 129×129
source crop at this stage of budding, not a gap any of these three approaches happens
to have. One of the remaining false positives (`frame64 cell4`) is a cell where the
classical and modern classifiers themselves disagree — genuinely ambiguous ground
truth, not a clean model error.

| | v1 (32px) | v2 (48px + aug) | v3 (hybrid) |
|---|---|---|---|
| test accuracy | 88.3% | 90.4% | **94.4%** |
| precision | 84.0% | 86.4% | **93.5%** |
| recall | 87.2% | 89.7% | **92.3%** |
| F1 | 85.5% | 88.1% | **92.9%** |
| test errors | 23/197 | 19/197 | **11/197** |
| false positives | 13 | 11 | **5** |
| false negatives | 10 | 8 | **6** |

Misclassified-crop galleries: `results/baseline_cnn/misclassified.png` (v2) and
`results/hybrid_cnn/misclassified.png` (v3) — each crop shown at both its original
resolution and the actual downsampled model input, with the true label, predicted
probability, and the classical method's verdict for context.

## Training details

Mini-batch SGD via Adam (lr 1e-3), batch size 32, up to 60 epochs with early stopping
on validation loss (patience 8, best-val-loss weights restored at the end — not the
final epoch's weights). Both v2 and v3 stopped around epoch 24–29. Full per-epoch
logs: `results/baseline_cnn/training_log.txt`, `results/hybrid_cnn/training_log.txt`.
