# Build plan (target: ~3 weeks)

## Week 1 — Data + classical pipeline
- [ ] Inventory original data: raw phase-contrast images, ImageJ macro, MATLAB
      scripts (cell length/eccentricity extraction, ellipse fitting), the ~6,140
      cropped training images for the CNN
- [ ] Set up repo: package structure, requirements.txt, pre-commit/lint, license
- [ ] Port ImageJ watershed segmentation macro to Python (scikit-image/OpenCV)
- [ ] Validate: segmentation output matches original ImageJ output on a sample set
      (cell counts, mask overlap)
- [ ] Port MATLAB ellipse-fit + local-curvature feature extraction to Python
- [ ] Port the classical budding/non-budding rule (curvature zero-crossing count)
- [ ] Validate against thesis figures/numbers on a known subset

## Week 2 — CNN + Cellpose benchmark
- [ ] Rebuild the CNN in Python (same architecture as thesis: 3 conv + 3 pool +
      4 dense, Adam, sparse categorical crossentropy) with a real
      train/val/test split (thesis only had train/val)
- [ ] Add proper metrics: accuracy, precision/recall/F1, confusion matrix
      (thesis reported accuracy/loss curves only)
- [ ] Error analysis: which cells does the CNN get wrong, and why
- [ ] Run Cellpose on the same raw images; compare segmentation quality
      (IoU/Dice against classical + against manual/ImageJ masks) and runtime
- [ ] Write up the classical-vs-CNN and classical-vs-Cellpose comparisons

## Week 3 — Packaging + demo + write-up
- [ ] Clean up repo: docstrings, README polish, example notebook, unit tests
      for segmentation/feature-extraction correctness
- [ ] Build the interactive demo (Streamlit/Gradio): upload an image, see
      segmentation + both classifiers' predictions side by side
- [ ] Deploy demo somewhere clickable (Streamlit Community Cloud / HF Spaces)
- [ ] Write the blog post / narrative write-up
- [ ] Final polish pass, push to GitHub, add to portfolio/resume

## Scope decision
Yeast budding classification only. Bacteria cell-length pipeline (thesis
ch. 2-4) is explicitly out of scope for this project.

## Open questions to resolve before Week 1 starts
- Where do the original raw images + ImageJ macro + MATLAB scripts currently
  live (this Mac, an external drive, lab servers, cloud storage)?
