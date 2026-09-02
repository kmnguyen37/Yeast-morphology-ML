"""
Visualize every CNN misclassification on the test set: the ORIGINAL crop
(what a human would look at) side by side with the 32x32 DOWNSAMPLED
version (what the model actually saw), labeled with the modern-classifier
ground truth, the CNN's predicted probability, and the classical
zero-crossing verdict for context.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tifffile

from dataset import _load_crop, DATA_DIR, INPUT_SIZE

ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=str(ROOT / "data" / "cnn_test_predictions.csv"))
    parser.add_argument("--out", default=str(ROOT / "data" / "cnn_misclassified.png"))
    args = parser.parse_args()

    manifest = {(int(r["frame"]), int(r["cell_label"])): r
                for r in csv.DictReader(open(DATA_DIR / "manifest.csv"))}
    preds = list(csv.DictReader(open(args.predictions)))
    wrong = [p for p in preds if int(p["true_modern_budding"]) != int(p["cnn_pred"])]
    wrong.sort(key=lambda p: (int(p["cnn_pred"]), -abs(float(p["cnn_proba"]) - 0.5)))

    n = len(wrong)
    ncols = 2  # original + downsampled, per crop
    fig, axes = plt.subplots(n, ncols, figsize=(4, 2 * n))

    for i, p in enumerate(wrong):
        key = (int(p["frame"]), int(p["cell_label"]))
        m = manifest[key]
        crop_path = m["crop_path"]
        orig = tifffile.imread(DATA_DIR / crop_path)
        small = _load_crop(crop_path)

        kind = "FALSE POS" if int(p["cnn_pred"]) == 1 else "FALSE NEG"
        title = (f"f{p['frame']}c{p['cell_label']}  {kind}\n"
                 f"true={'bud' if int(p['true_modern_budding']) else 'non'}  "
                 f"cnn_p={float(p['cnn_proba']):.2f}  "
                 f"classical={'bud' if m['classical_budding']=='True' else 'non'}")

        axes[i, 0].imshow(orig, cmap="gray")
        axes[i, 0].set_title(title, fontsize=7)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(small, cmap="gray")
        axes[i, 1].set_title(f"{INPUT_SIZE}x{INPUT_SIZE} model input", fontsize=6)
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(args.out, dpi=110)
    print(f"saved {args.out}  ({n} misclassified crops)")


if __name__ == "__main__":
    main()
