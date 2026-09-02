"""
Recompute the hand-engineered ShapeFeatures (from shape_features.py, via
the modern/convexity-defect classifier) for every included cell across all
100 frames, and save them keyed by (frame, cell_label) so they can be
joined against splits.csv as an auxiliary input to the hybrid CNN.

Re-running segmentation here (rather than reading it back from anywhere)
keeps this consistent with how the crops themselves were generated --
same pipeline, same masks, just capturing a bit of already-computed
per-cell geometry that build_dataset.py's manifest didn't retain in full
(it only kept max_defect_depth_ratio, not the other 5 shape features).
"""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tifffile

from segmentation import segment
from modern_classifier import analyze_frame_modern

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "sample" / "Set1"
OUT = ROOT / "data" / "generated_dataset" / "aux_features.csv"

FEATURE_NAMES = ["area", "eccentricity", "solidity", "extent", "aspect_ratio",
                  "n_defects", "max_defect_depth_ratio"]


def main():
    t0 = time.time()
    with open(OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "cell_label"] + FEATURE_NAMES)

        for frame in range(1, 101):
            raw = tifffile.imread(RAW_DIR / f"Image{frame}.tif")
            mask = segment(raw)
            results = analyze_frame_modern(mask, raw)
            for r in results:
                if not r.included:
                    continue
                feats = r.features
                writer.writerow([
                    frame, r.label, feats.area, round(feats.eccentricity, 4),
                    round(feats.solidity, 4), round(feats.extent, 4),
                    round(feats.aspect_ratio, 4), feats.n_defects,
                    round(feats.max_defect_depth_ratio, 4),
                ])
            if frame % 10 == 0:
                print(f"frame {frame} done, elapsed {time.time()-t0:.0f}s", flush=True)

    print(f"DONE. wrote {OUT}")


if __name__ == "__main__":
    main()
