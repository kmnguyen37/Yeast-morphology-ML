"""
Split the generated dataset into train/val/test by FRAME, not by crop.

Why by frame: crops from the same frame share lighting, background
texture, and often the same handful of cells' general size range. A
random per-crop split would let the model see cell A's mother in train
and cell A's daughter-frame-later in test -- leakage that inflates
validation accuracy without the model actually generalizing to a new
field of view. Splitting whole frames into whole splits means every
crop in val/test comes from a frame the model never saw during training.

Frames are shuffled (fixed seed) then cut 70/15/15. Since 100 frames
split unevenly by budding-fraction, we try a handful of seeds and keep
the one whose splits are closest to the overall 39.5% budding rate --
still a random split, just not letting bad luck starve one split of
positives.
"""
import csv
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
MANIFEST = ROOT / "data" / "generated_dataset" / "manifest.csv"
OUT = ROOT / "data" / "generated_dataset" / "splits.csv"

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # remainder (0.15) -> test
N_SEED_TRIES = 200


def budding_rate(rows):
    if not rows:
        return 0.0
    return sum(1 for r in rows if r["modern_budding"] == "True") / len(rows)


def main():
    rows = list(csv.DictReader(open(MANIFEST)))
    included = [r for r in rows if r["included"] == "True" and r["crop_path"]]
    overall_rate = budding_rate(included)

    by_frame = {}
    for r in included:
        by_frame.setdefault(int(r["frame"]), []).append(r)
    frames = sorted(by_frame)
    n = len(frames)
    n_train = round(n * TRAIN_FRAC)
    n_val = round(n * VAL_FRAC)

    best = None
    for seed in range(N_SEED_TRIES):
        rng = random.Random(seed)
        shuffled = frames[:]
        rng.shuffle(shuffled)
        train_f, val_f, test_f = shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]

        splits = {
            "train": [r for f in train_f for r in by_frame[f]],
            "val": [r for f in val_f for r in by_frame[f]],
            "test": [r for f in test_f for r in by_frame[f]],
        }
        max_dev = max(abs(budding_rate(splits[s]) - overall_rate) for s in splits)
        if best is None or max_dev < best[0]:
            best = (max_dev, seed, train_f, val_f, test_f, splits)

    max_dev, seed, train_f, val_f, test_f, splits = best

    with open(OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "cell_label", "split", "crop_path", "modern_budding"])
        for split_name, split_frames in [("train", train_f), ("val", val_f), ("test", test_f)]:
            for fr in split_frames:
                for r in by_frame[fr]:
                    writer.writerow([r["frame"], r["cell_label"], split_name, r["crop_path"], r["modern_budding"]])

    print(f"seed={seed} (best of {N_SEED_TRIES}), max budding-rate deviation={max_dev:.3f}")
    print(f"overall budding rate: {overall_rate:.1%}")
    for name, split_frames in [("train", train_f), ("val", val_f), ("test", test_f)]:
        rows_s = splits[name]
        print(f"{name:5s}: {len(split_frames):3d} frames, {len(rows_s):4d} crops, "
              f"{budding_rate(rows_s):.1%} budding")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
