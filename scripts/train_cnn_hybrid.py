"""
Train HybridYeastCNN: same conv trunk + augmentation as v2's train_cnn.py,
plus 7 hand-engineered shape features (from shape_features.py) concatenated
in before the dense layers. See src/cnn.py's HybridYeastCNN docstring for
why -- this targets the two specific failure modes v2's misclassified-crop
review found (elongated/uneven singles read as budding; faint small buds
missed).

Aux features are standardized using TRAIN-set mean/std only, then that same
transform is applied to val/test -- fitting on train only avoids leaking
val/test distribution info into training.
"""
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from cnn import HybridYeastCNN, Adam, bce_with_logits
from dataset import load_split_hybrid, augment_batch, INPUT_SIZE, AUX_FEATURE_NAMES

BATCH_SIZE = 32
MAX_EPOCHS = 60
PATIENCE = 8
LR = 1e-3
SEED = 0
USE_AUGMENTATION = True


def iterate_batches(X, aux, y, batch_size, rng):
    idx = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        b = idx[start:start + batch_size]
        yield X[b], aux[b], y[b]


def evaluate(model, X, aux, y):
    logits = model.forward(X, aux)
    loss, _ = bce_with_logits(logits, y)
    proba = 1 / (1 + np.exp(-logits.reshape(-1)))
    pred = (proba >= 0.5).astype(np.float64)
    acc = (pred == y).mean()
    tp = ((pred == 1) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum()
    fn = ((pred == 0) & (y == 1)).sum()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(loss=loss, acc=acc, precision=precision, recall=recall, f1=f1, proba=proba, pred=pred)


def snapshot(model):
    return {l: {"W": layer.W.copy(), "b": layer.b.copy()}
            for l, layer in zip(("conv1", "conv2", "fc1", "fc2"),
                                 (model.conv1, model.conv2, model.fc1, model.fc2))}


def restore(model, snap):
    for name, layer in zip(("conv1", "conv2", "fc1", "fc2"),
                            (model.conv1, model.conv2, model.fc1, model.fc2)):
        layer.W[:] = snap[name]["W"]
        layer.b[:] = snap[name]["b"]


def main():
    print("loading data...")
    Xtr, auxtr, ytr, _ = load_split_hybrid("train")
    Xval, auxval, yval, _ = load_split_hybrid("val")
    Xtest, auxtest, ytest, meta_test = load_split_hybrid("test")
    print(f"train {Xtr.shape}, val {Xval.shape}, test {Xtest.shape}")

    aux_mean, aux_std = auxtr.mean(axis=0), auxtr.std(axis=0)
    aux_std[aux_std < 1e-8] = 1.0  # guard against a constant feature
    auxtr = (auxtr - aux_mean) / aux_std
    auxval = (auxval - aux_mean) / aux_std
    auxtest = (auxtest - aux_mean) / aux_std
    print("aux feature means (train):", dict(zip(AUX_FEATURE_NAMES, aux_mean.round(2))))

    model = HybridYeastCNN(input_size=INPUT_SIZE, aux_dim=len(AUX_FEATURE_NAMES), seed=SEED)
    opt = Adam(model.layers, lr=LR)
    rng = np.random.default_rng(SEED)

    best_val_loss = np.inf
    best_snap = None
    epochs_no_improve = 0
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        for xb, auxb, yb in iterate_batches(Xtr, auxtr, ytr, BATCH_SIZE, rng):
            if USE_AUGMENTATION:
                # rotation/flip on pixels only -- the aux shape features
                # (eccentricity, solidity, aspect ratio, defect depth...)
                # are already rotation/flip invariant by construction, so
                # they don't need (and shouldn't get) separate augmentation
                xb = augment_batch(xb, rng)
            logits = model.forward(xb, auxb)
            loss, dlogits = bce_with_logits(logits, yb)
            model.backward(dlogits)
            opt.step()

        train_metrics = evaluate(model, Xtr, auxtr, ytr)
        val_metrics = evaluate(model, Xval, auxval, yval)
        print(f"epoch {epoch:3d}  train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.3f}  "
              f"|  val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.3f} f1 {val_metrics['f1']:.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

        if val_metrics["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_metrics["loss"]
            best_snap = snapshot(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
                break

    restore(model, best_snap)
    print(f"\nrestored best-val-loss weights (val loss {best_val_loss:.4f})")

    print("\n=== TEST SET (held out, never seen in training) ===")
    test_metrics = evaluate(model, Xtest, auxtest, ytest)
    print(f"loss {test_metrics['loss']:.4f}  acc {test_metrics['acc']:.3f}  "
          f"precision {test_metrics['precision']:.3f}  recall {test_metrics['recall']:.3f}  "
          f"f1 {test_metrics['f1']:.3f}")

    out_dir = Path(__file__).parent.parent / "data"
    np.savez(out_dir / "cnn_hybrid_weights.npz",
              conv1_W=model.conv1.W, conv1_b=model.conv1.b,
              conv2_W=model.conv2.W, conv2_b=model.conv2.b,
              fc1_W=model.fc1.W, fc1_b=model.fc1.b,
              fc2_W=model.fc2.W, fc2_b=model.fc2.b,
              aux_mean=aux_mean, aux_std=aux_std)
    print(f"saved weights to {out_dir / 'cnn_hybrid_weights.npz'}")

    import csv
    pred_path = out_dir / "cnn_hybrid_test_predictions.csv"
    with open(pred_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "cell_label", "true_modern_budding", "cnn_proba", "cnn_pred"])
        for (frame, cell_label), yt, p, pr in zip(meta_test, ytest, test_metrics["proba"], test_metrics["pred"]):
            writer.writerow([frame, cell_label, int(yt), round(float(p), 4), int(pr)])
    print(f"saved test predictions to {pred_path}")


if __name__ == "__main__":
    main()
