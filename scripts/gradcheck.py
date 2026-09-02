"""
Numerical gradient check for every hand-written layer in src/cnn.py.

For each parameter (and the input), perturb one scalar by +-eps, measure
the change in loss, and compare that finite-difference estimate against
the analytic gradient our backward() computed. This is the standard way
to catch a sign error / wrong transpose in a from-scratch backprop
implementation before trusting it for real training.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from cnn import Conv2D, MaxPool2D, Dense, ReLU, Flatten, bce_with_logits, YeastCNN, HybridYeastCNN

rng = np.random.default_rng(0)
EPS = 1e-4


def rel_error(a, b):
    return np.abs(a - b) / (np.maximum(np.abs(a), np.abs(b)) + 1e-8)


def numerical_grad(f, param, eps=EPS):
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = param[idx]
        param[idx] = orig + eps
        plus = f()
        param[idx] = orig - eps
        minus = f()
        param[idx] = orig
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


def check_layer(name, layer, x, dout_shape_fn=None):
    out = layer.forward(x.copy())
    dout = rng.normal(size=out.shape)
    din_analytic = layer.backward(dout.copy())

    def loss_fn():
        return np.sum(layer.forward(x) * dout)

    din_numeric = numerical_grad(loss_fn, x)
    err = rel_error(din_analytic, din_numeric).max()
    print(f"{name:20s} d/dx   max rel error: {err:.2e}  {'OK' if err < 1e-4 else 'FAIL'}")

    for pname in ("W", "b"):
        if not hasattr(layer, pname):
            continue
        param = getattr(layer, pname)

        def loss_fn_p():
            return np.sum(layer.forward(x) * dout)

        # re-run forward/backward since x didn't change but we need fresh cache
        layer.forward(x)
        layer.backward(dout)
        analytic = getattr(layer, f"d{pname}").copy()
        numeric = numerical_grad(loss_fn_p, param)
        err = rel_error(analytic, numeric).max()
        print(f"{name:20s} d/d{pname}   max rel error: {err:.2e}  {'OK' if err < 1e-4 else 'FAIL'}")


def main():
    print("=== per-layer gradient checks (small random inputs) ===")
    x_conv = rng.normal(size=(2, 3, 9, 9))
    check_layer("Conv2D(3->4,k3)", Conv2D(3, 4, 3, seed=1), x_conv)

    x_pool = rng.normal(size=(2, 3, 8, 8))
    check_layer("MaxPool2D", MaxPool2D(2), x_pool)

    x_dense = rng.normal(size=(5, 10))
    check_layer("Dense(10->4)", Dense(10, 4, seed=2), x_dense)

    x_relu = rng.normal(size=(5, 10))
    check_layer("ReLU", ReLU(), x_relu)

    print("\n=== BCE-with-logits gradient check ===")
    logits = rng.normal(size=(6,)) * 2
    y = rng.integers(0, 2, size=6).astype(np.float64)
    loss, grad = bce_with_logits(logits, y)

    def bce_loss_only(z):
        m = np.maximum(z, 0)
        return (m - z * y + np.log1p(np.exp(-np.abs(z)))).mean()

    num_grad = np.zeros_like(logits)
    for i in range(len(logits)):
        lp, lm = logits.copy(), logits.copy()
        lp[i] += EPS
        lm[i] -= EPS
        num_grad[i] = (bce_loss_only(lp) - bce_loss_only(lm)) / (2 * EPS)
    err = rel_error(grad.reshape(-1), num_grad).max()
    print(f"{'bce_with_logits':20s} d/dlogits max rel error: {err:.2e}  {'OK' if err < 1e-4 else 'FAIL'}")

    print("\n=== end-to-end YeastCNN gradient check (tiny 12x12 input) ===")
    model = YeastCNN(input_size=12, seed=3)
    x = rng.normal(size=(4, 1, 12, 12))
    y = rng.integers(0, 2, size=4).astype(np.float64)

    def full_loss():
        logits = model.forward(x)
        loss, _ = bce_with_logits(logits, y)
        return loss

    loss, dlogits = bce_with_logits(model.forward(x), y)
    model.backward(dlogits)

    for lname, layer in [("conv1", model.conv1), ("conv2", model.conv2),
                          ("fc1", model.fc1), ("fc2", model.fc2)]:
        for pname in ("W", "b"):
            param = getattr(layer, pname)
            analytic = getattr(layer, f"d{pname}")
            # subsample large param arrays to keep this fast
            flat_idx = rng.choice(param.size, size=min(8, param.size), replace=False)
            numeric_flat = np.zeros(len(flat_idx))
            flat_param = param.reshape(-1)
            for j, idx in enumerate(flat_idx):
                orig = flat_param[idx]
                flat_param[idx] = orig + EPS
                lp = full_loss()
                flat_param[idx] = orig - EPS
                lm = full_loss()
                flat_param[idx] = orig
                numeric_flat[j] = (lp - lm) / (2 * EPS)
            analytic_flat = analytic.reshape(-1)[flat_idx]
            err = rel_error(analytic_flat, numeric_flat).max()
            status = "OK" if err < 1e-3 else "FAIL"
            print(f"{lname}.{pname:12s} (sampled) max rel error: {err:.2e}  {status}")


def check_hybrid():
    print("\n=== end-to-end HybridYeastCNN gradient check (tiny 12x12 input + aux) ===")
    model = HybridYeastCNN(input_size=12, aux_dim=7, seed=3)
    x = rng.normal(size=(4, 1, 12, 12))
    aux = rng.normal(size=(4, 7))
    y = rng.integers(0, 2, size=4).astype(np.float64)

    def full_loss():
        logits = model.forward(x, aux)
        loss, _ = bce_with_logits(logits, y)
        return loss

    loss, dlogits = bce_with_logits(model.forward(x, aux), y)
    model.backward(dlogits)

    for lname, layer in [("conv1", model.conv1), ("conv2", model.conv2),
                          ("fc1", model.fc1), ("fc2", model.fc2)]:
        for pname in ("W", "b"):
            param = getattr(layer, pname)
            analytic = getattr(layer, f"d{pname}")
            flat_idx = rng.choice(param.size, size=min(8, param.size), replace=False)
            numeric_flat = np.zeros(len(flat_idx))
            flat_param = param.reshape(-1)
            for j, idx in enumerate(flat_idx):
                orig = flat_param[idx]
                flat_param[idx] = orig + EPS
                lp = full_loss()
                flat_param[idx] = orig - EPS
                lm = full_loss()
                flat_param[idx] = orig
                numeric_flat[j] = (lp - lm) / (2 * EPS)
            analytic_flat = analytic.reshape(-1)[flat_idx]
            err = rel_error(analytic_flat, numeric_flat).max()
            status = "OK" if err < 1e-3 else "FAIL"
            print(f"{lname}.{pname:12s} (sampled) max rel error: {err:.2e}  {status}")

    # also check gradient flows correctly INTO fc1's aux-feature columns
    # (i.e. fc1.dW's aux rows are nonzero and match a direct perturbation of `aux`)
    aux_grad_numeric = np.zeros_like(aux)
    for i in range(aux.shape[0]):
        for j in range(aux.shape[1]):
            a2 = aux.copy(); a2[i, j] += EPS
            lp = bce_with_logits(model.forward(x, a2), y)[0]
            a2 = aux.copy(); a2[i, j] -= EPS
            lm = bce_with_logits(model.forward(x, a2), y)[0]
            aux_grad_numeric[i, j] = (lp - lm) / (2 * EPS)
    # analytic: dL/daux = d(fc1 input)[:, flat_dim:] -- recompute via fc1.dW structure
    # (fc1.dW = x.T @ dout, so instead just check via finite differences on x too, already
    # covered by W checks above; here we confirm aux actually influences the loss)
    print(f"aux-feature finite-diff gradient nonzero: {np.abs(aux_grad_numeric).max():.2e} "
          f"(sanity: aux inputs DO affect the loss)")


if __name__ == "__main__":
    main()
    check_hybrid()
