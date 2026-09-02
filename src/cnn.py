"""
A CNN implemented from scratch in NumPy: every forward AND backward pass
(convolution, max-pool, dense, ReLU, sigmoid+BCE) is derived and coded by
hand here -- no autograd, no TensorFlow/PyTorch. Chosen because Cellpose/
TF/PyTorch couldn't be installed in this environment (no PyPI access), and
because the point of this piece of the portfolio project is to show the
mechanics, not call a library.

Architecture (see docs/CNN_DESIGN.md for the full reasoning):
    Conv(8, 5x5) -> ReLU -> MaxPool(2x2)
    Conv(16, 3x3) -> ReLU -> MaxPool(2x2)
    Flatten -> Dense(32) -> ReLU -> Dense(1) -> sigmoid (BCE loss)

Conv uses an im2col/col2im formulation (stride-tricks, no python loop over
pixels) so it's fast enough in pure NumPy for a dataset this size.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- layers --

class Conv2D:
    """Valid (no padding), stride-1 2D convolution over (N, C, H, W) input."""

    def __init__(self, in_ch: int, out_ch: int, k: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        # He init: right scale for ReLU-following layers.
        scale = np.sqrt(2.0 / (in_ch * k * k))
        self.W = rng.normal(0, scale, size=(out_ch, in_ch, k, k))
        self.b = np.zeros(out_ch)
        self.k, self.in_ch, self.out_ch = k, in_ch, out_ch
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)
        self.dW = self.db = None

    def _im2col(self, x: np.ndarray):
        """(N,C,H,W) -> cols (N, P, K) where P=out_h*out_w, K=C*k*k, via a
        zero-copy sliding-window view (no python loop over pixels)."""
        N, C, H, W = x.shape
        k = self.k
        out_h, out_w = H - k + 1, W - k + 1
        shape = (N, C, out_h, out_w, k, k)
        strides = (x.strides[0], x.strides[1], x.strides[2], x.strides[3], x.strides[2], x.strides[3])
        patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        cols = patches.transpose(0, 2, 3, 1, 4, 5).reshape(N, out_h * out_w, C * k * k)
        return cols, out_h, out_w

    def _col2im(self, dcols: np.ndarray, x_shape) -> np.ndarray:
        """Reverse of _im2col: scatter-add gradient patches back onto the
        input, accumulating at every pixel touched by more than one patch."""
        N, C, H, W = x_shape
        k = self.k
        out_h, out_w = self.out_h, self.out_w
        dcols = dcols.reshape(N, out_h, out_w, C, k, k)
        dx = np.zeros((N, C, H, W))
        for i in range(k):
            for j in range(k):
                dx[:, :, i:i + out_h, j:j + out_w] += dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        return dx

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        cols, out_h, out_w = self._im2col(x)
        self.cols, self.out_h, self.out_w = cols, out_h, out_w
        W_flat = self.W.reshape(self.out_ch, -1)          # (O, K)
        out = cols @ W_flat.T + self.b                     # (N, P, O)
        return out.transpose(0, 2, 1).reshape(x.shape[0], self.out_ch, out_h, out_w)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        N = dout.shape[0]
        dout_flat = dout.reshape(N, self.out_ch, -1).transpose(0, 2, 1)  # (N, P, O)
        W_flat = self.W.reshape(self.out_ch, -1)                          # (O, K)

        dW_flat = np.einsum('npk,npo->ko', self.cols, dout_flat)          # (K, O)
        self.dW = dW_flat.T.reshape(self.W.shape)
        self.db = dout_flat.sum(axis=(0, 1))

        dcols = dout_flat @ W_flat                                        # (N, P, K)
        return self._col2im(dcols, self.x.shape)


class MaxPool2D:
    """Non-overlapping max pooling (size == stride)."""

    def __init__(self, size: int = 2):
        self.size = size

    def forward(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        s = self.size
        out_h, out_w = H // s, W // s
        x = x[:, :, :out_h * s, :out_w * s]
        shape = (N, C, out_h, out_w, s, s)
        strides = (x.strides[0], x.strides[1], x.strides[2] * s, x.strides[3] * s, x.strides[2], x.strides[3])
        patches = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        self.x_shape = x.shape
        flat = patches.reshape(N, C, out_h, out_w, s * s)
        self.argmax = flat.argmax(axis=-1)
        return flat.max(axis=-1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        N, C, out_h, out_w = dout.shape
        s = self.size
        dx = np.zeros(self.x_shape)
        n_idx, c_idx = np.meshgrid(np.arange(N), np.arange(C), indexing='ij')
        for i in range(out_h):
            for j in range(out_w):
                idx = self.argmax[:, :, i, j]
                di, dj = idx // s, idx % s
                dx[n_idx, c_idx, i * s + di, j * s + dj] += dout[:, :, i, j]
        return dx


class Flatten:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout.reshape(self.shape)


class Dense:
    def __init__(self, in_dim: int, out_dim: int, seed: int | None = None):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0, scale, size=(in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)
        self.dW = self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.dW = self.x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.W.T


class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask


# ------------------------------------------------------------- loss/opt --

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def bce_with_logits(logits: np.ndarray, y: np.ndarray):
    """Numerically-stable binary cross-entropy computed directly on the
    pre-sigmoid logits (avoids log(0) from a saturated sigmoid).
    Returns (mean_loss, dL/dlogits)."""
    logits = logits.reshape(-1)
    y = y.reshape(-1)
    max_val = np.maximum(logits, 0)
    loss = max_val - logits * y + np.log1p(np.exp(-np.abs(logits)))
    grad = (sigmoid(logits) - y) / len(y)
    return loss.mean(), grad.reshape(-1, 1)


class Adam:
    """Adam optimizer, applied uniformly across every layer that exposes
    (W, dW, mW, vW) and (b, db, mb, vb)."""

    def __init__(self, layers, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.layers = layers
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t = 0

    def step(self):
        self.t += 1
        for layer in self.layers:
            if not hasattr(layer, "W"):
                continue
            for name in ("W", "b"):
                p = getattr(layer, name)
                g = getattr(layer, f"d{name}")
                m = getattr(layer, f"m{name}")
                v = getattr(layer, f"v{name}")
                m[:] = self.beta1 * m + (1 - self.beta1) * g
                v[:] = self.beta2 * v + (1 - self.beta2) * g ** 2
                mhat = m / (1 - self.beta1 ** self.t)
                vhat = v / (1 - self.beta2 ** self.t)
                p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ------------------------------------------------------------------ model --

class YeastCNN:
    """Conv(8,5x5)-ReLU-Pool -> Conv(16,3x3)-ReLU-Pool -> Dense(32)-ReLU -> Dense(1)."""

    def __init__(self, input_size: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 1_000_000, size=4)
        self.conv1 = Conv2D(1, 8, 5, seed=int(seeds[0]))
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2)
        self.conv2 = Conv2D(8, 16, 3, seed=int(seeds[1]))
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(2)
        self.flatten = Flatten()

        # infer flattened size by running a dummy forward pass
        dummy = np.zeros((1, 1, input_size, input_size))
        flat_dim = self._features_forward(dummy).shape[1]

        self.fc1 = Dense(flat_dim, 32, seed=int(seeds[2]))
        self.relu3 = ReLU()
        self.fc2 = Dense(32, 1, seed=int(seeds[3]))

        self.layers = [self.conv1, self.conv2, self.fc1, self.fc2]

    def _features_forward(self, x):
        x = self.pool1.forward(self.relu1.forward(self.conv1.forward(x)))
        x = self.pool2.forward(self.relu2.forward(self.conv2.forward(x)))
        return self.flatten.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, 1, H, W). Returns logits (N, 1) -- sigmoid not applied,
        bce_with_logits expects raw logits."""
        feats = self._features_forward(x)
        h = self.relu3.forward(self.fc1.forward(feats))
        return self.fc2.forward(h)

    def backward(self, dlogits: np.ndarray) -> None:
        d = self.fc2.backward(dlogits)
        d = self.fc1.backward(self.relu3.backward(d))
        d = self.flatten.backward(d)
        d = self.conv2.backward(self.relu2.backward(self.pool2.backward(d)))
        self.conv1.backward(self.relu1.backward(self.pool1.backward(d)))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(self.forward(x)).reshape(-1)


class HybridYeastCNN(YeastCNN):
    """Same conv trunk as YeastCNN, but fc1 also takes a small vector of
    hand-engineered shape features (from shape_features.py: area,
    eccentricity, solidity, extent, aspect_ratio, n_defects,
    max_defect_depth_ratio), concatenated onto the flattened conv output.

    Why: the plain pixel-only CNN's errors clustered on exactly the two
    things these features were built to capture -- elongated/uneven single
    cells read as budding (no direct access to boundary concavity), and
    faint small buds missed (a small geometric defect is precise in
    shape-feature space even when it's a handful of ambiguous pixels).
    This doesn't replace the conv path -- it gives the network both signals
    and lets training decide how to weigh them.
    """

    def __init__(self, input_size: int = 48, aux_dim: int = 7, seed: int = 0):
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 1_000_000, size=4)
        self.conv1 = Conv2D(1, 8, 5, seed=int(seeds[0]))
        self.relu1 = ReLU()
        self.pool1 = MaxPool2D(2)
        self.conv2 = Conv2D(8, 16, 3, seed=int(seeds[1]))
        self.relu2 = ReLU()
        self.pool2 = MaxPool2D(2)
        self.flatten = Flatten()

        dummy = np.zeros((1, 1, input_size, input_size))
        flat_dim = self._features_forward(dummy).shape[1]
        self._flat_dim = flat_dim
        self.aux_dim = aux_dim

        self.fc1 = Dense(flat_dim + aux_dim, 32, seed=int(seeds[2]))
        self.relu3 = ReLU()
        self.fc2 = Dense(32, 1, seed=int(seeds[3]))

        self.layers = [self.conv1, self.conv2, self.fc1, self.fc2]

    def forward(self, x: np.ndarray, aux: np.ndarray | None = None) -> np.ndarray:
        """x: (N,1,H,W). aux: (N, aux_dim), already standardized."""
        feats = self._features_forward(x)
        if aux is None:
            aux = np.zeros((x.shape[0], self.aux_dim))
        combined = np.concatenate([feats, aux], axis=1)
        h = self.relu3.forward(self.fc1.forward(combined))
        return self.fc2.forward(h)

    def backward(self, dlogits: np.ndarray) -> None:
        d = self.fc2.backward(dlogits)
        d = self.fc1.backward(self.relu3.backward(d))
        d_feats = d[:, :self._flat_dim]  # d w.r.t. the aux half is discarded --
                                          # aux features have no upstream layer
        d = self.flatten.backward(d_feats)
        d = self.conv2.backward(self.relu2.backward(self.pool2.backward(d)))
        self.conv1.backward(self.relu1.backward(self.pool1.backward(d)))

    def predict_proba(self, x: np.ndarray, aux: np.ndarray | None = None) -> np.ndarray:
        return sigmoid(self.forward(x, aux)).reshape(-1)
