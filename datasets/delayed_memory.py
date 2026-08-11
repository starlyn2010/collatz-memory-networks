"""Delayed Memory task (Hasani et al., 2021 style, sin descargas).

Secuencia: x = [a, PAD×D, b, PAD×D]  (largo 2D+2).
Target: a+b (escalar), evaluado en la ÚLTIMA posición.
a, b ∈ {1..8}. Vocab: 0=PAD, 1..8 símbolos.
Métrica: MSE final (y MAE).
"""

import numpy as np

PAD = 0
SYMBOLS = 8
VOCAB = SYMBOLS + 1
VMAX = 2 * SYMBOLS


def make_sample(D, rng):
    a = int(rng.integers(1, SYMBOLS + 1))
    b = int(rng.integers(1, SYMBOLS + 1))
    x = np.concatenate([[a], np.zeros(D, dtype=np.int64), [b], np.zeros(D, dtype=np.int64)])
    return x, a + b


class DelayedMemoryDataset:
    def __init__(self, D, n_examples=2048, base_seed=0):
        self.D = D
        self.n_examples = n_examples
        self.base_seed = base_seed
        self.seq_len = 2 * D + 2

    def batch(self, bs, seed):
        rng = np.random.default_rng(seed)
        xs, ys = [], []
        for _ in range(self.n_examples // bs):
            xb, yb = [], []
            for _ in range(bs):
                x, y = make_sample(self.D, rng)
                xb.append(x)
                yb.append(y)
            xs.append(np.stack(xb))
            ys.append(np.asarray(yb, dtype="float32"))
        return np.concatenate(xs), np.concatenate(ys)


def eval_delayed(model, x, y, bs=64, device="cpu"):
    """MSE y MAE en la predicción final de a+b (última posición)."""
    import torch

    model.eval()
    n = len(x)
    sse = 0.0
    sae = 0.0
    with torch.no_grad():
        for i in range(0, n, bs):
            xb = torch.from_numpy(x[i : i + bs]).to(device)
            yb = torch.from_numpy(y[i : i + bs]).to(device)
            logits = model(xb)
            pred = logits[:, -1, 0]
            sse += float(((pred - yb) ** 2).sum())
            sae += float((pred - yb).abs().sum())
    return {"mse": sse / n, "mae": sae / n}