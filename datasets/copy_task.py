"""Copy Task dataset (sintético, sin descargas).

Vocab: 0=PAD, 1=SEP, 2..9 = 8 símbolos de datos.
Secuencia: x = [a_1..a_L, SEP, a_1..a_L] (largo 2L+1).
Target (teacher forcing): y[t] = x[t+1], pérdida solo en t ∈ [L, 2L-1]
(las L posiciones de copia).
"""

import numpy as np

PAD, SEP = 0, 1
DATA_SYMBOLS = 8
VOCAB = DATA_SYMBOLS + 2


def make_sample(L, rng):
    a = rng.integers(2, 2 + DATA_SYMBOLS, size=L)
    return np.concatenate([a, [SEP], a])


class CopyTaskDataset:
    def __init__(self, L, n_examples=2048, base_seed=0):
        self.L = L
        self.n_examples = n_examples
        self.base_seed = base_seed
        self.seq_len = 2 * L + 1

    def batch(self, bs, seed):
        rng = np.random.default_rng(seed)
        xs, ys, ms = [], [], []
        for _ in range(self.n_examples // bs):
            x = np.stack([make_sample(self.L, rng) for _ in range(bs)])
            y = np.concatenate([x[:, 1:], np.full((bs, 1), PAD, dtype=x.dtype)], axis=1)
            m = np.zeros((bs, self.seq_len), dtype="float32")
            m[:, self.L : self.seq_len - 1] = 1.0
            xs.append(x)
            ys.append(y)
            ms.append(m)
        return np.concatenate(xs), np.concatenate(ys), np.concatenate(ms)


def eval_copy_metrics(model, x, y, m, bs=64, device="cpu"):
    """Métricas sobre posiciones de copia: token_acc, exact_acc, mse."""
    import torch

    model.eval()
    n = len(x)
    correct = 0
    exact = 0
    mse_acc = 0.0
    counts = 0
    n_seq = 0
    with torch.no_grad():
        for i in range(0, n, bs):
            xb = torch.from_numpy(x[i : i + bs]).to(device)
            yb = torch.from_numpy(y[i : i + bs]).to(device)
            mb = torch.from_numpy(m[i : i + bs]).to(device)
            logits = model(xb)
            pred = logits.argmax(-1)
            hit = (pred == yb).float() * mb
            correct += int(hit.sum())
            counts += int(mb.sum())
            n_seq += xb.shape[0]
            exact += int((hit.sum(-1) == mb.sum(-1)).sum())
            onehot = torch.nn.functional.one_hot(yb, logits.size(-1)).float()
            mse_acc += float((((logits - onehot) ** 2) * mb.unsqueeze(-1)).sum())
    return {
        "token_acc": correct / max(counts, 1),
        "exact_acc": exact / max(n_seq, 1),
        "mse": mse_acc / max(counts, 1),
    }