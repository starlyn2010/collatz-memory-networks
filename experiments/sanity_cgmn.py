"""Paso 2 sanity check: forward pass de CGMN (batch=4, seq_len=20).

Verifica: sin NaNs, formas correctas, y que el modo con valuations
precomputadas y el modo m=1 (NoCollatz) corren sin error.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cgmn import CGMN  # noqa: E402
from models.collatz_generator import collatz_valuation_index_mode  # noqa: E402


def main():
    torch.manual_seed(0)
    V, H, LAYERS, B, T = 12, 64, 2, 4, 20
    model = CGMN(vocab_size=V, hidden_size=H, n_layers=LAYERS)

    vals = collatz_valuation_index_mode(T=T, c=1, base_seed=42, K=50)

    x = torch.randint(0, V, (B, T))
    logits_default = model(x)
    assert logits_default.shape == (B, T, V), logits_default.shape
    assert torch.isfinite(logits_default).all()

    logits_val = model(x, valuations=torch.tensor(vals, dtype=torch.long))
    assert torch.isfinite(logits_val).all()

    logits_none = model(x, valuations=None)
    assert torch.isfinite(logits_none).all()

    params = sum(p.numel() for p in model.parameters())
    print(f"sanity OK: logits {tuple(logits_default.shape)}, "
          f"nantest ok, CGMN params = {params:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())