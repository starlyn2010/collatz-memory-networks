"""Utilidades de entrenamiento compartidas (exp1, exp2, exp4).

- train_loop: AdamW, lr=1e-3 coseno a 1e-5, weight_decay=1e-4, clip 1.0.
- Mide: tiempo por epoch, pico de RAM (resource.getrusage), CE por epoch.
- `make_valuations(seq_len)`: valuaciones Collatz precomputadas (modo índice).
"""

import math
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from models.collatz_generator import collatz_valuation_index_mode  # noqa: E402


def make_valuations(seq_len, base_seed=42, K=50):
    return torch.tensor(collatz_valuation_index_mode(seq_len, c=1, base_seed=base_seed, K=K),
                        dtype=torch.long)


def train_loop(model, data_batch_fn, eval_fn, seq_len, epochs, bs=32,
               lr=1e-3, wd=1e-4, grad_clip=1.0, device="cpu", seed=0,
               vals_fn=None, verbose=True, eval_every=10, log=None):
    """data_batch_fn(seed) -> (x, y, mask); eval_fn() -> dict métricas."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    eta_min, eta_max = 1e-5, lr
    step = 0
    total_steps = None
    best = None
    t0 = time.time()
    times = []
    ram_peak = 0.0
    loss_hist = []
    for ep in range(epochs):
        model.train()
        ep_t0 = time.time()
        x, y, m = data_batch_fn(bs, seed=1000 + ep)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        m = torch.from_numpy(m)
        losses = []
        if total_steps is None:
            total_steps = epochs * max(1, len(x) // bs)
        for i in range(0, len(x), bs):
            xb, yb, mb = x[i : i + bs], y[i : i + bs], m[i : i + bs]
            opt.zero_grad(set_to_none=True)
            if vals_fn is not None:
                logits = model(xb, valuations=vals_fn())
            else:
                logits = model(xb)
            ce = -(torch.log_softmax(logits, -1).gather(-1, yb.unsqueeze(-1)).squeeze(-1) * mb).sum() / mb.sum().clamp(min=1)
            ce.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            for g in opt.param_groups:
                g["lr"] = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * step / max(total_steps - 1, 1)))
            step += 1
            losses.append(ce.detach().item())
        loss_hist.append(float(np.mean(losses)))
        times.append(time.time() - ep_t0)
        ram_peak = max(ram_peak, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
        if verbose and (ep % eval_every == 0 or ep == epochs - 1):
            ev = eval_fn()
            if best is None or ev.get("token_acc", 0) > best.get("token_acc", -1):
                best = dict(ev, epoch=ep)
            if log is not None:
                log(f"[{model.__class__.__name__}] ep {ep+1}/{epochs} "
                    f"ce={loss_hist[-1]:.4f} token_acc={ev['token_acc']:.4f} "
                    f"exact_acc={ev['exact_acc']:.4f} lr={opt.param_groups[0]['lr']:.2e}")
    ev = eval_fn()
    if best is None or ev.get("token_acc", 0) > best.get("token_acc", -1):
        best = dict(ev, epoch=epochs - 1)
    return {
        "final": ev,
        "best": best,
        "loss_hist": loss_hist,
        "sec_per_epoch": float(np.mean(times)),
        "total_sec": time.time() - t0,
        "ram_peak_mb": ram_peak,
    }


def make_eval_fn(model, data, bs=64, device="cpu"):
    x, y, m = data
    return lambda: import_eval(model, x, y, m, bs, device)


def import_eval(model, x, y, m, bs, device="cpu"):
    from datasets.copy_task import eval_copy_metrics
    return eval_copy_metrics(model, x, y, m, bs=bs, device=device)