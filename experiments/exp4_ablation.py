"""Paso 6 — Ablación sobre Copy Task L=100 (7 modos × 5 semillas).

Modos:
  1. CollatzFix1 — CGMN completa, modo índice (valuations + W_m entrenable).
  2. NoCollatz   — m_t ≡ 1 (GRU pura).
  3. RandomGate  — m_t ~ Bernoulli(0.5) i.i.d. (reproducible por batch).
  4. SobolGate   — m_t = 1[Sobol < 0.5] (scipy qmc, 1-D).
  5. CollatzFix1 réplica — idéntico al modo 1 (chequeo de validez).
  6. CollatzFix2 — modo índice + ~1% perturbación Cranley-Patterson
                   (m_t = sigmoid(v/3), no entrenable).
  7. CollatzFix3 — entropy scheduling coseno α: 0→0.5, órbita única
                   (m_t = sigmoid(v/3), no entrenable).

Config documentada: L=100 (T=201), bs=64, 2048 muestras/epoch, 15 epochs,
AdamW lr=1e-3 coseno a 1e-5, wd=1e-4, clip 1.0.
--partition 0|1 divide los 35 runs en 2 workers (torch 1 thread c/u).
Salida: outputs/exp4_ablation.json + fig5_ablation.png (al completar).

Uso: python3 experiments/exp4_ablation.py [--partition 0] [--epochs 15]
"""

import argparse
import json
import math
import os
import resource
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.copy_task import VOCAB, CopyTaskDataset  # noqa: E402
from models.cgmn import CGMN  # noqa: E402
from models.gates import (  # noqa: E402
    mask_collatz_fix2,
    mask_collatz_fix3,
    mask_random,
    mask_sobol,
)
from experiments.train_common import make_valuations, import_eval  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
RESULTS = os.path.join(OUT, "exp4_ablation.json")
LOG = os.path.join(OUT, "exp4_log.txt")
L = 100
SEQ_LEN = 2 * L + 1
BS = 64
SEED0 = 1234
N_SEEDS = 5
MODES = list(range(1, 8))
MODE_NAMES = {
    1: "CollatzFix1",
    2: "NoCollatz",
    3: "RandomGate",
    4: "SobolGate",
    5: "CollatzFix1-replica",
    6: "CollatzFix2-CP1pct",
    7: "CollatzFix3-entropy",
}


def _log(s):
    print(s, flush=True)
    with open(LOG, "a") as f:
        f.write(s + "\n")
        f.flush()


def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            return json.load(f)
    return {}


def save_results(res):
    tmp = RESULTS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=2)
    os.replace(tmp, RESULTS)


def build_gate_factory(mode, seed):
    """Devuelve fn(xb) -> gate_override o None (modos 1/2/5)."""
    if mode in (1, 5):
        return None, None
    if mode == 2:
        return None, None
    run_seed = SEED0 + seed
    if mode == 3:
        def f(xb, batch_idx, _rs=run_seed):
            B = xb.shape[0]
            return mask_random(SEQ_LEN, B, 64, _rs * 10000 + batch_idx)
        return f, "random"
    if mode == 4:
        def f(xb, batch_idx, _rs=run_seed):
            B = xb.shape[0]
            return mask_sobol(SEQ_LEN, B, 64, seed=_rs * 10000 + batch_idx)
        return f, "sobol"
    if mode == 6:
        vals6 = mask_collatz_fix2(SEQ_LEN, 1, 1, base_seed=42, c=1, K=50, frac=0.01)
        return (lambda xb, batch_idx, _v=vals6: _v), "fix2"
    if mode == 7:
        vals7 = mask_collatz_fix3(SEQ_LEN, 1, 1, c=1, n0=None)
        return (lambda xb, batch_idx, _v=vals7: _v), "fix3"
    raise ValueError(mode)


def train_one(model, mode, seed, gate_factory, vals, eval_x, eval_y, eval_m, epochs):
    torch.manual_seed(SEED0 + seed)
    np.random.seed(SEED0 + seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ds = CopyTaskDataset(L, n_examples=2048, base_seed=SEED0)
    total_steps = None
    step = 0
    t0 = time.time()
    ep_times = []
    ram = 0.0
    best = None
    for ep in range(epochs):
        model.train()
        t = time.time()
        x, y, m = ds.batch(BS, seed=1000 + ep)
        x, y, m = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(m)
        if total_steps is None:
            total_steps = epochs * max(1, len(x) // BS)
        for i in range(0, len(x), BS):
            xb, yb, mb = x[i : i + BS], y[i : i + BS], m[i : i + BS]
            opt.zero_grad(set_to_none=True)
            batch_idx = i // BS
            if gate_factory is not None:
                go = gate_factory(xb, batch_idx)
                logits = model(xb, gate_override=go)
            elif vals is not None:
                logits = model(xb, valuations=vals())
            else:
                logits = model(xb)
            ce = -(torch.log_softmax(logits, -1).gather(-1, yb.unsqueeze(-1)).squeeze(-1) * mb).sum() / mb.sum().clamp(min=1)
            ce.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for g in opt.param_groups:
                g["lr"] = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * step / max(total_steps - 1, 1)))
            step += 1
        ep_times.append(time.time() - t)
        ram = max(ram, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
        if ep % 5 == 0 or ep == epochs - 1:
            ev = import_eval(model, eval_x, eval_y, eval_m, bs=64)
            if best is None or ev["token_acc"] > best["token_acc"]:
                best = dict(ev, epoch=ep)
            _log(f"[{MODE_NAMES[mode]} seed={seed}] ep {ep+1}/{epochs} token_acc={ev['token_acc']:.4f} ce={ce.item():.3f}")
    ev = import_eval(model, eval_x, eval_y, eval_m, bs=64)
    if best is None or ev["token_acc"] > best["token_acc"]:
        best = dict(ev, epoch=epochs - 1)
    return {
        "final": ev,
        "best": best,
        "sec_per_epoch": float(np.mean(ep_times)),
        "total_sec": time.time() - t0,
        "ram_peak_mb": ram,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--L", type=int, default=100)
    args = ap.parse_args()
    torch.set_num_threads(1)
    global L, SEQ_LEN
    L = args.L
    SEQ_LEN = 2 * L + 1

    eval_x, eval_y, eval_m = CopyTaskDataset(L, n_examples=512, base_seed=999).batch(64, seed=777)
    vals = make_valuations(SEQ_LEN)

    runs = [(mode, s) for mode in MODES for s in range(N_SEEDS)]
    if args.partition == 0:
        runs = runs[::2]
    else:
        runs = runs[1::2]

    results = load_results()
    for mode, seed in runs:
        key = str(mode)
        if key in results and str(seed) in results[key]:
            _log(f"modo {MODE_NAMES[mode]} seed={seed}: ya en JSON")
            continue
        _log(f"=== {MODE_NAMES[mode]} seed={seed} ===")
        torch.manual_seed(SEED0 + seed)
        model = CGMN(VOCAB, max_seq_len=SEQ_LEN + 8)
        gate_factory, kind = build_gate_factory(mode, seed)
        v = vals if mode in (1, 5) else None
        stats = train_one(model, mode, seed, gate_factory, (lambda _v=v: _v) if v is not None else None,
                          eval_x, eval_y, eval_m, args.epochs)
        results = load_results()  # re-leer: el otro worker pudo guardar mientras tanto
        results.setdefault(key, {})[str(seed)] = stats
        save_results(results)
        _log(f"{MODE_NAMES[mode]} seed={seed}: token_acc={stats['final']['token_acc']:.4f} exact={stats['final']['exact_acc']:.4f}\n")

    if all(str(m) in results and len(results[str(m)]) == N_SEEDS for m in MODES):
        report(results, args.epochs, L)


def report(results, epochs, L):
    _log("\n=========== TABLA ABLACIÓN (Copy Task L=%d, %d epochs) ===========" % (L, epochs))
    rows = {}
    for mode in MODES:
        accs = [results[str(mode)][str(s)]["final"]["token_acc"] for s in range(N_SEEDS)]
        exas = [results[str(mode)][str(s)]["final"]["exact_acc"] for s in range(N_SEEDS)]
        rows[mode] = (np.mean(accs), np.std(accs), np.mean(exas), np.std(exas))
        _log(f"{MODE_NAMES[mode]:22s} token_acc={np.mean(accs):.4f}±{np.std(accs):.4f}  exact_acc={np.mean(exas):.4f}±{np.std(exas):.4f}")
    make_figure(rows, L)


def make_figure(rows, L):
    names = [MODE_NAMES[m] for m in MODES]
    means = [rows[m][0] for m in MODES]
    stds = [rows[m][1] for m in MODES]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(names)), means, yerr=stds, capsize=4, color="#4C72B0")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel(f"token accuracy (copia L={L})")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Ablación CGMN — Copy Task L={L} (media±std, 5 semillas)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fname = "fig5_ablation.png" if L == 10 else f"fig5_ablation_L{L}.png"
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)
    _log(f"{fname} guardada")


if __name__ == "__main__":
    main()