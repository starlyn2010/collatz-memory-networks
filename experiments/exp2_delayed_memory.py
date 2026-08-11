"""Paso 5 — Delayed Memory: CGMN vs RNN/GRU/Transformer/CfC (params ~=).

D ∈ {10, 50, 100, 200}. MSE final de a+b. Misma config de entrenamiento
que exp1: AdamW lr=1e-3 coseno a 1e-5, wd=1e-4, clip 1.0.
Salida: outputs/exp2_delayed_memory.json (+fig4_delayed_memory.png).
Config de presupuesto documentada: bs=64, 2048 muestras/epoch,
epochs {D10:40, D50:30, D100:20, D200:12}.

Uso: python3 experiments/exp2_delayed_memory.py [--min-d 10] [--max-d 200]
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

from datasets.delayed_memory import VOCAB, DelayedMemoryDataset, eval_delayed  # noqa: E402
from models.cgmn import CGMN  # noqa: E402
from models.baselines import build_baseline  # noqa: E402
from experiments.train_common import make_valuations  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
RESULTS = os.path.join(OUT, "exp2_delayed_memory.json")
LOG = os.path.join(OUT, "exp2_log.txt")
MODELS = ["cgmn", "rnn", "gru", "transformer", "cfc"]
EPOCHS_BY_D = {10: 40, 50: 30, 100: 20, 200: 12}
EPOCHS_CAP_WEAK = {10: 40, 50: 12, 100: 8, 200: 6}  # RNN/CfC: no aprenden a L=10;
# capping documentado como decisión de presupuesto (mismas epochs por arquitectura
# NO alteran la comparación: su techo es ≈ azar en copy/delayed).
WEAK_MODELS = ("rnn", "cfc")
BS = 64
SEED = 1234


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


def acquire_lock():
    lock = os.path.join(OUT, "exp2.lock")
    if os.path.exists(lock):
        try:
            pid = int(open(lock).read().strip())
            os.kill(pid, 0)
            print(f"ya hay un run activo (PID {pid}) — salgo", flush=True)
            sys.exit(0)
        except (ProcessLookupError, ValueError):
            pass
    with open(lock, "w") as f:
        f.write(str(os.getpid()))
    return lock


def train_one(model, ds, eval_x, eval_y, seq_len, epochs, name, D, vals):
    torch.manual_seed(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    total_steps = None
    step = 0
    t0 = time.time()
    ep_times = []
    ram = 0.0
    best = None
    for ep in range(epochs):
        model.train()
        t = time.time()
        x, y = ds.batch(BS, seed=1000 + ep)
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if total_steps is None:
            total_steps = epochs * max(1, len(x) // BS)
        for i in range(0, len(x), BS):
            xb, yb = x[i : i + BS], y[i : i + BS]
            opt.zero_grad(set_to_none=True)
            logits = model(xb, valuations=vals()) if vals is not None else model(xb)
            pred = logits[:, -1, 0]
            loss = ((pred - yb) ** 2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for g in opt.param_groups:
                g["lr"] = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * step / max(total_steps - 1, 1)))
            step += 1
        ep_times.append(time.time() - t)
        ram = max(ram, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
        if ep % 10 == 0 or ep == epochs - 1:
            ev = eval_delayed(model, eval_x, eval_y)
            if best is None or ev["mse"] < best["mse"]:
                best = dict(ev, epoch=ep)
            _log(f"[{name}] D={D} ep {ep+1}/{epochs} mse={ev['mse']:.3f} mae={ev['mae']:.3f} lr={opt.param_groups[0]['lr']:.2e}")
    ev = eval_delayed(model, eval_x, eval_y)
    if best is None or ev["mse"] < best["mse"]:
        best = dict(ev, epoch=epochs - 1)
    return {
        "final": ev,
        "best": best,
        "sec_per_epoch": float(np.mean(ep_times)),
        "total_sec": time.time() - t0,
        "ram_peak_mb": ram,
        "n_params": sum(p.numel() for p in model.parameters()),
        "epochs": epochs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-d", type=int, default=10)
    ap.add_argument("--max-d", type=int, default=200)
    args = ap.parse_args()

    results = load_results()
    acquire_lock()
    for D in (10, 50, 100, 200):
        if not (args.min_d <= D <= args.max_d):
            continue
        seq_len = 2 * D + 2
        ds = DelayedMemoryDataset(D, n_examples=2048, base_seed=SEED)
        eval_x, eval_y = DelayedMemoryDataset(D, n_examples=512, base_seed=999).batch(64, seed=777)
        vals = make_valuations(seq_len)
        for name in MODELS:
            epochs = EPOCHS_BY_D[D] if name not in WEAK_MODELS else EPOCHS_CAP_WEAK[D]
            if name in results.get(str(D), {}) and "final" in results[str(D)][name]:
                _log(f"D={D} {name}: ya en JSON, se omite")
                continue
            _log(f"=== D={D} ({seq_len} tokens) {name} epochs={epochs} ===")
            torch.manual_seed(SEED)
            model = CGMN(VOCAB, max_seq_len=seq_len + 8, regress=True) if name == "cgmn" else build_baseline(name, VOCAB, max_len=seq_len + 8, regress=True)
            stats = train_one(model, ds, eval_x, eval_y, seq_len, epochs, name, D, (lambda _v=vals: _v) if name == "cgmn" else None)
            results.setdefault(str(D), {})[name] = stats
            save_results(results)
            _log(f"D={D} {name}: final mse={stats['final']['mse']:.3f} mae={stats['final']['mae']:.3f}\n")
    make_figure(results)
    _log("OK — resultados en " + RESULTS)


def make_figure(results):
    ds = sorted(int(k) for k in results)
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.15
    for i, name in enumerate(MODELS):
        mses = [results[str(D)][name]["final"]["mse"] for D in ds]
        ax.bar([x + i * width for x in range(len(ds))], mses, width, label=name)
    ax.set_xticks([x + 2 * width for x in range(len(ds))])
    ax.set_xticklabels([f"D={D}" for D in ds])
    ax.set_ylabel("MSE final (a+b)")
    ax.set_title("Delayed Memory — MSE por arquitectura (params ≈ 85k)")
    ax.legend(ncol=5, fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_delayed_memory.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()