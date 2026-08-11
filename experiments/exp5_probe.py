"""Exp 5 — Sondas locales decisivas A, B1, B2 y C.

A  — Forma temporal vs nivel medio de la puerta (Copy Task L=10, 40 ep).
     6 brazos × 5 semillas: {const, collatz} × {0.98, 0.90, 0.95}.
     --partition 0|1 para 2 workers (merge con reload, como exp4).
B1 — Rechazo de distractores: [a,d1,d2,b, PAD×D] -> a+b al final (D=50).
B2 — Suma de ventana deslizante: [v1..v24, SEP] -> suma de los últimos 6.
     B1/B2: CGMN vs GRU vs Transformer (regress), 3 semillas.
C  — Generalización de longitud: entrenar Copy L=10, evaluar L=50 y L=100.

Salidas: outputs/exp5_gate_shape.json + fig7, exp5_forget.json + fig8,
exp5_lengen.json + fig9. Log: outputs/exp5_log.txt.
Uso: python3 experiments/exp5_probe.py --task A [--partition 0]
     python3 experiments/exp5_probe.py --task B1
     python3 experiments/exp5_probe.py --task B2
     python3 experiments/exp5_probe.py --task C
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
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.copy_task import VOCAB as CT_VOCAB, CopyTaskDataset  # noqa: E402
from datasets.delayed_memory import eval_delayed  # noqa: E402
from experiments.train_common import import_eval, make_valuations  # noqa: E402
from models.baselines import build_baseline  # noqa: E402
from models.cgmn import CGMN  # noqa: E402
from models.gates import (  # noqa: E402
    mask_collatz_binary,
    mask_collatz_scale,
    mask_const,
)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
LOG = os.path.join(OUT, "exp5_log.txt")
SEED0 = 1234
N_SEEDS = 5
BS = 64
LR, WD, CLIP = 1e-3, 1e-4, 1.0
ARMS = {
    "const_098": lambda T, B, H: mask_const(T, B, H, 0.98),
    "collatz_098": lambda T, B, H: mask_collatz_scale(T, B, H, 0.98, 0.05),
    "const_090": lambda T, B, H: mask_const(T, B, H, 0.90),
    "collatz_090": lambda T, B, H: mask_collatz_scale(T, B, H, 0.90, 0.10),
    "const_095": lambda T, B, H: mask_const(T, B, H, 0.95),
    "collatz_bin_095": lambda T, B, H: mask_collatz_binary(T, B, H, lo=0.9, hi=1.0),
}
MODELS_BC = ["cgmn", "gru", "transformer"]


def _log(s):
    print(s, flush=True)
    with open(LOG, "a") as f:
        f.write(s + "\n")
        f.flush()


def load_results(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_results(res, path):
    """Guarda con merge: recarga el JSON del disco (puede tener resultados
    del otro worker) y combina antes de escribir (patrón exp4)."""
    merged = load_results(path)
    merged.update(res)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=2)
    os.replace(tmp, path)


def build_model(name, vocab, seq_len, regress=False):
    if name == "cgmn":
        return CGMN(vocab, max_seq_len=seq_len + 8, regress=regress)
    return build_baseline(name, vocab, max_len=seq_len + 8, regress=regress)


def make_gate_factory(mask_base):
    def f(xb, batch_idx):
        B = xb.shape[0]
        return mask_base.expand(-1, B, -1).contiguous()

    return f


def eval_copy_override(model, x, y, m, bs, gate_factory, device="cpu"):
    model.eval()
    n = len(x)
    correct = exact = n_seq = 0
    counts = 0
    mse_acc = 0.0
    with torch.no_grad():
        for i in range(0, n, bs):
            xb = torch.from_numpy(x[i : i + bs]).to(device)
            yb = torch.from_numpy(y[i : i + bs]).to(device)
            mb = torch.from_numpy(m[i : i + bs]).to(device)
            logits = model(xb, gate_override=gate_factory(xb, i // bs))
            pred = logits.argmax(-1)
            hit = (pred == yb).float() * mb
            correct += int(hit.sum())
            counts += int(mb.sum())
            n_seq += xb.shape[0]
            exact += int((hit.sum(-1) == mb.sum(-1)).sum())
            onehot = F.one_hot(yb, logits.size(-1)).float()
            mse_acc += float((((logits - onehot) ** 2) * mb.unsqueeze(-1)).sum())
    return {
        "token_acc": correct / max(counts, 1),
        "exact_acc": exact / max(n_seq, 1),
        "mse": mse_acc / max(counts, 1),
    }


def train_probe(model, ds, eval_fn, seq_len, epochs, bs, seed, name, mode,
                gate_factory=None, vals_fn=None):
    """mode: "ce" (clasificación con máscara) | "mse" (regresión final)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total_steps = None
    step = 0
    best = None
    t0 = time.time()
    ep_times = []
    ram = 0.0
    for ep in range(epochs):
        model.train()
        t = time.time()
        b = ds.batch(bs, seed=1000 + ep)
        x = torch.from_numpy(b[0])
        y = torch.from_numpy(b[1])
        m = torch.from_numpy(b[2]) if len(b) > 2 else None
        if total_steps is None:
            total_steps = epochs * max(1, len(x) // bs)
        for i in range(0, len(x), bs):
            xb, yb = x[i : i + bs], y[i : i + bs]
            mb = m[i : i + bs] if m is not None else None
            opt.zero_grad(set_to_none=True)
            go = gate_factory(xb, i // bs) if gate_factory is not None else None
            if go is not None:
                logits = model(xb, gate_override=go)
            elif vals_fn is not None:
                logits = model(xb, valuations=vals_fn())
            else:
                logits = model(xb)
            if mode == "ce":
                ce = -(torch.log_softmax(logits, -1).gather(-1, yb.unsqueeze(-1)).squeeze(-1) * mb).sum() / mb.sum().clamp(min=1)
                ce.backward()
            else:
                loss = ((logits[:, -1, 0] - yb) ** 2).mean()
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
            for g in opt.param_groups:
                g["lr"] = 1e-5 + 0.5 * (LR - 1e-5) * (1 + math.cos(math.pi * step / max(total_steps - 1, 1)))
            step += 1
        ep_times.append(time.time() - t)
        ram = max(ram, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
        if ep % 10 == 0 or ep == epochs - 1:
            ev = eval_fn()
            score = ev.get("token_acc", 0) if mode == "ce" else -ev.get("mse", 1e9)
            if best is None or score > best.get("_score", -1e9):
                best = dict(ev, epoch=ep)
                best["_score"] = score
            _log(f"[{name}] ep {ep+1}/{epochs} " + " ".join(f"{k}={v:.4f}" for k, v in ev.items() if isinstance(v, float)))
    ev = eval_fn()
    score = ev.get("token_acc", 0) if mode == "ce" else -ev.get("mse", 1e9)
    if best is None or score > best.get("_score", -1e9):
        best = dict(ev, epoch=epochs - 1)
    best.pop("_score", None)
    return {
        "final": ev,
        "best": best,
        "sec_per_epoch": float(np.mean(ep_times)),
        "total_sec": time.time() - t0,
        "ram_peak_mb": ram,
        "n_params": sum(p.numel() for p in model.parameters()),
        "epochs": epochs,
    }


class DistractorDataset:
    """B1: [a, d1, d2, b, PAD×D] -> a+b en la última posición.
    a,b,d1,d2 ∈ 1..8; VOCAB=9 (0=PAD). seq_len = D + 5."""

    def __init__(self, D, n_examples=2048, base_seed=0):
        self.D = D
        self.n_examples = n_examples
        self.base_seed = base_seed
        self.seq_len = D + 5
        self.vocab = 9

    def batch(self, bs, seed):
        rng = np.random.default_rng(seed)
        xs, ys = [], []
        for _ in range(self.n_examples // bs):
            xb, yb = [], []
            for _ in range(bs):
                a = int(rng.integers(1, 9))
                b = int(rng.integers(1, 9))
                d1 = int(rng.integers(1, 9))
                d2 = int(rng.integers(1, 9))
                xb.append(np.concatenate([[a, d1, d2, b], np.zeros(self.D, dtype=np.int64)]))
                yb.append(a + b)
            xs.append(np.stack(xb))
            ys.append(np.asarray(yb, dtype="float32"))
        return np.concatenate(xs), np.concatenate(ys)


class WindowSumDataset:
    """B2: [v1..vM, SEP] -> suma de los últimos W valores.
    v ∈ 1..4; VOCAB=5 (0=SEP). seq_len = M + 1."""

    def __init__(self, M=24, W=6, n_examples=2048, base_seed=0):
        self.M = M
        self.W = W
        self.n_examples = n_examples
        self.base_seed = base_seed
        self.seq_len = M + 1
        self.vocab = 5

    def batch(self, bs, seed):
        rng = np.random.default_rng(seed)
        xs, ys = [], []
        for _ in range(self.n_examples // bs):
            xb, yb = [], []
            for _ in range(bs):
                v = rng.integers(1, 5, size=self.M)
                xb.append(np.concatenate([v, [0]]))
                yb.append(int(v[-self.W :].sum()))
            xs.append(np.stack(xb))
            ys.append(np.asarray(yb, dtype="float32"))
        return np.concatenate(xs), np.concatenate(ys)


def report_A(results):
    _log("\n=========== TABLA A: forma temporal vs nivel medio (Copy L=10, 40 ep) ===========")
    rows = {}
    for key in sorted(results):
        arm, s = key.split("::")
        rows.setdefault(arm, []).append(results[key]["final"]["token_acc"])
    for arm in sorted(rows):
        a = np.asarray(rows[arm])
        _log(f"{arm:16s} token_acc={a.mean():.4f}±{a.std():.4f}")
    fig, ax = plt.subplots(figsize=(9, 5))
    names = sorted(rows)
    means = [np.mean(rows[n]) for n in names]
    stds = [np.std(rows[n]) for n in names]
    colors = ["#4C72B0" if "const" in n else "#DD8452" for n in names]
    ax.bar(range(len(names)), means, yerr=stds, capsize=4, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("token accuracy (Copy L=10)")
    ax.set_ylim(0, 1.02)
    ax.set_title("A: ¿importa la forma temporal de la puerta o solo su promedio? (media±std, 5 semillas)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_gate_shape.png"), dpi=150)
    plt.close(fig)
    _log("fig7_gate_shape.png guardada")


def task_A(partition, smoke):
    epochs = 2 if smoke else 40
    n_seeds = 1 if smoke else N_SEEDS
    T = 21
    H = 64
    path = os.path.join(OUT, "exp5_gate_shape.json")
    results = load_results(path)
    ds = CopyTaskDataset(10, n_examples=256 if smoke else 2048, base_seed=SEED0)
    eval_x, eval_y, eval_m = CopyTaskDataset(10, n_examples=128 if smoke else 512, base_seed=999).batch(64, seed=777)
    for idx, arm in enumerate(sorted(ARMS)):
        if idx % 2 != partition:
            continue
        base = ARMS[arm](T, 1, H)
        gf = make_gate_factory(base)
        for s in range(n_seeds):
            key = f"{arm}::{s}"
            if key in results:
                _log(f"A {arm} seed={s}: ya en JSON")
                continue
            _log(f"=== A {arm} seed={s} ===")
            torch.manual_seed(SEED0 + s)
            model = CGMN(CT_VOCAB, hidden_size=H, n_layers=2, max_seq_len=T + 8)
            eval_fn = lambda _gf=gf: eval_copy_override(model, eval_x, eval_y, eval_m, BS, _gf)
            stats = train_probe(model, ds, eval_fn, T, epochs, BS, SEED0 + s, arm, "ce", gate_factory=gf)
            results[key] = stats
            save_results(results, path)
            _log(f"A {arm} seed={s}: token_acc={stats['final']['token_acc']:.4f}\n")
    report_A(results)


def task_B1(smoke):
    D = 50
    epochs = 2 if smoke else 30
    n_seeds = 1 if smoke else 3
    path = os.path.join(OUT, "exp5_forget.json")
    results = load_results(path)
    ds = DistractorDataset(D, n_examples=256 if smoke else 2048, base_seed=SEED0)
    eval_x, eval_y = DistractorDataset(D, n_examples=128 if smoke else 512, base_seed=999).batch(64, seed=777)
    for name in MODELS_BC:
        for s in range(n_seeds):
            key = f"dist{D}::{name}::{s}"
            if key in results:
                _log(f"B1 {name} seed={s}: ya en JSON")
                continue
            seq_len = ds.seq_len
            _log(f"=== B1 D={D} {name} seed={s} ===")
            torch.manual_seed(SEED0 + s)
            model = build_model(name, ds.vocab, seq_len, regress=True)
            vals = make_valuations(seq_len) if name == "cgmn" else None
            vfn = (lambda _v=vals: _v) if vals is not None else None
            stats = train_probe(model, ds, lambda: eval_delayed(model, eval_x, eval_y), seq_len,
                                epochs, BS, SEED0 + s, name, "mse", vals_fn=vfn)
            results[key] = stats
            save_results(results, path)
            _log(f"B1 {name} seed={s}: mse={stats['final']['mse']:.3f}\n")
    report_BC(results, "fig8_forget.png", "B: olvido selectivo — MSE final (media±std, 3 semillas)")


def task_B2(smoke):
    epochs = 2 if smoke else 40
    n_seeds = 1 if smoke else 3
    path = os.path.join(OUT, "exp5_forget.json")
    results = load_results(path)
    ds = WindowSumDataset(M=24, W=6, n_examples=256 if smoke else 2048, base_seed=SEED0)
    eval_x, eval_y = WindowSumDataset(M=24, W=6, n_examples=128 if smoke else 512, base_seed=999).batch(64, seed=777)
    for name in MODELS_BC:
        for s in range(n_seeds):
            key = f"win24::{name}::{s}"
            if key in results:
                _log(f"B2 {name} seed={s}: ya en JSON")
                continue
            seq_len = ds.seq_len
            _log(f"=== B2 W=6 {name} seed={s} ===")
            torch.manual_seed(SEED0 + s)
            model = build_model(name, ds.vocab, seq_len, regress=True)
            vals = make_valuations(seq_len) if name == "cgmn" else None
            vfn = (lambda _v=vals: _v) if vals is not None else None
            stats = train_probe(model, ds, lambda: eval_delayed(model, eval_x, eval_y), seq_len,
                                epochs, BS, SEED0 + s, name, "mse", vals_fn=vfn)
            results[key] = stats
            save_results(results, path)
            _log(f"B2 {name} seed={s}: mse={stats['final']['mse']:.3f}\n")
    report_BC(results, "fig8_forget.png", "B: olvido selectivo — MSE final (media±std, 3 semillas)")


def report_BC(results, fname, title):
    _log("\n=========== TABLA B ===========")
    groups = {}
    for key in sorted(results):
        prefix, name, s = key.split("::")
        groups.setdefault(prefix, {}).setdefault(name, []).append(results[key]["final"]["mse"])
    fig, ax = plt.subplots(figsize=(9, 5))
    x = 0
    xticks, xlabels = [], []
    for prefix in sorted(groups):
        g = groups[prefix]
        for name in sorted(g):
            a = np.asarray(g[name])
            mean, std = a.mean(), a.std()
            ax.bar(x, mean, 0.6, yerr=std, capsize=4)
            _log(f"{prefix:8s} {name:12s} mse={mean:.3f}±{std:.3f}")
            xticks.append(x)
            xlabels.append(f"{prefix}\n{name}")
            x += 1
        x += 0.8
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("MSE final")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)
    _log(f"{fname} guardada")


def task_C(smoke):
    epochs = 2 if smoke else 40
    n_seeds = 1 if smoke else 3
    path = os.path.join(OUT, "exp5_lengen.json")
    results = load_results(path)
    max_len = 2 * 100 + 1 + 8
    for name in MODELS_BC:
        for s in range(n_seeds):
            key = f"{name}::{s}"
            if key in results:
                _log(f"C {name} seed={s}: ya en JSON")
                continue
            _log(f"=== C {name} seed={s} (train L=10, eval L=50/100) ===")
            torch.manual_seed(SEED0 + s)
            if name == "cgmn":
                model = CGMN(CT_VOCAB, max_seq_len=max_len)
            elif name == "gru":
                model = build_baseline("gru", CT_VOCAB, max_len=max_len)
            else:
                model = build_baseline("transformer", CT_VOCAB, max_len=max_len)
            ds = CopyTaskDataset(10, n_examples=2048, base_seed=SEED0)
            eval10_x, eval10_y, eval10_m = CopyTaskDataset(10, n_examples=512, base_seed=999).batch(64, seed=777)
            ev50 = CopyTaskDataset(50, n_examples=512, base_seed=999).batch(64, seed=777)
            ev100 = CopyTaskDataset(100, n_examples=512, base_seed=999).batch(64, seed=777)
            vals = make_valuations(21) if name == "cgmn" else None
            vfn = (lambda _v=vals: _v) if vals is not None else None
            stats = train_probe(model, ds, lambda: import_eval(model, eval10_x, eval10_y, eval10_m, 64), 21,
                                epochs, BS, SEED0 + s, name, "ce", vals_fn=vfn)
            model.eval()
            if name == "cgmn":
                model.set_valuations(make_valuations(101))
            r50 = import_eval(model, *ev50, bs=64)
            if name == "cgmn":
                model.set_valuations(make_valuations(201))
            r100 = import_eval(model, *ev100, bs=64)
            stats["eval_L50"] = r50
            stats["eval_L100"] = r100
            results[key] = stats
            save_results(results, path)
            _log(f"C {name} seed={s}: L10={stats['final']['token_acc']:.4f} L50={r50['token_acc']:.4f} L100={r100['token_acc']:.4f}\n")
    report_C(results)


def report_C(results):
    _log("\n=========== TABLA C: generalización de longitud (train L=10, 40 ep) ===========")
    groups = {}
    for key in sorted(results):
        name, s = key.split("::")
        groups.setdefault(name, []).append(results[key])
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["L=10 (train)", "L=50 (eval)", "L=100 (eval)"]
    x = 0
    xticks, xlabels = [], []
    for name in sorted(groups):
        g = groups[name]
        for li, field in enumerate(["final", "eval_L50", "eval_L100"]):
            vals = [r[field]["token_acc"] for r in g]
            mean, std = float(np.mean(vals)), float(np.std(vals))
            ax.bar(x, mean, 0.5, yerr=std, capsize=3)
            _log(f"{name:12s} {labels[li]:14s} token_acc={mean:.4f}±{std:.4f}")
            xticks.append(x)
            xlabels.append(f"{name}\n{labels[li]}")
            x += 1
        x += 0.5
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("token accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("C: entrenar en L=10, ¿generaliza a L=50/100? (media±std, 3 semillas)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig9_lengen.png"), dpi=150)
    plt.close(fig)
    _log("fig9_lengen.png guardada")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["A", "B1", "B2", "C"])
    ap.add_argument("--partition", type=int, default=0, help="solo para A (0|1)")
    ap.add_argument("--smoke", action="store_true", help="2 epochs, 1 semilla")
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.task == "A":
        task_A(args.partition, args.smoke)
    elif args.task == "B1":
        task_B1(args.smoke)
    elif args.task == "B2":
        task_B2(args.smoke)
    else:
        task_C(args.smoke)
    _log("OK — tarea " + args.task)


if __name__ == "__main__":
    main()