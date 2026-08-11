"""Paso 4 — Copy Task: CGMN vs RNN/GRU/Transformer/CfC con params ~=.

L ∈ {10, 50, 100, 200}. AdamW lr=1e-3 coseno a 1e-5, wd=1e-4, clip 1.0.
Métricas: token_acc, exact_acc, mse, s/epoch, pico RAM.
Salida: outputs/exp1_copy_task.json + fig3_copy_task.png (acumulativo).

Uso: python3 experiments/exp1_copy_task.py [--min-l 10] [--max-l 200] [--epochs N]
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ".")

from datasets.copy_task import VOCAB, CopyTaskDataset  # noqa: E402
from models.cgmn import CGMN  # noqa: E402
from models.baselines import build_baseline  # noqa: E402
from experiments.train_common import train_loop, make_valuations, import_eval  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
RESULTS = os.path.join(OUT, "exp1_copy_task.json")
LOG = os.path.join(OUT, "exp1_log.txt")
MODELS = ["cgmn", "rnn", "gru", "transformer", "cfc"]
EPOCHS_BY_L = {10: 40, 50: 30, 100: 20, 200: 12}
BS = 64
SEED = 1234


def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            return json.load(f)
    return {}


def _log(s):
    print(s, flush=True)
    with open(LOG, "a") as f:
        f.write(s + "\n")
        f.flush()


def save_results(res):
    tmp = RESULTS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=2)
    os.replace(tmp, RESULTS)


def acquire_lock():
    import os as _os

    lock = os.path.join(OUT, "exp1.lock")
    if _os.path.exists(lock):
        try:
            pid = int(open(lock).read().strip())
            _os.kill(pid, 0)
            print(f"ya hay un run activo (PID {pid}) — salgo", flush=True)
            sys.exit(0)
        except (ProcessLookupError, ValueError):
            pass
    with open(lock, "w") as f:
        f.write(str(_os.getpid()))
    return lock


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-l", type=int, default=10)
    ap.add_argument("--max-l", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()

    results = load_results()
    lock = acquire_lock()
    ls = [l for l in (10, 50, 100, 200) if args.min_l <= l <= args.max_l]
    models = [m for m in MODELS if not args.only or m == args.only]

    for L in ls:
        seq_len = 2 * L + 1
        epochs = args.epochs or EPOCHS_BY_L[L]
        ds = CopyTaskDataset(L, n_examples=2048, base_seed=SEED)
        eval_x, eval_y, eval_m = CopyTaskDataset(L, n_examples=512, base_seed=999).batch(64, seed=777)
        vals = make_valuations(seq_len) if "cgmn" in models else None

        for name in models:
            if name in results.get(str(L), {}) and "final" in results[str(L)][name]:
                print(f"L={L} {name}: ya en JSON, se omite")
                continue
            print(f"=== L={L} ({seq_len} tokens) {name} epochs={epochs} ===")
            torch.manual_seed(SEED)
            model = CGMN(VOCAB, max_seq_len=seq_len + 8) if name == "cgmn" else build_baseline(name, VOCAB, max_len=seq_len + 8)
            n_params = sum(p.numel() for p in model.parameters())
            ev_fn = lambda: import_eval(model, eval_x, eval_y, eval_m, bs=64)
            stats = train_loop(
                model,
                lambda bs, seed: ds.batch(bs, seed),
                ev_fn,
                seq_len,
                epochs,
                bs=BS,
                seed=SEED,
                vals_fn=(lambda _v=vals: _v) if name == "cgmn" else None,
                log=lambda s: _log(s),
            )
            entry = {k: v for k, v in stats.items() if k != "loss_hist"}
            entry["n_params"] = n_params
            entry["epochs"] = epochs
            results.setdefault(str(L), {})[name] = entry
            save_results(results)
            print(f"L={L} {name}: {entry['final']}\n")

    make_figure(results)
    print("OK — resultados en", RESULTS)


def make_figure(results):
    ls = sorted(int(k) for k in results)
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.15
    for i, name in enumerate(MODELS):
        accs = []
        for L in ls:
            entry = results[str(L)].get(name)
            accs.append(entry["final"]["token_acc"] if entry else 0.0)
        ax.bar([x + i * width for x in range(len(ls))], accs, width, label=name)
    ax.set_xticks([x + 2 * width for x in range(len(ls))])
    ax.set_xticklabels([f"L={L}" for L in ls])
    ax.set_ylabel("token accuracy (copia)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Copy Task — accuracy por arquitectura (params ≈ 85k)")
    ax.legend(ncol=5, fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_copy_task.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()