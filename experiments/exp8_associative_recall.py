"""exp8 — Associative Recall (MQAR simplificado, estilo Zoology) CGMN vs GRU.

Tarea: N pares clave-valor [k1 v1 ... kN vN], D PADs de separación, N consultas
(las claves en orden aleatorio), y al final N posiciones objetivo = los valores
en el orden de las consultas. La red debe RECUPERAR el valor de cada clave.

seq_len = 2N + D + N + N.

Vocab: 0=PAD, 1=GAP, claves 2..N_KEYS+1, valores N_KEYS+2..2*N_KEYS+1.

Modelos: CGMN vs GRU (~85-90k params). 3 semillas. Durante la evaluacion
final de CGMN se recolectan estadisticas de mecanismo por posicion
(z_eff, m, ||h||) para la seccion de analisis del paper.

Uso: python3 experiments/exp8_associative_recall.py [--partition 0|1] [--n 4] [--d 0|8]
"""

import argparse
import json
import math
import os
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cgmn import CGMN
from models.baselines import build_baseline

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
RESULTS = os.path.join(OUT, "exp8_associative.json")
LOG = os.path.join(OUT, "exp8_log.txt")

PAD, GAP = 0, 1
N_KEYS = 10
VOCAB = 2 * N_KEYS + 2
BS = 64
N_EXAMPLES = 2048
EVAL_EXAMPLES = 512
SEED0 = 1234
N_SEEDS = 3
EPOCHS = 40


def _log(s):
    print(s, flush=True)
    with open(LOG, "a") as f:
        f.write(s + "\n")
        f.flush()


def make_dataset(N, D, n_examples, base_seed):
    rng = np.random.RandomState(base_seed)
    keys = np.arange(2, 2 + N)
    seq_len = 4 * N + D
    xs = np.zeros((n_examples, seq_len), dtype=np.int64)
    ys = np.full((n_examples, seq_len), -1, dtype=np.int64)
    masks = np.zeros((n_examples, seq_len), dtype=np.float32)
    for e in range(n_examples):
        vals = rng.choice(np.arange(2 + N, 2 + 2 * N), size=N, replace=False)
        pairs = np.empty(2 * N, dtype=np.int64)
        pairs[0::2] = keys
        pairs[1::2] = vals
        queries = rng.permutation(keys)
        ans = np.array([vals[np.where(keys == q)[0][0]] for q in queries])
        x = np.concatenate([pairs, np.full(D, GAP, dtype=np.int64), queries, np.full(N, PAD, dtype=np.int64)])
        xs[e] = x
        ys[e, 4 * N + D : 4 * N + D + N] = ans
        masks[e, 4 * N + D : 4 * N + D + N] = 1.0
    return xs, ys, masks, seq_len


def make_valuations(seq_len, base_seed=42, K=50):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.collatz_generator import collatz_valuation_index_mode

    return torch.tensor(collatz_valuation_index_mode(seq_len, 1, base_seed, K), dtype=torch.long)


def eval_metrics(model, xs, ys, masks, vals, track=None):
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(xs)
        if isinstance(model, CGMN):
            logits = model(xb, valuations=vals(), track=track)
        else:
            logits = model(xb)
        pred = logits.argmax(-1).numpy()
        m = masks
        correct = (pred == ys) * m
        token_acc = float(correct.sum() / max(m.sum(), 1))
        exact = float((correct.sum(axis=1) == m.sum(axis=1)).mean())
    return {"token_acc": token_acc, "exact_acc": exact}


def train_one(model, N, D, epochs, vals, seed):
    torch.manual_seed(SEED0 + seed)
    np.random.seed(SEED0 + seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xs_tr, ys_tr, m_tr, seq_len = make_dataset(N, D, N_EXAMPLES, SEED0)
    xs_ev, ys_ev, m_ev, _ = make_dataset(N, D, EVAL_EXAMPLES, 999)
    total_steps = epochs * (N_EXAMPLES // BS)
    step = 0
    t0 = time.time()
    ep_times = []
    ram = 0.0
    best = None
    for ep in range(epochs):
        model.train()
        ti = time.time()
        perm = np.random.permutation(N_EXAMPLES)
        for i in range(0, N_EXAMPLES, BS):
            idx = perm[i : i + BS]
            xb = torch.from_numpy(xs_tr[idx])
            yb = torch.from_numpy(ys_tr[idx])
            mb = torch.from_numpy(m_tr[idx])
            opt.zero_grad(set_to_none=True)
            if isinstance(model, CGMN):
                logits = model(xb, valuations=vals())
            else:
                logits = model(xb)
            ce = -(torch.log_softmax(logits, -1).gather(-1, yb.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mb).sum() / mb.sum().clamp(min=1)
            ce.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for g in opt.param_groups:
                g["lr"] = 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * step / max(total_steps - 1, 1)))
            step += 1
        ep_times.append(time.time() - ti)
        ram = max(ram, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
        if ep % 5 == 0 or ep == epochs - 1:
            ev = eval_metrics(model, xs_ev, ys_ev, m_ev, vals)
            if best is None or ev["token_acc"] > best["token_acc"]:
                best = dict(ev, epoch=ep)
            torch.set_num_threads(2)
            _log(f"[{model.__class__.__name__[:-8] if hasattr(model,'__class__') else 'm'}] N={N} D={D} seed={seed} ep {ep+1}/{epochs} token={ev['token_acc']:.4f} exact={ev['exact_acc']:.4f} ce={ce.item():.3f}")
    torch.set_num_threads(2)
    ev = eval_metrics(model, xs_ev, ys_ev, m_ev, vals)
    if best is None or ev["token_acc"] > best["token_acc"]:
        best = dict(ev, epoch=epochs - 1)
    stats = None
    if isinstance(model, CGMN):
        track = []
        eval_metrics(model, xs_ev, ys_ev, m_ev, vals, track=track)
        stats = {"z": [r[0] for r in track], "m": [r[1] for r in track], "h": [r[2] for r in track]}
    return {
        "final": ev,
        "best": best,
        "sec_per_epoch": float(np.mean(ep_times)),
        "epochs": epochs,
        "n_params": sum(p.numel() for p in model.parameters()),
        "ram_peak_mb": round(ram, 1),
        "stats": stats,
    }


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", type=int, default=0)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--d", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_grad_enabled(True)

    N = args.n
    for D in ([0, 8] if args.d == 0 else [args.d]):
        if args.d != 0 and D != args.d:
            continue
        seq_len = 4 * N + D
        vals = make_valuations(seq_len)

        runs = [(name, s) for name in ("cgmn", "gru") for s in range(N_SEEDS)]
        if args.partition == 1:
            runs = runs[1::2]
        else:
            runs = runs[::2]

        results = load_results()
        key = f"N{N}D{D}"
        for name, seed in runs:
            if key in results and name in results[key]:
                _log(f"N={N} D={D} {name} seed={seed}: ya en JSON")
                continue
            _log(f"=== N={N} D={D} {name} seed={seed} ===")
            torch.manual_seed(SEED0 + seed)
            model = CGMN(VOCAB, max_seq_len=seq_len + 8) if name == "cgmn" else build_baseline(name, VOCAB, max_len=seq_len + 8)
            stats = train_one(model, N, D, EPOCHS, (lambda _v=vals: _v) if name == "cgmn" else None, seed)
            results.setdefault(key, {})[name] = stats
            results[key][name]["seed"] = seed
            save_results(results)
            _log(f"N={N} D={D} {name} seed={seed}: token={stats['final']['token_acc']:.4f} exact={stats['final']['exact_acc']:.4f}\n")
        _log(f"progreso N{N}D{D}: " + ", ".join(f"{n}:{round(results[key][n]['final']['token_acc'],4)}" for n in results.get(key, {})))


if __name__ == "__main__":
    main()