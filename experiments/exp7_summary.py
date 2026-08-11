"""Paso 7 — Script de cierre: re-genera figuras y tablas resumen de la fase local.

Lee exp1_copy_task.json, exp2_delayed_memory.json, exp4_ablation.json y
fig6_flops.json; emite:
- fig3_copy_task.png, fig4_delayed_memory.png, fig5_ablation.png (regen.)
- outputs/summary_local.json (tablas agregadas)
- Impresión de tablas para la bitácora.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
MODELS = ["cgmn", "rnn", "gru", "transformer", "cfc"]
MODEL_LABELS = {"cgmn": "CGMN", "rnn": "RNN", "gru": "GRU", "transformer": "Transf.", "cfc": "CfC"}
ABL_NAMES = {
    "1": "CollatzFix1", "2": "NoCollatz", "3": "RandomGate", "4": "SobolGate",
    "5": "CollatzFix1-replica", "6": "CollatzFix2-CP1pct", "7": "CollatzFix3-entropy",
}


def load(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        return json.load(f)


def fig3_exp1(res1):
    ls = sorted(int(k) for k in res1)
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.15
    for i, name in enumerate(MODELS):
        accs = [res1[str(L)].get(name, {}).get("final", {}).get("token_acc", 0.0) for L in ls]
        ax.bar([x + i * width for x in range(len(ls))], accs, width, label=MODEL_LABELS[name])
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


def fig4_exp2(res2):
    ds = sorted(int(k) for k in res2)
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.15
    for i, name in enumerate(MODELS):
        mses = [res2[str(D)].get(name, {}).get("final", {}).get("mse", float("nan")) for D in ds]
        ax.bar([x + i * width for x in range(len(ds))], mses, width, label=MODEL_LABELS[name])
    ax.set_xticks([x + 2 * width for x in range(len(ds))])
    ax.set_xticklabels([f"D={D}" for D in ds])
    ax.set_ylabel("MSE final (a+b)")
    ax.set_title("Delayed Memory — MSE por arquitectura (params ≈ 85k)")
    ax.legend(ncol=5, fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_delayed_memory.png"), dpi=150)
    plt.close(fig)


def fig5_exp4(res4):
    rows = []
    for mode in sorted(res4, key=int):
        accs = [res4[mode][s]["final"]["token_acc"] for s in sorted(res4[mode])]
        rows.append((mode, np.mean(accs), np.std(accs)))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(rows)), [r[1] for r in rows], yerr=[r[2] for r in rows],
           capsize=4, color="#4C72B0")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([ABL_NAMES[r[0]] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("token accuracy (copia L=100)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Ablación CGMN — Copy Task L=100 (media±std, 5 semillas)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_ablation.png"), dpi=150)
    plt.close(fig)


def main():
    res1 = load("exp1_copy_task.json")
    res2 = load("exp2_delayed_memory.json")
    res4 = load("exp4_ablation_L10.json")
    res4b = load("exp4_ablation_L50.json")
    flops = load("fig6_flops.json")

    if res1:
        fig3_exp1(res1)
    if res2:
        fig4_exp2(res2)
    if res4:
        fig5_exp4(res4)
    if res4b:
        fig5_exp4(res4b)

    summary = {"exp1": res1, "exp2": res2, "exp4_L10": res4, "exp4_L50": res4b, "fig6": flops}
    with open(os.path.join(OUT, "summary_local.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=== TABLAS RESUMEN ===")
    if res1:
        print("\nCopy Task — token_acc final:")
        print(f"{'L':>4} " + " ".join(f"{MODEL_LABELS[m]:>10}" for m in MODELS))
        for L in sorted(res1, key=int):
            row = [res1[str(L)].get(m, {}).get("final", {}).get("token_acc", float("nan")) for m in MODELS]
            print(f"{L:>4} " + " ".join(f"{v:>10.4f}" for v in row))
    if res2:
        print("\nDelayed Memory — MSE final:")
        print(f"{'D':>4} " + " ".join(f"{MODEL_LABELS[m]:>10}" for m in MODELS))
        for D in sorted(res2, key=int):
            row = [res2[str(D)].get(m, {}).get("final", {}).get("mse", float("nan")) for m in MODELS]
            print(f"{D:>4} " + " ".join(f"{v:>10.3f}" for v in row))
    for label, r in (("L=10", res4), ("L=50", res4b)):
        if r:
            print(f"\nAblación {label} — token_acc media±std (5 semillas):")
            for mode in sorted(r, key=int):
                accs = [r[mode][s]["final"]["token_acc"] for s in sorted(r[mode])]
                exas = [r[mode][s]["final"]["exact_acc"] for s in sorted(r[mode])]
                print(f"{ABL_NAMES[mode]:22s} {np.mean(accs):.4f}±{np.std(accs):.4f}  exact {np.mean(exas):.4f}±{np.std(exas):.4f}")
    print("\nsummary_local.json guardado en outputs/")


if __name__ == "__main__":
    main()