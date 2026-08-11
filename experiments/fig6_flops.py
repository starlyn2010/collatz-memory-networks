"""Paso 7 — Figura 6 (v2): FLOPs por paso vs longitud de secuencia T.

Incluye el término cuadrático de la atención (qk^T + softmax ∝ T) para el
Transformer; los recurrentes son O(1) por paso. Así se ve la ventaja
estructural del reloj Collatz/GRU para secuencias largas.

Salida: outputs/fig6_flops.png + fig6_flops.json
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")

D = 64
KAPPA = 10
TS = [10, 50, 100, 200, 500, 1000]

ARCHS = {
    "CGMN": {"h": 64, "type": "cgmn"},
    "GRU": {"h": 64, "type": "gru"},
    "RNN": {"h": 96, "type": "rnn"},
    "Transformer": {"d": 56, "ffn": 112, "nhead": 2, "type": "trans"},
    "CfC": {"h": 53, "type": "cfc"},
}


def fstep(name, a, T):
    if a["type"] == "cgmn":
        return 2 * 3 * a["h"] * (D + a["h"]) + 2 * a["h"] * (KAPPA + 1)
    if a["type"] == "gru":
        return 2 * 3 * a["h"] * (D + a["h"])
    if a["type"] == "rnn":
        return 2 * a["h"] * (D + a["h"])
    if a["type"] == "cfc":
        return 2 * 5 * a["h"] * (D + a["h"])
    if a["type"] == "trans":
        d, h = a["d"], a["nhead"]
        attn_scores = 2 * h * d * T + 3 * T * d          # qk^T ∝ T, softmax ∝ T
        attn_apply = 2 * h * d * T                        # v weighted
        mlp = 2 * d * a["ffn"]
        return 2 * (attn_scores + attn_apply + mlp)       # 2 capas
    raise ValueError


def main():
    rows = {}
    for name, a in ARCHS.items():
        rows[name] = {f"T{T}": fstep(name, a, T) for T in TS}
        print(f"{name:12s}", " ".join(f"T{T}:{fstep(name,a,T)/1e6:.2f}M" for T in TS))

    with open(os.path.join(OUT, "fig6_flops.json"), "w") as f:
        json.dump(rows, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ARCHS:
        ax.plot(TS, [rows[name][f"T{T}"] / 1e6 for T in TS], "o-", label=name, linewidth=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Longitud de secuencia T")
    ax.set_ylabel("MFLOPs por paso (estimado, forward)")
    ax.set_title("Costo por paso vs T: atención cuadrática vs recurrencia O(1)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_flops.png"), dpi=150)
    print("fig6_flops.png guardada")


if __name__ == "__main__":
    main()