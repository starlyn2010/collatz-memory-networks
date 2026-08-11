"""Paso 1 validation: empirical valuation frequencies vs theoretical 2^-k.

Runs collatz_valuation_index_mode(10000, c=1, base_seed=42, K=50),
compares empirical P(v2 = k) against 2^-k for k=1..10.
Success criterion: max abs error < 0.01 for k=1..10 (same strictness as
the published papers).

Outputs: outputs/fig2_validacion_generador.png, outputs/fig2_validacion.json
"""

import json
import sys
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.collatz_generator import collatz_valuation_index_mode  # noqa: E402

T = 10000
C = 1
BASE_SEED = 42
K = 50
KMAX = 16


def main():
    vals = collatz_valuation_index_mode(T, c=C, base_seed=BASE_SEED, K=K)
    counts = np.bincount(np.array(vals, dtype=np.int64), minlength=KMAX + 1)[1 : KMAX + 1]
    emp = counts / T
    theo = 2.0 ** -np.arange(1, KMAX + 1)
    abs_err = np.abs(emp - theo)

    max_err_1_10 = float(abs_err[:10].max())
    passed = max_err_1_10 < 0.01

    results = {
        "T": T, "c": C, "base_seed": BASE_SEED, "K": K,
        "empirical": {int(k): float(v) for k, v in enumerate(emp.tolist(), start=1)},
        "theoretical": {int(k): float(v) for k, v in enumerate(theo.tolist(), start=1)},
        "abs_error": {int(k): float(v) for k, v in enumerate(abs_err.tolist(), start=1)},
        "max_abs_error_k1_10": max_err_1_10,
        "passed_criterion_0_01": passed,
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "fig2_validacion.json"), "w") as f:
        json.dump(results, f, indent=2)

    ks = np.arange(1, KMAX + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(ks - 0.2, emp, width=0.4, label="Empírico", color="#4C72B0")
    ax.bar(ks + 0.2, theo, width=0.4, label="Teórico 2^-k", color="#DD8452")
    ax.set_xlabel("Valuación k = v2(3n+1)")
    ax.set_ylabel("Frecuencia")
    ax.set_yscale("log")
    ax.set_title("Validación generador Collatz (modo índice, T=10000, K=50)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig2_validacion_generador.png"), dpi=150)
    plt.close(fig)

    print(f"max_abs_error k=1..10: {max_err_1_10:.6f} -> {'PASA' if passed else 'NO PASA'} (criterio < 0.01)")
    for k in range(1, 11):
        print(f"  k={k:2d}  emp={emp[k-1]:.5f}  theo={theo[k-1]:.5f}  err={abs_err[k-1]:.5f}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
