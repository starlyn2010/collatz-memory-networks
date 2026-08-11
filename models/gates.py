"""Máscaras de compuerta para la ablación de CGMN (Copy Task L=100).

Modos (7):
  1. CollatzFix1   — modo índice puro (valuations precomputadas + W_m
                     entrenable; gate_override=None).
  2. NoCollatz     — m_t = 1 (GRU estándar; gate_override=None).
  3. RandomGate    — m_t ~ Bernoulli(0.5) i.i.d. (reproducible por seed).
  4. SobolGate     — m_t = 1[Sobol_1d(t·B·hidden+i) < 0.5] (scipy qmc).
  5. CollatzFix2   — modo índice + ~1% perturbación Cranley-Patterson en
                     la semilla; m_t = sigmoid(v_t/3) fijo (no entrenable).
  6. CollatzFix3   — entropy scheduling: órbita única + α(t) coseno 0→0.5;
                     m_t = sigmoid(v_t/3) fijo (no entrenable).

Nota metrológica (registrada en bitácora): RandomGate y SobolGate producen
máscaras {0,1} con media 0.5; los modos Collatz fixos producen valores
continuos sigmoid(v/3) ∈ (~0.55, ~0.95), media ~0.75. La diferencia de
escala es una propiedad del mecanismo, se reporta como tal.
"""

import torch

from .collatz_generator import collatz_valuation_entropy_scheduled, derive_seed, syracuse_step


def mask_random(T, B, hidden, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(T, B, hidden, generator=g) < 0.5).float()


def mask_sobol(T, B, hidden, seed=1):
    try:
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=1, scramble=False, seed=seed)
        pts = sampler.random(T * B * hidden)[:, 0]
        return torch.from_numpy((pts < 0.5).astype("float32")).reshape(T, B, hidden)
    except Exception as e:
        raise RuntimeError("Sobol no disponible: " + str(e))


def _fix1_none():
    return None


def mask_collatz_fix2(T, B, hidden, base_seed=42, c=1, K=50, frac=0.01):
    """RQMC híbrido: semillas del modo índice perturbadas ~1% (CP)."""
    phi = 0.6180339887498949
    vals = []
    for t in range(T):
        shift = int(round(frac * (1 << 61) * ((t * phi) % 1.0)))
        n = ((derive_seed(base_seed, t) + shift) | 1)
        v = 0
        for _ in range(K):
            n, v = syracuse_step(n, c)
        vals.append(v)
    v = torch.as_tensor(vals, dtype=torch.float32)
    return torch.sigmoid(v / 3.0).view(T, 1, 1).expand(T, B, hidden).contiguous()


def mask_collatz_fix3(T, B, hidden, c=1, n0=None):
    """Entropy scheduling: α(t) coseno 0→0.5 sobre órbita única."""
    vals = collatz_valuation_entropy_scheduled(T, c=c, n0=n0, alpha_fn=None)
    v = torch.as_tensor(vals, dtype=torch.float32)
    return torch.sigmoid(v / 3.0).view(T, 1, 1).expand(T, B, hidden).contiguous()


def mask_const(T, B, hidden, val):
    """Puerta constante (control de nivel medio)."""
    return torch.full((T, B, hidden), float(val))


def _valuation_ks(T, base_seed=42, K=50):
    from .collatz_generator import collatz_valuation_index_mode

    ks = collatz_valuation_index_mode(T, c=1, base_seed=base_seed, K=K)
    return torch.as_tensor(ks, dtype=torch.float32)


def mask_collatz_scale(T, B, hidden, mean, eps, base_seed=42, K=50):
    """Puerta Collatz con PROMEDIO EMPAREJADO: m_t = clamp(mean + eps·(2^-k - 1/3)).

    E[2^-k] = 1/3 (ley exacta del autor) => E[m_t] = mean exacto. La forma
    temporal (cuándo sube/baja) es la única diferencia vs mask_const."""
    k = _valuation_ks(T, base_seed, K)
    g = torch.pow(2.0, -k)
    m = (mean + eps * (g - 1.0 / 3.0)).clamp(0.0, 1.0)
    return m.view(T, 1, 1).expand(T, B, hidden).contiguous()


def mask_collatz_binary(T, B, hidden, lo=0.9, hi=1.0, base_seed=42, K=50):
    """Puerta Collatz binaria con promedio emparejado: P(k=1)=1/2.

    m_t = hi si k_t == 1 (evento común), lo si k_t >= 2 (raro).
    Promedio = (hi + lo) / 2 — emparejar con mask_const((hi+lo)/2)."""
    k = _valuation_ks(T, base_seed, K)
    m = torch.where(k == 1, torch.full_like(k, float(hi)), torch.full_like(k, float(lo)))
    return m.view(T, 1, 1).expand(T, B, hidden).contiguous()