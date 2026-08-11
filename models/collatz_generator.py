"""Collatz generator module (CGMN).

Deterministic memory-gate source based on generalized Collatz orbits (3n+c).

Design decision (DEST bug warning): the default is INDEX MODE — a fresh odd
seed is derived per index t via a splitmix64 mix of (base_seed, t), avoiding
the single-orbit collapse to the fixed point 1 that plagued DEST.

Functions
---------
syracuse_step(n, c) -> (n', v)
    One U_c step: u = 3n+c, v = v2(u), return (u >> v, v).
derive_seed(base_seed, t) -> odd int in [2^62, 2^63)
    Deterministic, distinct, large odd seed per index t.
collatz_valuation_index_mode(T, c, base_seed, K) -> list[int]
    T valuations, one per index t (the last of K Syracuse steps).
collatz_valuation_entropy_scheduled(T, c, n0, alpha_fn, K) -> list[int]
    Single-orbit variant with growing perturbation alpha_fn(t) (ablation
    mode 7, CollatzFix3Gate). At alpha=0 it degenerates (DEST-like); the
    perturbation restores entropy as alpha grows toward 0.5.
"""

import math

_MASK61 = (1 << 61) - 1


def syracuse_step(n, c=1):
    """One step of U_c(x) = (3x+c) / 2^v2(3x+c). Returns (new_n, valuation)."""
    u = 3 * n + c
    v = (u & -u).bit_length() - 1
    return u >> v, v


def derive_seed(base_seed, t):
    """Deterministic odd seed in [2^62, 2^63) for index t (splitmix64-style)."""
    x = (base_seed ^ (t * 0x9E3779B97F4A7C15)) & _MASK61
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & _MASK61
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & _MASK61
    x ^= x >> 31
    return (x | 1) + (1 << 62)


def collatz_valuation_index_mode(T, c=1, base_seed=42, K=50):
    """Valuation of the K-th Syracuse step from a per-index fresh seed.

    Returns a list of T ints (valuations >= 1). Empirically these follow
    P(v2(3n+c) = k) = 2^-k for odd seeds (validated in exp0).
    """
    out = [0] * T
    for t in range(T):
        n = derive_seed(base_seed, t)
        v = 0
        for _ in range(K):
            n, v = syracuse_step(n, c)
        out[t] = v
    return out


def _default_alpha_fn(T):
    def alpha(t):
        if T <= 1:
            return 0.0
        return 0.25 * (1.0 - math.cos(math.pi * t / (T - 1)))
    return alpha


def collatz_valuation_entropy_scheduled(T, c=1, n0=None, alpha_fn=None, K=1):
    """Single orbit, growing Cranley-Patterson-style perturbation.

    For index t: state = orbit state after t steps from n0; perturbed =
    state + round(alpha_fn(t) * 2^53 * hash01(t)); valuation of the
    perturbed state (forced odd) is recorded.

    alpha_fn: float in [0, 1]; default cosine 0 -> 0.5.
    At alpha=0 the values come straight from one orbit (DEST-like collapse
    for large t). This is intentional: it tests entropy scheduling.
    """
    if n0 is None:
        n0 = (1 << 62) | 12345
    if alpha_fn is None:
        alpha_fn = _default_alpha_fn(T)
    out = [0] * T
    n = n0
    for t in range(T):
        if K > 0:
            for _ in range(K):
                n, _ = syracuse_step(n, c)
        a = alpha_fn(t)
        if a > 0.0:
            h = _hash01(t, salt=0xC0FFEE)
            delta = int(round(a * (1 << 53) * h))
            m = n + delta
            m |= 1
        else:
            m = n
        _, v = syracuse_step(m, c)
        out[t] = v
    return out


def _hash01(t, salt=0xDEADBEEF):
    x = (t * 0x9E3779B97F4A7C15) ^ salt
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & _MASK61
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & _MASK61
    x ^= x >> 31
    return (x & _MASK61) / float(1 << 61)
