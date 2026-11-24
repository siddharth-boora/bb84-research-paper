"""Exact probability that Eve evades the QBER check (steady noise).

Per round: n qubits sent; sifted k ~ Binomial(n, 1/2); check sample m = floor(c*k);
errors X ~ Binomial(m, q) with q = e + (f/4)(1-2e). Abort if X/m > limit.
P(evade) = sum_k P(k) * P(X <= floor(limit*m)).
"""
import numpy as np
from scipy.stats import binom
from bb84 import qber_theory


def p_evade(n, f, e, limit=0.11, check_fraction=0.5):
    q = qber_theory(f, e)
    k = np.arange(n + 1)
    pk = binom.pmf(k, n, 0.5)
    m = np.floor(check_fraction * k).astype(int)
    thr = np.floor(limit * m + 1e-12)
    pe = np.where(m > 0, binom.cdf(thr, np.maximum(m, 1), q), 1.0)
    return float(np.sum(pk * pe))


def f_at(n, e, target, limit=0.11, check_fraction=0.5):
    """Largest f with P(evade) >= target (bisection; P falls as f rises)."""
    if p_evade(n, 0.0, e, limit, check_fraction) < target:
        return 0.0
    lo, hi = 0.0, 1.0
    if p_evade(n, 1.0, e, limit, check_fraction) >= target:
        return 1.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if p_evade(n, mid, e, limit, check_fraction) >= target:
            lo = mid
        else:
            hi = mid
    return lo
