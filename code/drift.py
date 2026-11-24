"""Drifting channel noise and Eve's strategies.

Noise follows the Ornstein-Uhlenbeck model from Mordarski et al. (arXiv:2610.01792):
    p[t+1] = p[t] + kappa*(mu - p[t]) + sigma*N(0,1), clipped to [0.01, 0.30]
A round is aborted if the measured QBER on the check sample is above 11%.
"""
import numpy as np
from scipy.stats import binom
from bb84 import qber_theory

LIMIT = 0.11
F_GRID = np.round(np.linspace(0, 1, 401), 4)
P_GRID = np.round(np.arange(0.0, 0.3001, 0.0025), 4)


def ou_paths(rng, episodes, T, mu=0.10, sigma=0.03, kappa=0.1, lo=0.01, hi=0.30):
    """Noise paths, starting from the stationary distribution."""
    sd = sigma / np.sqrt(2 * kappa)
    p = np.empty((episodes, T))
    p[:, 0] = np.clip(mu + sd * rng.standard_normal(episodes), lo, hi)
    for t in range(1, T):
        p[:, t] = np.clip(p[:, t - 1] + kappa * (mu - p[:, t - 1])
                          + sigma * rng.standard_normal(episodes), lo, hi)
    return p


def pass_table(n, check_fraction=0.5):
    """P(round passes the 11% check) on a grid of noise p (rows) and intercept f (cols)."""
    k = np.arange(n + 1)
    pk = binom.pmf(k, n, 0.5)
    m = np.floor(check_fraction * k).astype(int)
    thr = np.floor(LIMIT * m + 1e-12)
    table = np.empty((P_GRID.size, F_GRID.size))
    for i, p in enumerate(P_GRID):
        q = qber_theory(F_GRID, p)[:, None]
        pe = binom.cdf(thr[None, :], np.maximum(m, 1)[None, :], q)
        pe[:, m == 0] = 1.0
        table[i] = pe @ pk
    return table


class Eve:
    """Picks Eve's intercept fraction f each round.

    fixed:   same f every round (knob = f)
    delayed: largest f whose pass chance is >= knob, using the noise estimated from
             the previous round's QBER
    instant: same, but using the true noise (not used in the paper)
    """

    def __init__(self, kind, knob, table, mu):
        self.kind, self.knob, self.table, self.mu = kind, knob, table, mu

    def f_for(self, p_est):
        if self.kind == "fixed":
            return np.full(p_est.shape, self.knob)
        row = np.clip(np.rint(p_est / 0.0025).astype(int), 0, P_GRID.size - 1)
        ok = self.table[row] >= self.knob
        idx = np.where(ok.any(1), F_GRID.size - 1 - np.argmax(ok[:, ::-1], 1), 0)
        f = F_GRID[idx]
        f[~ok[:, 0]] = 0.0
        return f


def run_episodes(rng, eve, p_paths, n, check_fraction=0.5, return_qber=False):
    """Simulate every session round by round."""
    E, T = p_paths.shape
    f_hist = np.zeros((E, T)); acc = np.zeros((E, T), bool); q_hist = np.zeros((E, T))
    p_est = np.full(E, eve.mu)
    for t in range(T):
        p = p_paths[:, t]
        f = eve.f_for(p if eve.kind == "instant" else p_est)
        k = rng.binomial(n, 0.5, E)
        m = np.floor(check_fraction * k).astype(int)
        x = rng.binomial(m, qber_theory(f, p))
        qhat = np.where(m > 0, x / np.maximum(m, 1), 0.0)
        acc[:, t] = qhat <= LIMIT + 1e-12
        f_hist[:, t] = f
        q_hist[:, t] = qhat
        # estimate the noise from the QBER, removing Eve's own errors
        p_est = np.clip((qhat - f / 4) / np.maximum(1 - f / 2, 1e-9), 0.0, 0.30)
    if return_qber:
        return f_hist, acc, q_hist
    return f_hist, acc


def gain_at(pts, budget):
    """Eve's best gain with at most `budget` extra aborts (upper envelope, including not attacking)."""
    pts = np.vstack([[0.0, 0.0], pts])
    pts[:, 0] = np.maximum(pts[:, 0], 0.0)
    o = np.argsort(pts[:, 0]); x = pts[o, 0]; y = np.maximum.accumulate(pts[o, 1])
    return float(np.interp(budget, x, y, right=y[-1]))


def curves(n, paths, kind, knobs, table, mu, seed=7, rules=("round", "episode")):
    """(extra aborts, gain) for each knob value.

    round:   each round with QBER > 11% is aborted (used in the paper)
    episode: a whole session is aborted if its average QBER > 11% (not used in the paper)
    """
    def accepted(acc, q, rule):
        return acc if rule == "round" else np.repeat((q.mean(1) <= LIMIT + 1e-12)[:, None], q.shape[1], 1)
    _, a0, q0 = run_episodes(np.random.default_rng(seed + 1), Eve("fixed", 0.0, table, mu), paths, n, return_qber=True)
    honest = {r: 1 - accepted(a0, q0, r).mean() for r in rules}
    pts = {r: [] for r in rules}
    for kn in knobs:
        f, acc, q = run_episodes(np.random.default_rng(seed), Eve(kind, kn, table, mu), paths, n, return_qber=True)
        for r in rules:
            a = accepted(acc, q, r)
            pts[r].append((1 - a.mean() - honest[r], float(np.mean(f * a))))
    return honest, {r: np.array(p) for r, p in pts.items()}


def crossover(mu, diff):
    """Noise level where drift gain minus steady gain crosses zero."""
    for i in range(1, len(mu)):
        if diff[i - 1] <= 0 < diff[i]:
            return mu[i - 1] + (mu[i] - mu[i - 1]) * (-diff[i - 1]) / (diff[i] - diff[i - 1])
    return np.nan if diff[0] <= 0 else mu[0]
