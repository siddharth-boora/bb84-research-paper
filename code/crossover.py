"""Where does drift switch from hurting to helping the adaptive Eve? (paper Figure 4 and Table 2)
Average noise is swept from 4.5% to 10%, and the whole experiment is repeated with 5 random seeds
to get confidence intervals. Takes about 25 minutes on 3 cores.
Run from the repo root: python code/crossover.py"""
import numpy as np, csv
from multiprocessing import Pool
from scipy.stats import t as tdist
from drift import ou_paths, pass_table, gain_at, curves, crossover

TARGETS = [0.5, 0.7, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999, 0.999999]

NS, T, E, SEEDS = [1024, 2048, 4096], 50, 1500, [11, 22, 33, 44, 55]
MUS = np.round(np.arange(0.045, 0.1001, 0.005), 4)
FIXED = np.round(np.arange(0.005, 0.3001, 0.01), 3)


def job(args):
    mu, n, seed = args
    rng = np.random.default_rng(seed * 1000 + int(mu * 1e4))
    drift = ou_paths(rng, E, T, mu=mu, sigma=0.3 * mu)
    steady = np.full_like(drift, round(drift.mean() / 0.0025) * 0.0025)
    table = pass_table(n)
    hs, ps = curves(n, steady, "fixed", FIXED, table, mu, seed=seed)
    hd, pd = curves(n, drift, "delayed", TARGETS, table, mu, seed=seed + 1)
    return [[mu, n, seed, r, gain_at(ps[r], 0.05), gain_at(pd[r], 0.05)] for r in ("round", "episode")]


if __name__ == "__main__":
    with Pool(3) as pool:
        out = [r for rows in pool.map(job, [(mu, n, s) for s in SEEDS for mu in MUS for n in NS]) for r in rows]
    with open("results/crossover_runs.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["mu", "n", "seed", "abort_rule", "steady_gain_5pct", "delayed_gain_5pct"])
        w.writerows([[f"{r[0]:.3f}", r[1], r[2], r[3], f"{r[4]:.5f}", f"{r[5]:.5f}"] for r in out])

    # crossover point for each seed, then mean and 95% confidence interval
    lines = []
    for n in NS:
        xs = []
        for s in SEEDS:
            rows = sorted([r for r in out if r[1] == n and r[2] == s and r[3] == "round"])
            mu = np.array([r[0] for r in rows]) * 100
            diff = np.array([r[5] - r[4] for r in rows]) * 100
            xs.append(crossover(mu, diff))
        xs = np.array(xs); m, sd = xs.mean(), xs.std(ddof=1)
        half = tdist.ppf(0.975, len(xs) - 1) * sd / np.sqrt(len(xs))
        lines.append(f"n={n}: crossover {m:.2f}%  (95% CI {m-half:.2f} to {m+half:.2f}; per seed {np.round(xs, 2)})")
    open("results/crossover_summary.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
