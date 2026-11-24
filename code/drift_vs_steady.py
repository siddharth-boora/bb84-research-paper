"""Drifting vs steady noise at 1024, 2048 and 4096 qubits per round (data for paper Figure 3).
Run from the repo root: python code/drift_vs_steady.py"""
import numpy as np, csv
from drift import ou_paths, pass_table, curves, gain_at

NS, T, E = [1024, 2048, 4096], 50, 3000
SCEN = {"Mordarski drift (mean 10%)": (0.10, 0.030), "Lower-noise drift (mean 5%)": (0.05, 0.015)}
FIXED = np.round(np.r_[0.0, np.arange(0.005, 0.4501, 0.005)], 3)
TARGETS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999, 0.99999, 0.999999]
STRATS = [("steady, fixed f", "steady", "fixed", FIXED), ("drift, fixed f", "drift", "fixed", FIXED),
          ("drift, delayed adaptation", "drift", "delayed", TARGETS)]

out = []
for si, (name, (mu, sig)) in enumerate(SCEN.items()):
    drift = ou_paths(np.random.default_rng(100 + si), E, T, mu=mu, sigma=sig)
    steady = np.full_like(drift, round(drift.mean() / 0.0025) * 0.0025)
    for n in NS:
        table = pass_table(n)
        for label, which, kind, knobs in STRATS:
            hs, ps = curves(n, steady if which == "steady" else drift, kind, knobs, table, mu, rules=("round",))
            h, pts = hs["round"], ps["round"]
            out.append([name, n, label, h, gain_at(pts, 0.01), gain_at(pts, 0.05)])
            print(f"{name:28s} n={n:4d} {label:27s} honest aborts {h*100:5.1f}%  "
                  f"gain@1% {out[-1][4]*100:5.2f}%  gain@5% {out[-1][5]*100:5.2f}%", flush=True)

with open("results/drift_vs_steady.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["scenario", "n", "strategy", "honest_abort", "gain_at_1pct_extra", "gain_at_5pct_extra"])
    w.writerows([[r[0], r[1], r[2]] + [f"{x:.4f}" for x in r[3:]] for r in out])

