"""Steady noise: Eve's chance of passing the 11% check (paper Figure 2 and Table 1).
Run from the repo root: python code/steady_noise.py > results/steady_noise.txt"""
import numpy as np, csv
from bb84 import run_bb84, f_max
from evasion import p_evade, f_at

NS = [1024, 2048, 4096]
ES = [0.02, 0.05, 0.08, 0.10]

# (a) Simulation vs exact calculation
rng = np.random.default_rng(2026)
print("(a) Simulated vs exact P(evade), 20,000 rounds each, check = half of sifted bits")
print(f"{'n':>5} {'e':>5} {'f':>5} {'sim':>7} {'exact':>7}")
worst = 0
for n in [1024, 4096]:
    for e, f in [(0.02, 0.30), (0.05, 0.25), (0.08, 0.14), (0.10, 0.05)]:
        R = 20000
        ev = np.mean([run_bb84(n, f, e, rng, check_fraction=0.5)["qber"] <= 0.11 for _ in range(R)])
        ex = p_evade(n, f, e)
        se = np.sqrt(ex * (1 - ex) / R)
        worst = max(worst, abs(ev - ex) / se if se else 0)
        print(f"{n:5d} {e:5.2f} {f:5.2f} {ev:7.4f} {ex:7.4f}")
print(f"largest gap = {worst:.1f} standard errors")

# (b) Table of thresholds
print("\n(b) Intercept fraction Eve can use, by required evasion probability")
hdr = ["noise_e", "n", "f_max_formula", "f_at_P50", "f_at_P90", "f_at_P99",
       "P_evade_at_f_max", "false_alarm_no_eve"]
rows = []
for e in ES:
    for n in NS:
        fm = f_max(e)
        r = [e, n, fm, f_at(n, e, 0.5), f_at(n, e, 0.9), f_at(n, e, 0.99),
             p_evade(n, fm, e), 1 - p_evade(n, 0.0, e)]
        rows.append(r)
        print(f"e={e:.2f} n={n:5d}  f_max={fm:.3f}  P50={r[3]:.3f}  P90={r[4]:.3f}  "
              f"P99={r[5]:.3f}  P(evade|f_max)={r[6]:.3f}  false alarm={r[7]:.4f}")
with open("results/steady_thresholds.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(hdr)
    w.writerows([[f"{x:.4f}" if isinstance(x, float) else x for x in r] for r in rows])

# (c) Chart: P(evade) vs f, one panel per noise level, one line per n
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
colors = {1024: "#d64545", 2048: "#2e9e5b", 4096: "#2a78d6"}
styles = {1024: "-", 2048: "--", 4096: ":"}
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), sharey=True)
fs = np.linspace(0, 0.5, 201)
for ax, e in zip(axes, ES):
    for n in NS:
        ax.plot(fs, [p_evade(n, f, e) for f in fs], color=colors[n], ls=styles[n], lw=2.2, label=f"{n} qubits")
    fm = f_max(e)
    ax.axvline(fm, color="#888888", lw=1, ls="--")
    ax.text(fm + 0.012, 0.90, f"f_max = {fm:.3f}", color="#555555", fontsize=8.5,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_title(f"Channel noise {e:.0%}", fontsize=10.5)
    ax.set_xlim(0, 0.5); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Eve's intercept fraction f")
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#999999"); ax.spines["bottom"].set_color("#999999")
axes[0].set_ylabel("P(Eve evades the 11% check)")
axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
fig.suptitle("Chance Eve escapes detection under steady noise (check sample = half of sifted bits)",
             fontsize=11.5, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("figures/fig2.png", dpi=200)
print("\nsaved step2_evasion_steady.png, step2_thresholds.csv")
