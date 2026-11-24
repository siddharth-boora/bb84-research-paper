"""Draws paper Figures 1, 3 and 4 from the results files. Run from the repo root: python code/make_figures.py"""
import numpy as np, pandas as pd, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from drift import ou_paths

COL = {1024: "#d64545", 2048: "#2e9e5b", 4096: "#2a78d6"}
LS = {1024: "-", 2048: "--", 4096: ":"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
def tidy(ax):
    ax.grid(axis="y", color="#e6e6e6", lw=0.8)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#999999")

# Figure 1: example drifting noise
rng = np.random.default_rng(3)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, (mu, title) in zip(axes, [(0.05, "Average noise 5%"), (0.10, "Average noise 10%")]):
    p = ou_paths(rng, 3, 50, mu=mu, sigma=0.3 * mu)
    for i, c in enumerate(["#2a78d6", "#e07b39", "#2e9e5b"]):
        ax.plot(range(1, 51), p[i] * 100, color=c, lw=1.6, label=f"drifting noise, session {i + 1}")
    ax.axhline(mu * 100, color="#8a8a8a", ls="--", lw=1.6, label="steady noise")
    ax.axhline(11, color="#d64545", lw=1.4, label="11% abort line")
    ax.set_title(title, loc="left", fontsize=10.5); ax.set_xlabel("Round"); tidy(ax)
axes[0].set_ylabel("Channel noise (%)")
axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout(); fig.savefig("figures/fig1.png", dpi=200); plt.close(fig)

# Figure 3: Eve's information gain vs sample size
d = pd.read_csv("results/drift_vs_steady.csv")
style = {"steady, fixed f": ("#8a8a8a", "--", "s", "Steady noise"),
         "drift, fixed f": ("#e07b39", "-", "o", "Drifting noise, Eve doesn't adapt"),
         "drift, delayed adaptation": ("#7a5cc4", "-.", "D", "Drifting noise, Eve adapts")}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, (scen, title) in zip(axes, [("Lower-noise drift (mean 5%)", "Average noise 5%"),
                                     ("Mordarski drift (mean 10%)", "Average noise 10%")]):
    for lab, (c, ls, mk, name) in style.items():
        y = [d[(d.scenario == scen) & (d.n == n) & (d.strategy == lab)].gain_at_5pct_extra.iloc[0] * 100 for n in [1024, 2048, 4096]]
        ax.plot(range(3), y, color=c, ls=ls, marker=mk, lw=2, ms=6, label=name)
    ax.set_xticks(range(3), ["1024", "2048", "4096"]); ax.set_xlabel("Qubits sent per round")
    ax.set_title(title, loc="left", fontsize=10.5); ax.set_ylim(bottom=0); tidy(ax)
axes[0].set_ylabel("Eve's information gain (% of qubits)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=9, loc="lower center", ncol=3)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig("figures/fig3.png", dpi=200); plt.close(fig)

# Figure 4: crossover with error bands (needs step 7)
if os.path.exists("results/crossover_runs.csv"):
    u = pd.read_csv("results/crossover_runs.csv"); u = u[u.abort_rule == "round"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for n in [1024, 2048, 4096]:
        g = u[u.n == n].assign(diff=lambda x: (x.delayed_gain_5pct - x.steady_gain_5pct) * 100)
        s = g.groupby("mu")["diff"].agg(["mean", "std", "count"])
        mu = s.index.values * 100; half = 2.776 * s["std"] / np.sqrt(s["count"])
        ax.plot(mu, s["mean"], color=COL[n], ls=LS[n], lw=2.2, marker="o", ms=3.5, label=f"{n} qubits per round")
        ax.fill_between(mu, s["mean"] - half, s["mean"] + half, color=COL[n], alpha=0.15, lw=0)
    ax.axhline(0, color="#555555", lw=1)
    ax.text(0.02, 0.96, "above 0: drift helps Eve", transform=ax.transAxes, va="top", fontsize=9, color="#555555")
    ax.text(0.98, 0.04, "below 0: drift hurts Eve", transform=ax.transAxes, ha="right", fontsize=9, color="#555555")
    ax.set_xlabel("Average channel noise (%)"); ax.set_ylabel("Eve's information gain with drift\nminus gain with steady noise (% points)")
    ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1, 0.08)); tidy(ax)
    fig.tight_layout(); fig.savefig("figures/fig4.png", dpi=200); plt.close(fig)
print("done")
