# Does Drifting Channel Noise Help an Eavesdropper Hide in the BB84 Protocol?

Research Paper: **[paper/paper.pdf](paper/paper.pdf)**

## The question

In BB84, Alice and Bob check for an eavesdropper (Eve) by measuring the error rate of a sample of
their bits and stopping if it is above 11%. Eve can hide by attacking only some qubits so their errors
blend in with the channel's normal noise. Real channel noise drifts up and down over time.
**Does drifting noise make it easier or harder for Eve to avoid being caught, compared with steady
noise of the same average?**

## What I found

- Even with steady noise, the textbook limit `f_max = 4(0.11 - e)/(1 - 2e)` only lets Eve pass the
  check about half the time (Figure 2).
- On cleaner channels, drift **hurts** Eve: sudden noise increases push their rounds over 11%.
- On noisier channels, drift **helps** them: they attack more when the noise drops (Figure 3).
- The switch happens at an average noise of about **6.1%, 6.9% and 7.4%** for 1024, 2048 and 4096
  qubits per round (5 runs each, 95% confidence intervals about ±0.1–0.15%) (Figure 4).

## Files

| Paper | Code (`code/`) | Output |
|---|---|---|
| Section 2.6 (checks) | `validate.py` | `results/validation.txt` |
| Figure 2, Table 1 | `steady_noise.py` | `figures/fig2.png`, `results/steady_thresholds.csv` |
| Figure 3 | `drift_vs_steady.py` | `results/drift_vs_steady.csv` |
| Figure 4, Table 2 | `crossover.py` | `results/crossover_runs.csv`, `results/crossover_summary.txt` |
| Figures 1, 3, 4 | `make_figures.py` | `figures/` |

Shared code: `bb84.py` (one BB84 round, qubit by qubit, plus the QBER formula), `evasion.py`
(exact chance Eve passes the check), `drift.py` (drifting noise and Eve's strategies).

## Run it

```bash
pip install -r requirements.txt
python code/validate.py > results/validation.txt
python code/steady_noise.py > results/steady_noise.txt
python code/drift_vs_steady.py
python code/crossover.py        
python code/make_figures.py
```

Run the commands from the repository's top folder.

## Credits

The drifting-noise model comes from Mordarski et al., arXiv:2610.01792 (2026).
# bb84-research-paper
