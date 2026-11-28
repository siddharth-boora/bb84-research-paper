# Does Drifting Channel Noise Help an Eavesdropper Hide in the BB84 Protocol?

Code and data for the paper [bb84-research-paper.pdf](paper/bb84-research-paper.pdf).

## Requirements

Python 3 and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Reproducing the results

Run from the main folder:

```bash
python code/validate.py > results/validation.txt
python code/steady_noise.py > results/steady_noise.txt
python code/drift_vs_steady.py
python code/crossover.py
python code/make_figures.py
```

`crossover.py` can take 10-30 minutes depending on the computer. The other scripts take a few minutes or less.

## Repository structure

- `code/` – simulation code
- `results/` – output data
- `figures/` – figures used in the research paper
- `paper/` – the research paper

## Acknowledgments

The drifting-noise model is from Mordarski et al., arXiv:2610.01792 (2026).
