# BrainPy Learning Notes

A small computational-neuroscience learning repository for experimenting with BrainPy-style modeling workflows in Jupyter notebooks.

## What it demonstrates

- Interest in neuroscience, dynamical systems, and brain-inspired computation
- Notebook-based exploration of Python scientific-computing workflows
- Early-stage research/project organization for computational neuroscience learning

## Contents

- `examples.ipynb` — exploratory notebook for BrainPy/modeling examples

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install jupyter brainpy numpy pandas matplotlib
jupyter notebook examples.ipynb
```

## Current limitations

This repo is currently a learning sandbox rather than a polished research package. Some notebook cells may depend on local helper files or older experimental code. The next portfolio upgrade should clean the notebook, remove stale outputs, and add a minimal reproducible BrainPy example.

## Next improvements

- Add one clean notebook that runs from a fresh environment
- Document the model being simulated and the neuroscience question behind it
- Add plots/results screenshots to the README
- Add `requirements.txt` for reproducibility
