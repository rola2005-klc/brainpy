# BrainPy Learning Notes

A small computational-neuroscience learning repository for experimenting with BrainPy-style modeling workflows and reproducible neuron-model simulations.

## What it demonstrates

- Interest in neuroscience, dynamical systems, and brain-inspired computation
- Notebook-based exploration of Python scientific-computing workflows
- A clean, dependency-light spiking-neuron demo that runs from a fresh environment
- Early-stage research/project organization for computational neuroscience learning

## Contents

- `izhikevich_demo.py` — minimal reproducible simulation of a regular-spiking Izhikevich neuron
- `outputs/izhikevich_voltage_trace.png` — generated voltage trace from the demo
- `examples.ipynb` — exploratory notebook for BrainPy/modeling notes
- `requirements.txt` — lightweight environment for the reproducible demo

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python izhikevich_demo.py
```

Expected output:

```text
Simulated 1000 time steps
Spike count: <number of spikes>
Saved plot: outputs/izhikevich_voltage_trace.png
```

## Why this model?

The Izhikevich neuron model is a compact dynamical-systems model that can produce biologically plausible spiking patterns with low computational cost. It is useful as a first bridge between mathematical modeling, neuroscience intuition, and Python simulation workflows.

## Current limitations

This repo is still a learning artifact, not a finished research package. The notebook may contain older exploratory cells, while `izhikevich_demo.py` is the clean reproducible entry point.

## Next improvements

- Add a BrainPy-native version of the same simulation
- Compare regular-spiking, fast-spiking, and bursting parameter sets
- Add notebook explanations of the differential equations
- Add tests for deterministic simulation output
