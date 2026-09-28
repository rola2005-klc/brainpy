"""Shared helpers for running BrainPy models and measuring their output.

One convention matters everywhere in this repo: ``bp.DSRunner`` stamps each
monitored sample with the *start* time of its integration step, but the value
it stores is the state at the *end* of that step. :func:`run` returns
end-of-step times, so ``(ts, V)`` pairs line up with the true trajectory
(``tests/test_utils.py`` checks this against the exact LIF solution).
"""

from pathlib import Path
from typing import Dict, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"

# Series colors in a fixed order, checked for color-vision-deficiency separation between
# neighbours. Assign them in this order and never cycle past the eighth; slots 3–5 are
# low-contrast on white, so series drawn in them get a legend or a direct label.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7"}

# Shared figure style for every script: plt.rcParams.update(PLOT_STYLE)
PLOT_STYLE = {
    "axes.prop_cycle": f"cycler(color={COLORS})",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK["axis"],
    "axes.labelcolor": INK["secondary"],
    "axes.titlesize": 10,
    "axes.titlecolor": INK["primary"],
    "axes.labelsize": 9,
    "grid.color": INK["grid"],
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.5,
    "text.color": INK["primary"],
    "xtick.color": INK["axis"],
    "ytick.color": INK["axis"],
    "xtick.labelcolor": INK["secondary"],
    "ytick.labelcolor": INK["secondary"],
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 100,
    "savefig.facecolor": "white",
}


def n_steps(duration: float, dt: float) -> int:
    """Number of integration steps in ``duration`` (robust to float rounding)."""
    return int(round(duration / dt))


def step_current(duration: float, dt: float, amplitude, onset: float = 0.0, offset: float = None) -> np.ndarray:
    """Piecewise-constant input: ``amplitude`` during ``[onset, offset)``, zero elsewhere.

    ``amplitude`` may be a scalar or a 1-D array with one value per neuron, giving
    an array of shape ``(n_steps,)`` or ``(n_steps, n_neurons)``. The value at step
    ``i`` drives the update from ``t_i`` to ``t_i + dt``. Edges are computed on
    step indices, so ``onset=0.3`` with ``dt=0.1`` never lands one step late.
    """
    total = n_steps(duration, dt)
    index = np.arange(total)
    start = n_steps(onset, dt)
    stop = total if offset is None else n_steps(offset, dt)
    on = ((index >= start) & (index < stop)).astype(float)
    return np.multiply.outer(on, np.asarray(amplitude, dtype=float))


def run(model, inputs, dt: float, monitors: Sequence[str] = ("V", "spike"), reset: bool = True) -> Dict[str, np.ndarray]:
    """Run a BrainPy model step by step and return its monitors as NumPy arrays.

    ``inputs`` is an array whose first axis is time; ``inputs[i]`` is passed to
    ``model.update`` at step ``i``. The returned ``"ts"`` holds end-of-step times
    (see the module docstring). With ``reset=True`` the model starts from its
    initial state, so repeated calls give identical results.
    """
    import brainpy as bp  # imported here so the NumPy-only helpers work without JAX

    if reset:
        bp.reset_state(model)
    runner = bp.DSRunner(model, monitors=list(monitors), dt=dt, progress_bar=False)
    runner.run(inputs=np.asarray(inputs))
    out = {name: np.asarray(runner.mon[name]) for name in monitors}
    out["ts"] = np.asarray(runner.mon.ts) + dt
    return out


def spike_times(spikes: np.ndarray, ts: np.ndarray):
    """Spike times for one neuron (1-D boolean array) or a list, one per neuron (2-D)."""
    spikes = np.asarray(spikes, dtype=bool)
    if spikes.ndim == 1:
        return ts[spikes]
    return [ts[spikes[:, k]] for k in range(spikes.shape[1])]


def firing_rate(spikes: np.ndarray, dt: float, start: float = 0.0, stop: float = None) -> np.ndarray:
    """Mean firing rate in Hz (``dt`` in ms) over ``[start, stop)``, per neuron."""
    spikes = np.asarray(spikes, dtype=bool)
    first = n_steps(start, dt)
    last = spikes.shape[0] if stop is None else n_steps(stop, dt)
    return spikes[first:last].sum(axis=0) / ((last - first) * dt) * 1e3


def cv_isi(times: np.ndarray) -> float:
    """Coefficient of variation of inter-spike intervals (NaN with < 2 intervals)."""
    intervals = np.diff(np.asarray(times))
    if len(intervals) < 2:
        return float("nan")
    return float(intervals.std() / intervals.mean())


def save_figure(fig, name: str, dpi: int = 160) -> Path:
    """Save ``fig`` as ``outputs/<name>`` and return the path."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
