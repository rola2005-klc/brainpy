"""Day 1 — BrainPy basics: integrators, their accuracy, and JIT compilation.

Saves outputs/01_integrator_accuracy.png:
  left   global error against step size on the logistic equation (orders 1, 2, 4)
  right  a fast linear decay (tau = 1 ms) stepped at dt = 1.5 and 2.5 tau: forward
         Euler oscillates and then blows up, exponential Euler stays exact

and prints the fitted convergence orders and a JIT timing comparison. Runs in
float64 so the RK4 errors sit well above rounding noise.

Run:
    python scripts/01_brainpy_basics.py
"""

import time

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

from neuromodels.basics import decay_trajectory, logistic_error
from neuromodels.neurons import LIF
from neuromodels.utils import COLORS, INK, PLOT_STYLE, save_figure, step_current

METHODS = [("euler", "forward Euler"), ("heun2", "Heun (RK2)"), ("rk4", "classical RK4")]
DTS = np.array([0.4, 0.2, 0.1, 0.05, 0.025])


def jit_timing(n_neurons: int = 1000, dt: float = 0.1) -> dict:
    """Milliseconds per step for the same LIF population with and without JIT compilation.

    The compiled run takes 2000 steps (compilation included); the interpreted run only
    100, because every step dispatches each array operation from Python separately.
    """
    per_step = {}
    for jit, steps in ((True, 2000), (False, 100)):
        runner = bp.DSRunner(LIF(n_neurons), monitors=["spike"], dt=dt, jit=jit, progress_bar=False)
        start = time.perf_counter()
        runner.run(inputs=step_current(steps * dt, dt, 25.0))
        per_step[jit] = (time.perf_counter() - start) / steps * 1e3
    return per_step


def main():
    bm.enable_x64()
    plt.rcParams.update(PLOT_STYLE)
    fig, (ax_err, ax_decay) = plt.subplots(1, 2, figsize=(11.5, 3.6), constrained_layout=True)

    for (method, label), color in zip(METHODS, COLORS):
        errors = np.array([logistic_error(method, dt) for dt in DTS])
        order = np.polyfit(np.log(DTS), np.log(errors), 1)[0]
        ax_err.loglog(DTS, errors, "o-", color=color, markersize=4, label=f"{label} (slope {order:.2f})")
        print(f"{label:>14}: fitted order {order:.2f}")
    ax_err.set_xticks(DTS, [f"{dt:g}" for dt in DTS])
    ax_err.minorticks_off()
    ax_err.set(xlabel="step size dt", ylabel="max |x_numeric - x_exact|",
               title="Logistic equation: error shrinks as dt^order")
    ax_err.legend(loc="lower right")

    t_exact = np.linspace(0.0, 12.5, 300)
    ax_decay.plot(t_exact, np.exp(-t_exact), color=INK["muted"], linewidth=3, alpha=0.6, label="exact  e^(-t/τ)")
    for dt, color in [(1.5, COLORS[0]), (2.5, COLORS[1])]:
        ts, xs = decay_trajectory("euler", dt, duration=12.5)
        ax_decay.plot(np.r_[0.0, ts], np.r_[1.0, xs], "o-", color=color, markersize=4,
                      label=f"forward Euler, dt = {dt}τ (×{1 - dt:+.1f} per step)")
    ts, xs = decay_trajectory("exp_euler", 2.5, duration=12.5)
    ax_decay.plot(np.r_[0.0, ts], np.r_[1.0, xs], "s", color=COLORS[2], markersize=5,
                  label="exponential Euler, dt = 2.5τ")
    ax_decay.axhline(0.0, color=INK["axis"], linewidth=0.8)
    ax_decay.set(xlabel="time (ms), τ = 1 ms", ylabel="x", ylim=(-4.0, 4.0),
                 title="Stiff decay: Euler is stable only for dt < 2τ")
    ax_decay.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))

    path = save_figure(fig, "01_integrator_accuracy.png")
    per_step = jit_timing()
    print(f"1000 LIF neurons: {per_step[True]:.3f} ms/step compiled, {per_step[False]:.1f} ms/step interpreted "
          f"({per_step[False] / per_step[True]:.0f}x slower)")
    print(f"Saved {path.relative_to(path.parents[1])}")


if __name__ == "__main__":
    main()
