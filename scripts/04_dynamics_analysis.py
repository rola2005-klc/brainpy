"""Day 3 — dynamics analysis: phase planes and bifurcations of the Izhikevich model.

Saves to outputs/:
  04_phase_planes.png  the RS neuron at I = 0, just below its Hopf point, and while firing:
                       nullclines, fixed points (filled = stable), flow, and a trajectory
  04_bifurcation.png   resting states and firing rate against input for a Hopf case
                       (RS: b > a) and a saddle-node case (b < a)

Also prints the fixed points BrainPy's PhasePlane2D finds next to the closed form.
Runs in float64.

Run:
    python scripts/04_dynamics_analysis.py
"""

import contextlib
import io

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

from neuromodels.analysis import izhikevich_fixed_points, izhikevich_jacobian, izhikevich_rest_loss
from neuromodels.neurons import Izhikevich
from neuromodels.utils import COLORS, INK, PLOT_STYLE, run, save_figure, step_current

DT = 0.05
RS = dict(a=0.02, b=0.2, c=-65.0, d=8.0)
SADDLE_NODE = dict(a=0.02, b=-0.1, c=-55.0, d=6.0)
KIND = {"hopf": "Hopf bifurcation", "saddle-node": "saddle-node bifurcation"}


def is_stable(V, a, b):
    return bool(np.all(np.linalg.eigvals(izhikevich_jacobian(V, a, b)).real < 0))


def brainpy_fixed_points(current, a, b):
    """Fixed points found by BrainPy's phase-plane analyzer (the course's tool)."""
    integral = bp.odeint(bp.JointEq(lambda V, t, u, I: 0.04 * V * V + 5 * V + 140 - u + I,
                                    lambda u, t, V: a * (b * V - u)), method="rk4")
    analyzer = bp.analysis.PhasePlane2D(model=integral, target_vars={"V": [-80.0, -40.0], "u": [-18.0, -8.0]},
                                        pars_update={"I": current}, resolutions={"V": 0.05, "u": 0.05})
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        analyzer.plot_nullcline(with_plot=False)
        return np.asarray(analyzer.plot_fixed_point(with_plot=False, with_return=True))


def trajectory_segments(out):
    """Split a trajectory at resets so a spike is not drawn as a line back across the plane.

    Monitors store the post-reset state at the spiking step, so each segment starts there.
    """
    V, u = out["V"][:, 0], out["u"][:, 0]
    breaks = np.flatnonzero(out["spike"][:, 0])
    return list(zip(np.split(V, breaks), np.split(u, breaks))), [(V[k], u[k]) for k in breaks]


def phase_planes():
    a, b = RS["a"], RS["b"]
    # current, start (V, u), duration, axis limits, title
    cases = [(0.0, (-58.0, -13.0), 150.0, ((-80, 35), (-20, 12)), "I = 0: a kick decays back to rest"),
             (3.7, (-61.4, -12.55), 800.0, ((-66, -55), (-13.4, -10.6)),
              "I = 3.7 (zoomed): damped spiral into a stable focus"),
             (10.0, (-70.0, -14.0), 300.0, ((-80, 35), (-20, 12)), "I = 10: no rest state; repetitive firing")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    for ax, (current, start, duration, (xlim, ylim), title) in zip(axes, cases):
        V_grid, u_grid = np.meshgrid(np.linspace(*xlim, 60), np.linspace(*ylim, 60))
        ax.streamplot(V_grid, u_grid, 0.04 * V_grid ** 2 + 5 * V_grid + 140 - u_grid + current,
                      a * (b * V_grid - u_grid), color=INK["grid"], density=0.9, linewidth=0.6, arrowsize=0.7)
        V = np.linspace(*xlim, 400)
        ax.plot(V, 0.04 * V ** 2 + 5 * V + 140 + current, color=COLORS[0], label="V-nullcline (dV/dt = 0)")
        ax.plot(V, b * V, color=COLORS[1], label="u-nullcline (du/dt = 0)")

        out = run(Izhikevich(1, V_init=start[0], u_init=start[1], **RS), step_current(duration, DT, current), DT,
                  monitors=("V", "u", "spike"))
        segments, resets = trajectory_segments(out)
        for k, (V_seg, u_seg) in enumerate(segments):
            ax.plot(V_seg, u_seg, color=INK["primary"], linewidth=1, label="trajectory" if k == 0 else None)
        for V_reset, u_reset in resets:  # the jump from the spike peak to (c, u + d)
            ax.plot([30.0, V_reset], [u_reset - RS["d"], u_reset], ":", color=INK["muted"], linewidth=0.8)
        ax.plot(*start, "o", color=INK["primary"], markersize=4)

        for V_fp, u_fp in izhikevich_fixed_points(current, b):
            stable = is_stable(V_fp, a, b)
            ax.plot(V_fp, u_fp, "o", markersize=8, markeredgecolor=INK["primary"],
                    markerfacecolor=INK["primary"] if stable else "white")
        ax.set(xlim=xlim, ylim=ylim, xlabel="V (mV)", ylabel="u", title=title)
    axes[0].legend(loc="upper left")
    return save_figure(fig, "04_phase_planes.png")


def onset_rates(pars, currents, duration=8000.0):
    """Steady firing rate from the last inter-spike intervals, each neuron starting at its own rest.

    Starting from rest matters for the Hopf case: that bifurcation is subcritical, so just
    below it rest coexists with firing and a large enough kick would start spikes early.
    """
    rests = [izhikevich_fixed_points(current, pars["b"]) for current in currents]
    V0 = np.array([points[0][0] + 0.1 if points else -60.0 for points in rests])
    out = run(Izhikevich(len(currents), V_init=V0, u_init=pars["b"] * V0, **pars),
              step_current(duration, DT, currents), DT, monitors=("spike",))
    rates = []
    for k in range(len(currents)):
        times = out["ts"][out["spike"][:, k]]
        times = times[times > duration / 2]
        rates.append(1000.0 / np.mean(np.diff(times)) if len(times) > 2 else 0.0)
    return np.array(rates)


def bifurcation():
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), constrained_layout=True)
    for col, (name, pars) in enumerate([("RS (a = 0.02 < b = 0.2)", RS), ("b = -0.1 < a = 0.02", SADDLE_NODE)]):
        loss = izhikevich_rest_loss(pars["a"], pars["b"])
        currents = np.linspace(loss.current - 3.0, loss.current + 1.0, 400)
        lower, upper = [], []
        for current in currents:
            points = izhikevich_fixed_points(current, pars["b"])
            lower.append(points[0][0] if points else np.nan)
            upper.append(points[-1][0] if points else np.nan)
        lower, upper = np.array(lower), np.array(upper)
        stable = np.array([not np.isnan(V) and is_stable(V, pars["a"], pars["b"]) for V in lower])

        ax = axes[0, col]
        ax.plot(currents[stable], lower[stable], color=COLORS[0], label="rest (stable)")
        ax.plot(currents[~stable], lower[~stable], "--", color=COLORS[0], label="rest (unstable)")
        ax.plot(currents, upper, ":", color=INK["secondary"], label="saddle (threshold)")
        ax.plot(loss.current, loss.voltage, "o", color=COLORS[1], markersize=7, label=f"{KIND[loss.kind]} at I = {loss.current:.3f}")
        ax.set(xlabel="constant input I", ylabel="fixed-point V (mV)", title=f"{name}: rest ends in a {KIND[loss.kind]}")
        ax.legend(loc="center left")

        ax = axes[1, col]
        fine = loss.current + np.array([-0.05, -0.02, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0])
        rates = onset_rates(pars, fine)
        ax.plot(fine, rates, "o-", color=COLORS[1], markersize=4)
        ax.axvline(loss.current, color=INK["muted"], linewidth=0.8, linestyle="--")
        ax.set(xlabel="constant input I (starting from rest)", ylabel="firing rate (Hz)",
               title="firing starts at a finite rate" if loss.kind == "hopf" else "firing starts from zero rate")
        print(f"{name}: {KIND[loss.kind]} at I = {loss.current:.4f}; rates at I_c + (0.002, 0.005, 0.01): "
              f"{np.round(rates[2:5], 2)} Hz")
    return save_figure(fig, "04_bifurcation.png")


def main():
    bm.enable_x64()
    plt.rcParams.update(PLOT_STYLE)
    found = brainpy_fixed_points(3.7, RS["a"], RS["b"])
    exact = izhikevich_fixed_points(3.7, RS["b"])
    for (V_bp, u_bp), (V, u) in zip(sorted(map(tuple, found)), exact):
        print(f"I = 3.7 fixed point: PhasePlane2D V = {V_bp:.4f}, closed form V = {V:.4f} ({'stable' if is_stable(V, RS['a'], RS['b']) else 'unstable'})")
    for path in (phase_planes(), bifurcation()):
        print(f"Saved {path.relative_to(path.parents[1])}")


if __name__ == "__main__":
    main()
