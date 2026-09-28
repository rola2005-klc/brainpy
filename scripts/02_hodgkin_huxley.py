"""Day 2 — the Hodgkin–Huxley model: spike anatomy, gating kinetics, excitability.

Saves to outputs/:
  02_hh_spike_anatomy.png  input, V, gating variables, and ionic currents through three spikes
  02_hh_gating.png         steady-state activation/inactivation and time constants of m, h, n
  02_hh_excitability.png   F-I curves, anode-break excitation, and rest, for Hodgkin &
                           Huxley's leak (0.3 mS/cm²) against BrainPy's default (0.03)

Runs in float64 with dt = 0.01 ms.

Run:
    python scripts/02_hodgkin_huxley.py
"""

import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

from neuromodels.analysis import fi_curve, hh_jacobian, hh_rest_state
from neuromodels.neurons import HH
from neuromodels.utils import COLORS, INK, PLOT_STYLE, n_steps, run, save_figure, step_current

DT = 0.01
SODIUM, POTASSIUM, INACTIVATION = COLORS[0], COLORS[1], COLORS[2]
LEAKS = [(0.3, "gL = 0.3 (Hodgkin & Huxley)", COLORS[0]), (0.03, "gL = 0.03 (bp.dyn.HH default)", COLORS[1])]


def spike_anatomy():
    hh = HH(1)
    inputs = step_current(45.0, DT, 10.0, onset=5.0)
    out = run(hh, inputs, DT, monitors=("V", "m", "h", "n", "spike"))
    ts, V, m, h, n = out["ts"], out["V"][:, 0], out["m"][:, 0], out["h"][:, 0], out["n"][:, 0]
    I_Na, I_K, I_L = (np.asarray(x) for x in hh.currents(V, m, h, n))

    fig, axes = plt.subplots(4, 1, figsize=(8.5, 7.2), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [0.5, 1.4, 1, 1.2]})
    axes[0].plot(ts, inputs, color=INK["secondary"])
    axes[0].set(ylabel="I (µA/cm²)", title="Hodgkin–Huxley neuron: a 10 µA/cm² step")
    axes[1].plot(ts, V, color=INK["primary"], linewidth=1.2)
    axes[1].set(ylabel="V (mV)")
    for trace, label, color in [(m, "m (Na activation)", SODIUM), (h, "h (Na inactivation)", INACTIVATION),
                                (n, "n (K activation)", POTASSIUM)]:
        axes[2].plot(ts, trace, color=color, label=label)
    axes[2].set(ylabel="gate open fraction", ylim=(0, 1))
    axes[2].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    for trace, label, color in [(I_Na, "I_Na", SODIUM), (I_K, "I_K", POTASSIUM), (I_L, "I_leak", INK["muted"])]:
        axes[3].plot(ts, trace, color=color, label=label)
    axes[3].axhline(0.0, color=INK["axis"], linewidth=0.8)
    axes[3].set(xlabel="time (ms)", ylabel="current (µA/cm²)\ninward < 0 < outward")
    axes[3].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    print(f"Spike anatomy: {out['spike'].sum()} spikes, peak I_Na {I_Na.min():.0f} µA/cm², peak I_K {I_K.max():.0f} µA/cm²")
    return save_figure(fig, "02_hh_spike_anatomy.png")


def gating():
    hh = HH(1)
    V = np.linspace(-100.0, 50.0, 601)
    steady, taus = hh.gate_kinetics(V)
    fig, (ax_inf, ax_tau) = plt.subplots(1, 2, figsize=(9, 3.4), constrained_layout=True)
    for x_inf, tau, label, color in zip(steady, taus, ["m", "h", "n"], [SODIUM, INACTIVATION, POTASSIUM]):
        ax_inf.plot(V, np.asarray(x_inf), color=color, label=label)
        ax_tau.plot(V, np.asarray(tau), color=color, label=label)
    ax_inf.axvline(-65.0, color=INK["muted"], linewidth=0.8, linestyle="--")
    ax_inf.text(-64.0, 0.92, "rest", color=INK["secondary"], fontsize=8)
    ax_inf.set(xlabel="V (mV)", ylabel="steady state x∞(V)", title="Where each gate settles")
    ax_tau.set(xlabel="V (mV)", ylabel="time constant τx(V) (ms)", title="How fast it gets there: m ≪ h, n")
    ax_inf.legend(loc="center right")
    ax_tau.legend(loc="upper right")
    return save_figure(fig, "02_hh_gating.png")


def excitability():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    currents = np.arange(0.0, 20.01, 0.1)
    pulse = step_current(120.0, DT, -10.0, onset=20.0, offset=40.0)
    for gL, label, color in LEAKS:
        rates = fi_curve(lambda size: HH(size, gL=gL), currents, duration=1000.0, dt=DT, transient=200.0)
        onset = int(np.argmax(rates > 0))
        axes[0].plot(currents, rates, ".", color=color, markersize=4, label=label)
        print(f"{label}: repetitive firing from I = {currents[onset]:.1f} µA/cm² at {rates[onset]:.0f} Hz")

        out = run(HH(1, gL=gL), pulse, DT)
        axes[1].plot(out["ts"], out["V"][:, 0], color=color, label=label)

        rest = run(HH(1, gL=gL), np.zeros(n_steps(60.0, DT)), DT)
        axes[2].plot(rest["ts"], rest["V"][:, 0], color=color, label=label)
        print(f"{label}: rest settles at {rest['V'][-1, 0]:.2f} mV")

    axes[0].set(xlabel="constant input I (µA/cm²)", ylabel="firing rate (Hz)",
                title="F-I curve jumps from 0 (class 2)")
    axes[1].axvspan(20.0, 40.0, color=INK["muted"], alpha=0.15, linewidth=0)
    axes[1].text(21.0, 30.0, "I = -10 µA/cm²", color=INK["secondary"], fontsize=8)
    axes[1].set(xlabel="time (ms)", ylabel="V (mV)", title="Anode break: release fires a spike")
    axes[2].set(xlabel="time (ms)", ylabel="V (mV)", title="Rest with no input, from -65 mV")
    for ax, loc in zip(axes, ["upper left", "lower right", "lower right"]):
        ax.legend(loc=loc)
    return save_figure(fig, "02_hh_excitability.png")


def hopf_current(low: float = 6.0, high: float = 12.0) -> float:
    """Input at which the rest state's leading eigenvalue crosses zero (bisection)."""
    def growth(current):
        return np.linalg.eigvals(hh_jacobian(hh_rest_state(current), current)).real.max()

    for _ in range(30):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if growth(middle) < 0 else (low, middle)
    return 0.5 * (low + high)


def main():
    bm.enable_x64()
    plt.rcParams.update(PLOT_STYLE)
    for path in (spike_anatomy(), gating(), excitability()):
        print(f"Saved {path.relative_to(path.parents[1])}")
    print(f"Rest stays linearly stable up to I = {hopf_current():.3f} µA/cm² (subcritical Hopf), so between the "
          f"firing onset and that point rest and repetitive firing coexist")


if __name__ == "__main__":
    main()
