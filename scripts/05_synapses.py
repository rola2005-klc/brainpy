"""Day 4a — synapse models: gating kinetics, receptors, and synaptic output onto a LIF neuron.

Saves to outputs/:
  05_synapses_kinetics.png   exponential, dual-exponential and alpha responses to one spike and to a
                             100 Hz train, next to the bp.dyn built-ins with their default settings
  05_synapses_receptors.png  AMPA / GABA_A two-state kinetics, NMDA gating, the Mg2+ block and I-V curves
  05_synapses_coba_cuba.png  PSPs at different holding potentials: COBA vs CUBA, GABA_A reversal, NMDA vs AMPA

Runs in float64 with dt = 0.1 ms.

Run:
    python scripts/05_synapses.py
"""

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

from neuromodels.synapses import (
    AMPA, GABAA, NMDA, Alpha, DualExponential, Exponential, SynapticLIF, coba_current, dual_exp_peak_time,
    markov_pulse_response, mg_block, nmda_current, psp_amplitude, regular_times, spike_input,
)
from neuromodels.utils import COLORS, INK, PLOT_STYLE, run, save_figure

DT = 0.1
T_SPIKE = 10.0  # ms; every single-spike experiment uses this spike time
KERNELS = [  # label, constructor, BrainPy built-in with default settings
    ("exponential, τ = 5 ms", lambda: Exponential(1, tau=5.0), lambda: bp.dyn.Expon(1, tau=5.0)),
    ("dual exp., τr = 1, τd = 5 ms", lambda: DualExponential(1, 1.0, 5.0),
     lambda: bp.dyn.DualExpon(1, tau_rise=1.0, tau_decay=5.0)),
    ("alpha, τ = 3 ms", lambda: Alpha(1, tau=3.0), lambda: bp.dyn.Alpha(1, tau_decay=3.0)),
]
AMPA_C, CUBA_C, GABA_C, NMDA_C = COLORS[0], COLORS[1], COLORS[2], COLORS[3]


def gating(model, times, duration):
    out = run(model, spike_input(times, duration, DT), DT, monitors=["g"])
    return out["ts"], out["g"][:, 0]


def kinetics():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
    for (label, make, make_builtin), color in zip(KERNELS, COLORS):
        ts, g = gating(make(), [T_SPIKE], 40.0)
        _, g_builtin = gating(make_builtin(), [T_SPIKE], 40.0)
        axes[0].plot(ts - T_SPIKE, g, color=color, label=label)
        axes[1].plot(ts - T_SPIKE, g, color=color)
        axes[1].plot(ts - T_SPIKE, g_builtin, color=color, linestyle="--")
        print(f"{label:30s} peak {g.max():.4f} at {ts[np.argmax(g)] - T_SPIKE:.1f} ms; "
              f"bp.dyn default peak {g_builtin.max():.4f}")
    for t_peak in (dual_exp_peak_time(1.0, 5.0), 3.0):
        axes[0].plot(t_peak, 1.0, "o", color=INK["primary"], markersize=3.5)
    axes[0].text(4.0, 1.04, "closed-form peak times", color=INK["secondary"], fontsize=8)
    axes[0].set(xlabel="time after spike (ms)", ylabel="gating g", xlim=(-2, 30), ylim=(0, 1.12),
                title="One spike: unit-peak kernels")
    axes[0].legend(loc="center right")
    axes[1].plot([], [], color=INK["secondary"], label="neuromodels")
    axes[1].plot([], [], color=INK["secondary"], linestyle="--", label="bp.dyn built-in, defaults")
    axes[1].text(4.8, 1.04, "DualExpon, exp_auto: peak +5 %", color=COLORS[1], fontsize=8)
    axes[1].text(11.0, 0.40, "Alpha: peak 1/e", color=COLORS[2], fontsize=8)
    axes[1].text(15.0, 0.22, "Expon: identical", color=COLORS[0], fontsize=8)
    axes[1].set(xlabel="time after spike (ms)", xlim=(-2, 30), ylim=(0, 1.12),
                title="Same equations, different conventions")
    axes[1].legend(loc="center right")

    times = regular_times(100.0, 8, T_SPIKE)
    for (label, make, _), color in zip(KERNELS, COLORS):
        ts, g = gating(make(), times, 120.0)
        axes[2].plot(ts, g, color=color, label=label)
    q = np.exp(-10.0 / 5.0)
    envelope = (1 - q ** np.arange(1, 9)) / (1 - q)
    axes[2].plot(times, envelope, "o", color=INK["primary"], markersize=3,
                 label="exp.: (1 - qⁿ)/(1 - q)")
    axes[2].plot(times, np.full_like(times, -0.08), "|", color=INK["muted"], markersize=8)
    axes[2].set(xlabel="time (ms)", ylabel="gating g", xlim=(0, 120), ylim=(-0.15, 1.8),
                title="100 Hz train: temporal summation")
    axes[2].legend(loc="upper right")
    print(f"100 Hz exponential steady peak {envelope[-1]:.4f} (1/(1 - e^-2) = {1 / (1 - q):.4f})")
    return save_figure(fig, "05_synapses_kinetics.png")


def receptors():
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.6), constrained_layout=True)
    ax = axes[0, 0]
    ax.axvspan(0.0, 1.0, color=INK["muted"], alpha=0.15, linewidth=0)
    ax.annotate("1 ms transmitter pulse", xy=(1.0, 0.99), xytext=(5.0, 0.93), color=INK["secondary"], fontsize=8,
                arrowprops={"arrowstyle": "-", "color": INK["muted"], "linewidth": 0.8})
    for model, color, name in [(AMPA(1), AMPA_C, "AMPA"), (GABAA(1), GABA_C, "GABA_A")]:
        ts, g = gating(model, [T_SPIKE], 50.0)
        exact = markov_pulse_response(ts - T_SPIKE, float(model.alpha), float(model.beta), model.T_max, model.T_dur)
        ax.plot(ts - T_SPIKE, g, color=color, label=f"{name}: α = {float(model.alpha):g}, β = {float(model.beta):g}")
        ax.plot(ts - T_SPIKE, np.where(ts >= T_SPIKE, exact, 0.0), color=INK["primary"], linestyle=":",
                linewidth=1.0)
        print(f"{name}: peak open fraction {g.max():.3f}, decay τ = 1/β = {1 / float(model.beta):.2f} ms, "
              f"max |sim - closed form| = {np.abs(g - np.where(ts >= T_SPIKE, exact, 0.0)).max():.1e}")
    ax.plot([], [], color=INK["primary"], linestyle=":", linewidth=1.0, label="closed form")
    ax.set(xlabel="time after spike (ms)", ylabel="open fraction g", xlim=(-2, 35), ylim=(0, 1.05),
           title="Two-state receptor kinetics")
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    times = regular_times(100.0, 5, T_SPIKE)
    ts, single = gating(NMDA(1), [T_SPIKE], 600.0)
    _, train = gating(NMDA(1), times, 600.0)
    linear = sum(np.concatenate([np.zeros(k), single[:len(single) - k]])
                 for k in np.rint((times - T_SPIKE) / DT).astype(int))
    ax.plot(ts, linear, color=INK["muted"], linestyle=":", label="5 × single, superposed")
    ax.plot(ts, train, color=NMDA_C, label="5 spikes at 100 Hz")
    ax.plot(ts, single, color=NMDA_C, linestyle="--", label="1 spike")
    ax.set(xlabel="time (ms)", ylabel="NMDA gating g", xlim=(0, 500), title="NMDA: slow and saturating")
    ax.legend(loc="upper right")
    print(f"NMDA: single-spike peak {single.max():.3f}; 5 spikes at 100 Hz peak {train.max():.3f} "
          f"vs {linear.max():.3f} if summed linearly")

    V = np.linspace(-100.0, 40.0, 281)
    v_half = np.log(1.2 / 3.57) / 0.062
    ax = axes[1, 0]
    ax.plot(V, np.ones_like(V), color=INK["muted"], linestyle="--", label="[Mg²⁺] = 0")
    ax.plot(V, mg_block(V, 1.2), color=NMDA_C, label="[Mg²⁺] = 1.2 mM")
    ax.axvline(v_half, color=INK["muted"], linewidth=0.8, linestyle=":")
    ax.text(v_half + 2, 0.08, f"half block at {v_half:.1f} mV", color=INK["secondary"], fontsize=8)
    ax.set(xlabel="V (mV)", ylabel="unblocked fraction B(V)", ylim=(0, 1.05),
           title="Mg²⁺ block (Jahr & Stevens 1990)")
    ax.legend(loc="center left")

    ax = axes[1, 1]
    iv = nmda_current(1.0, V)
    v_top = V[np.argmax(iv)]
    ax.axhline(0.0, color=INK["axis"], linewidth=0.8)
    ax.axvspan(V[0], v_top, color=NMDA_C, alpha=0.08, linewidth=0)
    ax.text(-97, 20, "negative-slope\nregion", color=INK["secondary"], fontsize=8)
    ax.plot(V, coba_current(1.0, V, E=0.0), color=AMPA_C, label="AMPA, or NMDA without Mg²⁺")
    ax.plot(V, iv, color=NMDA_C, label="NMDA, 1.2 mM Mg²⁺")
    ax.set(xlabel="V (mV)", ylabel="I_syn at g = 1 (mV, > 0 depolarizes)", ylim=(-45, 105),
           title="Synaptic I–V curves")
    ax.legend(loc="upper right")
    print(f"Mg block: B(-65 mV) = {mg_block(-65.0):.3f}, B(0 mV) = {mg_block(0.0):.3f}, half block at "
          f"{v_half:.1f} mV; NMDA current peaks at {v_top:.1f} mV")
    return save_figure(fig, "05_synapses_receptors.png")


def holding(make_synapse, output, weight, V0, duration=150.0):
    """PSP traces of neurons held at ``V0`` (one per neuron) after one spike at ``T_SPIKE``."""
    lif = SynapticLIF(make_synapse(len(V0)), output, weight, V_th=100.0, I_ext=np.asarray(V0) + 65.0)
    out = run(lif, spike_input([T_SPIKE], duration, DT), DT, monitors=["V"])
    return out["ts"] - T_SPIKE, out["V"] - np.asarray(V0)


def coba_versus_cuba():
    gbar, J = 0.05, 0.05 * 65.0  # CUBA weight matched to COBA at -65 mV: J = gbar (E - V)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
    V0 = np.array([-80.0, -65.0, -50.0, -35.0])
    ts, coba = holding(AMPA, "coba", gbar, V0, 60.0)
    _, cuba = holding(AMPA, "cuba", J, V0, 60.0)
    for k, v in enumerate(V0):
        axes[0].plot(ts, coba[:, k], color=AMPA_C)
        axes[0].text(ts[np.argmax(coba[:, k])] + 1.5, coba[:, k].max() + 0.012, f"{v:.0f} mV", color=AMPA_C,
                     fontsize=8, va="bottom")
    axes[0].plot(ts, cuba[:, 0], color=CUBA_C, linestyle="--", label="CUBA, every holding V")
    axes[0].plot([], [], color=AMPA_C, label="COBA (AMPA), holding V labelled")
    axes[0].set(xlabel="time after spike (ms)", ylabel="V - V_hold (mV)", xlim=(-2, 40), ylim=(-0.02, 0.6),
                title="EPSPs at four holding potentials")
    axes[0].legend(loc="upper right")
    ratio = psp_amplitude(coba, 0.0) / (0.0 - V0)
    print(f"COBA EPSP (mV) at {V0} mV: {np.round(psp_amplitude(coba, 0.0), 4)}; "
          f"EPSP/(E - V_hold) = {np.round(ratio, 6)}")
    print(f"CUBA EPSP (mV) at the same potentials: {np.round(psp_amplitude(cuba, 0.0), 4)}")

    grid = np.arange(-95.0, -4.0, 5.0)
    curves = [("COBA AMPA", AMPA, "coba", gbar, AMPA_C), ("CUBA", AMPA, "cuba", J, CUBA_C),
              ("COBA GABA_A", GABAA, "coba", 0.03, GABA_C), ("NMDA + Mg²⁺", NMDA, "nmda", gbar, NMDA_C)]
    ax = axes[1]
    ax.axhline(0.0, color=INK["axis"], linewidth=0.8)
    ax.axvline(GABAA.E, color=INK["muted"], linewidth=0.8, linestyle=":")
    ax.text(GABAA.E + 1.5, -0.45, "E_GABA", color=INK["secondary"], fontsize=8)
    for label, make, output, weight, color in curves:
        _, deviation = holding(make, output, weight, grid, 300.0)
        amplitude = psp_amplitude(deviation, 0.0)
        ax.plot(grid, amplitude, "o-", color=color, markersize=3, label=label)
        if output == "nmda":
            print(f"NMDA EPSP (mV) at -80/-60/-40/-20 mV: "
                  f"{np.round(np.interp([-80, -60, -40, -20], grid, amplitude), 3)}")
    ax.set(xlabel="holding potential (mV)", ylabel="PSP peak (mV)", ylim=(-0.5, 0.75),
           title="Driving force and Mg²⁺ block")
    ax.legend(loc="upper right")

    ax = axes[2]
    for make, output, color, name in [(AMPA, "coba", AMPA_C, "AMPA"), (NMDA, "nmda", NMDA_C, "NMDA")]:
        ts, deviation = holding(make, output, gbar, np.array([-80.0, -40.0]), 300.0)
        for k, style in enumerate(["--", "-"]):
            ax.plot(ts, deviation[:, k], color=color, linestyle=style)
    ax.plot([], [], color=INK["secondary"], linestyle="--", label="held at -80 mV")
    ax.plot([], [], color=INK["secondary"], label="held at -40 mV")
    ax.text(150, 0.19, "NMDA", color=NMDA_C, fontsize=8)
    ax.text(8, 0.46, "AMPA", color=AMPA_C, fontsize=8)
    ax.set(xlabel="time after spike (ms)", ylabel="V - V_hold (mV)", xlim=(-5, 280),
           title="NMDA grows with depolarization")
    ax.legend(loc="upper right")
    return save_figure(fig, "05_synapses_coba_cuba.png")


def main():
    bm.enable_x64()
    plt.rcParams.update(PLOT_STYLE)
    for figure in (kinetics, receptors, coba_versus_cuba):
        path = figure()
        print(f"Saved {path.relative_to(path.parents[1])}")


if __name__ == "__main__":
    main()
