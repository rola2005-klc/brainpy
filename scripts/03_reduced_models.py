"""Day 3 — reduced neuron models: one step current, five models, seven Izhikevich cell types.

Saves to outputs/:
  03_reduced_traces.png       LIF, QIF, ExpIF, AdEx, and Izhikevich (RS) responses to a current step
  03_fi_curves.png            firing rate against input relative to rheobase: HH jumps to
                              ~50 Hz (class 2), the integrate-and-fire models start from zero (class 1)
  03_izhikevich_patterns.png  the seven cell types of Izhikevich (2003), Fig. 2

Runs in float64.

Run:
    python scripts/03_reduced_models.py
"""

import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

from neuromodels.analysis import adex_rest_loss, expif_rheobase, fi_curve, lif_rate, qif_rheobase
from neuromodels.neurons import HH, LIF, QIF, AdEx, ExpIF, Izhikevich
from neuromodels.utils import COLORS, INK, PLOT_STYLE, run, save_figure, spike_times, step_current

DT = 0.05
# name, population factory, step amplitude and unit, color (one color per model across figures)
MODELS = [
    ("LIF", lambda n: LIF(n, t_ref=2.0), 25.0, "", COLORS[1]),
    ("QIF", lambda n: QIF(n), 6.0, "", COLORS[2]),
    ("ExpIF", lambda n: ExpIF(n), 4.0, "", COLORS[3]),
    ("AdEx", lambda n: AdEx(n), 800.0, " pA", COLORS[4]),
    ("Izhikevich RS", lambda n: Izhikevich.from_preset("RS", n), 10.0, "", COLORS[5]),
]
# preset, input protocol, description
PATTERNS = [
    ("RS", step_current(400.0, DT, 10.0, onset=50.0), "regular spiking: step to I = 10, spike-frequency adaptation"),
    ("IB", step_current(400.0, DT, 10.0, onset=50.0), "intrinsically bursting: an initial burst, then single spikes"),
    ("CH", step_current(400.0, DT, 10.0, onset=50.0), "chattering: repeated high-frequency bursts"),
    ("FS", step_current(400.0, DT, 10.0, onset=50.0), "fast spiking: high rate, almost no adaptation"),
    ("LTS", step_current(400.0, DT, 10.0, onset=50.0), "low-threshold spiking: fast onset, then adaptation"),
    ("TC", step_current(400.0, DT, -10.0, onset=50.0, offset=250.0),
     "thalamo-cortical: rebound burst when a hyperpolarizing step (I = -10) ends"),
    ("RZ", step_current(400.0, DT, 0.3, onset=50.0), "resonator: step to I = 0.3, just past its Hopf point (0.26)"),
]


def mark_spikes(ax, times, level, color):
    ax.plot(times, np.full(len(times), level), "|", color=color, markersize=6)


def reduced_traces():
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(8, 8.5), sharex=True, constrained_layout=True)
    for ax, (name, make, amplitude, unit, color) in zip(axes, MODELS):
        out = run(make(1), step_current(300.0, DT, amplitude, onset=20.0), DT)
        V = out["V"][:, 0]
        times = spike_times(out["spike"][:, 0], out["ts"])
        ax.plot(out["ts"], V, color=color)
        mark_spikes(ax, times, V.max() + 0.12 * np.ptp(V), color)
        ax.set_title(f"{name}: step to I = {amplitude:g}{unit} at 20 ms ({len(times)} spikes)", loc="left")
        ax.set_ylabel("V (mV)")
    axes[-1].set_xlabel("time (ms)")
    return save_figure(fig, "03_reduced_traces.png")


def fi_curves():
    relative = np.linspace(0.0, 3.0, 121)
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)

    # HH's onset has no closed form (it is bistable near onset), so find it numerically.
    coarse = np.arange(5.0, 8.0, 0.05)
    hh_onset = coarse[np.argmax(fi_curve(lambda n: HH(n), coarse, duration=600.0, dt=0.01, transient=200.0) > 0)]
    curves = [("HH", lambda n: HH(n), hh_onset, 0.01, COLORS[0]),
              ("LIF", MODELS[0][1], 20.0, DT, COLORS[1]),
              ("QIF", MODELS[1][1], qif_rheobase(), DT, COLORS[2]),
              ("ExpIF", MODELS[2][1], expif_rheobase(), DT, COLORS[3])]
    for name, make, rheobase, dt, color in curves:
        rates = fi_curve(make, relative * rheobase, duration=3000.0, dt=dt, transient=500.0)
        ax.plot(relative, rates, ".", color=color, markersize=4, label=f"{name} (rheobase {rheobase:.2f})")
        first = int(np.argmax(rates > 0))
        print(f"{name:>5}: rheobase {rheobase:.3f}, first rate above it {rates[first]:.1f} Hz at {relative[first]:.3f} x rheobase")
    ax.plot(relative, lif_rate(relative * 20.0, t_ref=2.0), color=COLORS[1], linewidth=1, label="LIF closed form")
    ax.axvline(1.0, color=INK["muted"], linewidth=0.8, linestyle="--")
    ax.set(xlabel="input / rheobase", ylabel="steady firing rate (Hz)",
           title="Class 2 (HH) jumps to a finite rate; class 1 starts from zero")
    ax.legend(loc="upper left")
    return save_figure(fig, "03_fi_curves.png")


def izhikevich_patterns():
    fig, axes = plt.subplots(len(PATTERNS), 1, figsize=(8, 10), sharex=True, constrained_layout=True)
    for ax, (name, inputs, description) in zip(axes, PATTERNS):
        out = run(Izhikevich.from_preset(name), inputs, DT)
        on = inputs != 0
        ax.fill_between(out["ts"], -90, 40, where=on, color=INK["muted"], alpha=0.12, linewidth=0)
        ax.plot(out["ts"], out["V"][:, 0], color=COLORS[5], linewidth=1)
        ax.set_title(f"{name} — {description}", loc="left")
        ax.set(ylim=(-90, 40), ylabel="V (mV)")
    axes[-1].set_xlabel("time (ms); shading marks the input")
    return save_figure(fig, "03_izhikevich_patterns.png")


def main():
    bm.enable_x64()
    plt.rcParams.update(PLOT_STYLE)
    for path in (reduced_traces(), fi_curves(), izhikevich_patterns()):
        print(f"Saved {path.relative_to(path.parents[1])}")
    print(f"AdEx (Brette & Gerstner 2005) rheobase: {adex_rest_loss().current:.1f} pA")


if __name__ == "__main__":
    main()
