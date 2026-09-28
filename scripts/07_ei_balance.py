"""E/I balanced COBA network: irregular, asynchronous firing made by the network.

Simulates the 4000-neuron COBA benchmark network (Brette et al. 2007, after
Vogels & Abbott 2005) for 1.2 s under a constant 20 mV drive, then the same
network with inhibition removed. Statistics skip the first 200 ms. Runs in
float64, like the tests, so the numbers are comparable.

Saves
    outputs/07_ei_balance_raster.png      rasters and population rates, with and without inhibition
    outputs/07_ei_balance_currents.png    excitatory, inhibitory and net input to one neuron, and its V
    outputs/07_ei_balance_statistics.png  distributions of ISI CV and firing rate

Run:
    python scripts/07_ei_balance.py
"""

import time

import brainpy.math as bm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neuromodels.networks.ei_balance import (E_EXC, E_INH, TAU_REF, V_TH, EINet, isi_cvs, network_statistics,
                                             population_rate, simulate, synaptic_currents)
from neuromodels.utils import COLORS, INK, PLOT_STYLE, n_steps, save_figure

DT, DURATION, TRANSIENT, DRIVE = 0.1, 1200.0, 200.0, 20.0
WINDOW = (900.0, 1100.0)  # ms shown in the raster and current plots
SHOW_E, SHOW_I = 320, 80  # neurons drawn per population in the rasters (10 %)


def raster_panels(axes, out, title, legend=True):
    """Raster of a neuron sample (top) and smoothed population rates (bottom)."""
    ts = out["ts"]
    shown = (ts > WINDOW[0]) & (ts <= WINDOW[1])
    for key, n_show, offset, color, label in (("spike_E", SHOW_E, 0, COLORS[0], "E"),
                                              ("spike_I", SHOW_I, SHOW_E, COLORS[1], "I")):
        steps, cells = np.nonzero(out[key][shown][:, :n_show])
        axes[0].scatter(ts[shown][steps], cells + offset, s=1.0, marker=".", linewidths=0, color=color,
                        rasterized=True)
        rate = population_rate(out[key], DT, width=2.0)
        axes[1].plot(ts[shown], rate[shown], color=color, lw=1.0, label=label)
    axes[0].set_title(title)
    axes[0].set_ylabel("neuron")
    axes[0].set_ylim(-5, SHOW_E + SHOW_I + 5)
    axes[1].set_ylabel("population rate (Hz)")
    axes[1].set_xlabel("time (ms)")
    if legend:
        axes[1].legend(loc="upper right")


def main():
    bm.set_platform("cpu")
    bm.enable_x64()
    plt.rcParams.update(PLOT_STYLE)
    start = time.perf_counter()

    balanced = simulate(EINet(seed=0), DURATION, DT, DRIVE, n_record=20)
    no_inhibition = simulate(EINet(w_inh=0.0, seed=0), DURATION, DT, DRIVE, n_record=20)
    stats = network_statistics(balanced, DT, TRANSIENT)
    stats_off = network_statistics(no_inhibition, DT, TRANSIENT)

    # Rasters and population rates.
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    raster_panels(axes[:, 0], balanced, "Balanced (w_I = 6.7 g_L)")
    raster_panels(axes[:, 1], no_inhibition, "Inhibition removed (w_I = 0)", legend=False)
    fig.tight_layout()
    raster_path = save_figure(fig, "07_ei_balance_raster.png")

    # Inputs to one excitatory neuron: the recorded neuron whose rate is closest to the mean.
    skip = n_steps(TRANSIENT, DT)
    ts = balanced["ts"]
    rec_rates = balanced["spike_E"][skip:, :20].mean(axis=0) / DT * 1e3
    cell = int(np.argmin(np.abs(rec_rates - stats["rate_E"])))
    V, g_E, g_I = (balanced[key][:, cell] for key in ("V", "g_E", "g_I"))
    I_E, I_I = synaptic_currents(V, g_E, g_I)
    exc = I_E + DRIVE
    shown = (ts > WINDOW[0]) & (ts <= WINDOW[1])
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    axes[0].plot(ts[shown], exc[shown], color=COLORS[0], lw=1.0, label="excitatory (drive + recurrent)")
    axes[0].plot(ts[shown], I_I[shown], color=COLORS[1], lw=1.0, label="inhibitory")
    axes[0].plot(ts[shown], (exc + I_I)[shown], color=COLORS[2], lw=1.0, label="net")
    axes[0].axhline(0.0, color=INK["muted"], lw=0.8)
    axes[0].set_ylabel("input R·I (mV)")
    means = [exc[skip:].mean(), I_I[skip:].mean(), (exc + I_I)[skip:].mean()]
    axes[0].set_title(f"Inputs to one E neuron ({rec_rates[cell]:.0f} Hz): "
                      "mean {:.0f} + ({:.0f}) = {:.0f} mV".format(*means))
    axes[0].legend(loc="upper right", ncol=3)
    axes[1].plot(ts[shown], V[shown], color=COLORS[0], lw=1.0)
    spikes = ts[shown][balanced["spike_E"][shown, cell]]
    axes[1].vlines(spikes, V_TH, V_TH + 8.0, color=COLORS[0], lw=1.0)
    axes[1].axhline(V_TH, color=INK["muted"], lw=0.8, ls="--")
    axes[1].text(WINDOW[1], V_TH + 0.5, "threshold", color=INK["secondary"], fontsize=8, ha="right", va="bottom")
    axes[1].set_ylabel("V (mV)")
    axes[1].set_xlabel("time (ms)")
    fig.tight_layout()
    currents_path = save_figure(fig, "07_ei_balance_currents.png")

    # ISI CV and rate distributions.
    spikes_E, spikes_I = balanced["spike_E"][skip:], balanced["spike_I"][skip:]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for spikes, color, label in ((spikes_E, COLORS[0], "E"), (spikes_I, COLORS[1], "I")):
        cvs = isi_cvs(spikes, ts[skip:])
        axes[0].hist(cvs[np.isfinite(cvs)], bins=np.linspace(0, 4, 41), density=True, histtype="step",
                     color=color, lw=1.5, label=label)
        rates = spikes.sum(axis=0) / (spikes.shape[0] * DT) * 1e3
        # Rates over a 1 s window are whole numbers of Hz, so bins are 2 Hz wide.
        axes[1].hist(rates, bins=np.arange(0, 162, 2), histtype="step", color=color, lw=1.5, label=label,
                     weights=np.full(rates.size, 1 / rates.size))
    axes[0].axvline(1.0, color=INK["muted"], lw=0.8, ls="--")
    axes[0].text(1.05, axes[0].get_ylim()[1] * 0.92, "Poisson", color=INK["secondary"], fontsize=8)
    axes[0].set_xlabel("ISI coefficient of variation")
    axes[0].set_ylabel("density")
    axes[0].set_title(f"Irregular firing: mean CV = {stats['cv_mean']:.2f}")
    rates_E = spikes_E.sum(axis=0) / (spikes_E.shape[0] * DT) * 1e3
    axes[1].set_xlabel("firing rate (Hz)")
    axes[1].set_ylabel("fraction of neurons")
    axes[1].set_title(f"Broad rates: mean {stats['rate_E']:.1f} Hz, {np.mean(rates_E < 1):.0%} of E below 1 Hz")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    stats_path = save_figure(fig, "07_ei_balance_statistics.png")

    print(f"Balanced network (4000 LIF neurons, {DURATION:.0f} ms, statistics after {TRANSIENT:.0f} ms):")
    print(f"  mean rate E / I:           {stats['rate_E']:.1f} / {stats['rate_I']:.1f} Hz")
    print(f"  mean ISI CV:               {stats['cv_mean']:.2f} ({stats['cv_counted']} neurons with >= 5 spikes)")
    print(f"  pairwise count correlation (10 ms bins): {stats['correlation']:.4f}")
    print(f"  mean inputs (recorded E):  excitation {stats['I_exc']:.0f} mV, inhibition {stats['I_inh']:.0f} mV, "
          f"net {stats['I_net']:.0f} mV (|net| / excitation = {stats['net_to_exc']:.2f})")
    print("Inhibition removed:")
    print(f"  mean rate E / I:           {stats_off['rate_E']:.1f} / {stats_off['rate_I']:.1f} Hz "
          f"(refractory limit {1e3 / TAU_REF:.0f} Hz)")
    print(f"  mean ISI CV:               {stats_off['cv_mean']:.3f}")
    print(f"  pairwise count correlation: {stats_off['correlation']:.3f}")
    print(f"  mean excitatory input:     {stats_off['I_exc']:.0f} mV")
    print(f"Reversal potentials E_E = {E_EXC:.0f} mV, E_I = {E_INH:.0f} mV")
    for path in (raster_path, currents_path, stats_path):
        print(f"Saved {path}")
    print(f"Finished in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
