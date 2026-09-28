"""Day 4b — synaptic plasticity: short-term dynamics, STDP, and the Oja and BCM rules.

Saves to outputs/:
  06_plasticity_stp.png      depressing vs facilitating Tsodyks-Markram synapses: 20 Hz trains, u and x,
                             and the steady state against input rate (simulation vs closed form)
  06_plasticity_stdp.png     STDP windows measured by pairing, and a Song, Miller & Abbott (2000) neuron whose
                             1000 plastic inputs split into strong and weak while its output rate settles
  06_plasticity_hebbian.png  Oja's rule finds the principal eigenvector; BCM becomes selective to one pattern

Runs in float64. Spiking models use dt = 0.1 ms, except the 200 s STDP neuron (dt = 0.25 ms); the
rate-based rules take one input vector per step (dt = 1).

Run:
    python scripts/06_plasticity.py
"""

import time

import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

from neuromodels.plasticity import (
    BI_POO_TAUS, DEPRESSING, FACILITATING, BCMNeuron, OjaNeuron, STDPNeuron, STPSynapse, TsodyksMarkram,
    pattern_sequence, stdp_pairing, stdp_window, stp_steady_state,
)
from neuromodels.synapses import regular_times, spike_input
from neuromodels.utils import COLORS, INK, PLOT_STYLE, run, save_figure

DT = 0.1
SYNAPSES = [("depressing", DEPRESSING, COLORS[0]), ("facilitating", FACILITATING, COLORS[1])]


def short_term():
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4), constrained_layout=True)
    times = np.append(regular_times(20.0, 8, 20.0), 20.0 + 7 * 50.0 + 500.0)  # 8 spikes, recovery test at +500 ms
    inputs = spike_input(times, 1000.0, DT)
    for name, params, color in SYNAPSES:
        out = run(STPSynapse(1, tau_syn=5.0, **params), inputs, DT, monitors=["syn.g", "stp.u", "stp.x", "stp.release"])
        ts = out["ts"]
        release = out["stp.release"][inputs > 0, 0]
        label = f"{name}: U = {params['U']:g}, τd = {params['tau_d']:g}, τf = {params['tau_f']:g} ms"
        axes[0, 0].plot(ts, out["syn.g"][:, 0], color=color, label=label)
        axes[0, 1].plot(ts, out["stp.u"][:, 0], color=color)
        axes[0, 1].plot(ts, out["stp.x"][:, 0], color=color, linestyle="--")
        print(f"{name}: release per spike / U = {np.round(release / params['U'], 2)} (last = recovery test)")
    axes[0, 0].plot(times, np.full_like(times, -0.03), "|", color=INK["muted"], markersize=8)
    axes[0, 0].set(xlabel="time (ms)", ylabel="PSC (jump = release u⁺x⁻)", ylim=(-0.06, 0.62),
                   title="20 Hz train, then a spike 500 ms later")
    axes[0, 0].legend(loc="upper right")
    axes[0, 1].plot([], [], color=INK["secondary"], label="u (utilization)")
    axes[0, 1].plot([], [], color=INK["secondary"], linestyle="--", label="x (available resources)")
    axes[0, 1].set(xlabel="time (ms)", ylabel="fraction", ylim=(0, 1.05), title="What changes at the synapse")
    axes[0, 1].legend(loc=(0.52, 0.42))

    rates = np.array([2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    n_spikes = 150  # enough for the slowest (facilitating, 100 Hz) recursion to settle within 1e-4
    trains = [regular_times(rate, n_spikes, 10.0) for rate in rates]
    sweep_inputs = spike_input(trains, trains[0][-1] + 10.0, DT)
    dense = np.logspace(0, 2.2, 200)
    for name, params, color in SYNAPSES:
        release = run(TsodyksMarkram(len(rates), **params), sweep_inputs, DT, monitors=["release"])["release"]
        last = np.array([release[sweep_inputs[:, k] > 0, k][-1] for k in range(len(rates))])
        steady = np.array([stp_steady_state(rate=r, **params)[2] for r in dense])
        axes[1, 0].plot(dense, steady, color=color, label=f"{name}, closed form")
        axes[1, 0].plot(rates, last, "o", color=color, markersize=4)
        axes[1, 1].plot(dense, steady * dense, color=color)
        axes[1, 1].plot(rates, last * rates, "o", color=color, markersize=4)
        exact = np.array([stp_steady_state(rate=r, **params)[2] for r in rates])
        print(f"{name}: steady release at {rates.astype(int)} Hz = {np.round(last, 4)}; "
              f"max |sim - closed form| = {np.abs(last - exact).max():.1e}")
    axes[1, 0].plot([], [], "o", color=INK["secondary"], markersize=4, label="simulated (150th spike)")
    axes[1, 0].set(xscale="log", xlabel="input rate (Hz)", ylabel="steady release per spike u*x*",
                   title="Steady state under a regular train")
    axes[1, 0].legend(loc="upper right")
    for name, params, color in SYNAPSES:
        limit = 1000.0 / params["tau_d"]  # r* -> T/tau_d when T << tau_d, so r* f -> 1/tau_d
        axes[1, 1].axhline(limit, color=INK["muted"], linewidth=0.8, linestyle=":")
        axes[1, 1].text(1.1, limit * 1.12, f"1/τd = {limit:.2f} per s", color=INK["secondary"], fontsize=8)
    axes[1, 1].set(xscale="log", yscale="log", xlabel="input rate (Hz)", ylabel="release per second (1/s)",
                   title="Resource recovery caps the release rate")
    return save_figure(fig, "06_plasticity_stp.png")


def spike_timing():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5), constrained_layout=True)
    ax = axes[0]
    delta = np.arange(-100.0, 101.0, 5.0)
    dense = np.linspace(-100.0, 100.0, 801)
    windows = [("Bi & Poo fit: τ+ = 16.8, τ- = 33.7 ms, A- = 0.5", dict(A_plus=1.0, A_minus=0.5, **BI_POO_TAUS),
                COLORS[0]),
               ("Song et al.: τ± = 20 ms, A- = 1.05", dict(A_plus=1.0, A_minus=1.05, tau_plus=20.0, tau_minus=20.0),
                COLORS[1])]
    ax.axhline(0.0, color=INK["axis"], linewidth=0.8)
    ax.axvline(0.0, color=INK["axis"], linewidth=0.8)
    for label, params, color in windows:
        dw = stdp_pairing(delta, dt=DT, **params)
        ax.plot(dense[dense != 0], stdp_window(dense[dense != 0], **params), color=color, label=label)
        ax.plot(delta, dw, "o", color=color, markersize=3)
        err = np.abs(dw - stdp_window(delta, **params)).max()
        print(f"STDP pairing ({label}): dw(+10 ms) = {dw[delta == 10][0]:.4f}, dw(-10 ms) = {dw[delta == -10][0]:.4f}, "
              f"max |sim - window| = {err:.1e}")
    ax.plot([], [], "o", color=INK["secondary"], markersize=3, label="one simulated pair each")
    ax.set(xlabel="t_post - t_pre (ms)", ylabel="weight change / A+", ylim=(-1.2, 1.45),
           title="STDP window from pairing")
    ax.legend(loc="upper left", fontsize=7)

    dt, chunk, n_chunks, g_max = 0.25, 50000.0, 4, 0.015
    neuron = STDPNeuron(A_plus=0.01, seed=0)
    w_start = np.asarray(neuron.stdp.w.value) / g_max
    started, spikes = time.time(), []
    for k in range(n_chunks):  # chunks keep the monitors small; the state carries over (reset=False)
        spikes.append(run(neuron, np.zeros(int(chunk / dt)), dt, monitors=["spike"], reset=(k == 0))["spike"][:, 0])
    spikes = np.concatenate(spikes)
    w_end = np.asarray(neuron.stdp.w.value) / g_max
    bin_ms = 5000.0
    rate = spikes.reshape(-1, int(bin_ms / dt)).sum(axis=1) / bin_ms * 1e3
    centers = (np.arange(len(rate)) + 0.5) * bin_ms / 1e3
    axes[1].plot(centers, rate, "o-", color=COLORS[1], markersize=3)
    axes[1].set(xlabel="time (s)", ylabel="output rate (Hz)", ylim=(0, None),
                title="Output rate settles (1000 inputs at 20 Hz)")
    bins = np.linspace(0.0, 1.0, 21)
    axes[2].hist(w_end, bins=bins, color=COLORS[1], alpha=0.85, label=f"after {n_chunks * chunk / 1e3:.0f} s")
    axes[2].hist(w_start, bins=bins, histtype="step", color=INK["secondary"], linewidth=1.2, label="start (uniform)")
    axes[2].set(xlabel="peak conductance / g_max", ylabel="number of synapses", title="Weights split to the bounds")
    axes[2].legend(loc="upper center")
    print(f"STDP neuron ({time.time() - started:.0f} s wall clock): output rate {rate[0]:.1f} Hz in the first 5 s, "
          f"{rate[-4:].mean():.1f} Hz in the last 20 s; weights < 0.1 g_max: {np.mean(w_end < 0.1):.0%}, "
          f"> 0.9 g_max: {np.mean(w_end > 0.9):.0%}")
    return save_figure(fig, "06_plasticity_stdp.png")


def hebbian():
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.8), constrained_layout=True)
    rng = np.random.default_rng(0)
    angle = np.deg2rad(30.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    covariance = rotation @ np.diag([4.0, 0.5]) @ rotation.T
    samples = rng.multivariate_normal(np.zeros(2), covariance, size=5000)
    oja = OjaNeuron(2, eta=0.005, w_init=[-0.3, 0.1])
    w = run(oja, samples, 1.0, monitors=["w"])["w"]
    top = np.linalg.eigh(covariance)[1][:, -1]
    ax = axes[0, 0]
    ax.scatter(samples[:, 0], samples[:, 1], s=2, color=INK["muted"], alpha=0.3, linewidth=0)
    circle = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(circle), np.sin(circle), color=INK["secondary"], linewidth=0.8, linestyle=":")
    ax.plot([-5 * top[0], 5 * top[0]], [-5 * top[1], 5 * top[1]], color=INK["secondary"], linewidth=0.8,
            linestyle="--", label="principal eigenvector")
    ax.plot(w[:, 0], w[:, 1], color=COLORS[0], label="w during learning")
    ax.plot(*w[0], "o", color=COLORS[0], markersize=4, markerfacecolor="white")
    ax.plot(*w[-1], "o", color=COLORS[0], markersize=4)
    ax.set(xlabel="x₁", ylabel="x₂", xlim=(-5, 5), ylim=(-4, 4), aspect="equal",
           title="Oja's rule on 2-D Gaussian inputs")
    ax.legend(loc="lower right")
    cosine = np.abs(w @ top) / np.linalg.norm(w, axis=1)
    axes[0, 1].plot(cosine, color=COLORS[0], label="|cos(w, e₁)|")
    axes[0, 1].plot(np.linalg.norm(w, axis=1), color=COLORS[1], label="|w|")
    axes[0, 1].set(xlabel="input samples", ylabel="value", xscale="log", ylim=(0, 1.1), title="Converges to unit norm")
    axes[0, 1].legend(loc="lower right")
    print(f"Oja: |cos(w, e1)| = {cosine[-1]:.4f}, |w| = {np.linalg.norm(w[-1]):.4f} after {len(samples)} samples")

    patterns = np.array([[1.0, 0.3], [0.3, 1.0]])
    for ax, seed in zip(axes[1], (0, 1)):
        inputs, _ = pattern_sequence(patterns, 40000, seed=seed)
        out = run(BCMNeuron(2, seed=seed), inputs, 1.0, monitors=["w", "theta"])
        responses = np.maximum(out["w"] @ patterns.T, 0.0)
        steps = np.arange(1, len(inputs) + 1)
        ax.axhline(2.0, color=INK["muted"], linewidth=0.8, linestyle=":")
        window = 200  # theta tracks y² over ~10 presentations, so show its running mean
        theta = np.convolve(out["theta"][:, 0], np.ones(window) / window, mode="valid")
        ax.plot(steps[window - 1:], theta, color=INK["secondary"], linewidth=1.0,
                label=f"θ, mean over {window} presentations")
        ax.plot(steps, responses[:, 0], color=COLORS[2], label="response to pattern A (1, 0.3)")
        ax.plot(steps, responses[:, 1], color=COLORS[3], label="response to pattern B (0.3, 1)")
        ax.text(1.2, 2.08, "1/p = 2", color=INK["secondary"], fontsize=8)
        ax.set(xlabel="presentations", ylabel="response y", xscale="log", ylim=(0, 3.2),
               title=f"BCM, initial weights from seed {seed}")
        ax.legend(loc="upper left", fontsize=7)
        late = responses[-10000:].mean(axis=0)
        print(f"BCM seed {seed}: late responses A/B = {late[0]:.3f}/{late[1]:.3f}, "
              f"theta = {out['theta'][-10000:, 0].mean():.3f} (theory: 2 and 0, theta = 2)")
    return save_figure(fig, "06_plasticity_hebbian.png")


def main():
    bm.enable_x64()
    plt.rcParams.update(PLOT_STYLE)
    for figure in (short_term, spike_timing, hebbian):
        path = figure()
        print(f"Saved {path.relative_to(path.parents[1])}")


if __name__ == "__main__":
    main()
