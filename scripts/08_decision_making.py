"""Decision making in the reduced Wong & Wang (2006) model.

Simulates 1000 trials per motion coherence (100 ms without stimulus, then
2 s of stimulus at mu0 = 30 Hz) and reads out choice and decision time
(first crossing of 15 Hz by either population's rate). Draws psychometric and
chronometric curves, example trials, and the phase plane with and without the
stimulus. Finally runs one trial of the spiking network of Wang (2002) that the
reduced model summarises. Runs in float64, like the tests.

Saves
    outputs/08_decision_making_trials.png         example trials at two coherences
    outputs/08_decision_making_psychometric.png   accuracy and decision time vs coherence
    outputs/08_decision_making_phase_plane.png    nullclines, fixed points and trajectories
    outputs/08_decision_making_spiking.png        raster and population rates of the spiking network

Run:
    python scripts/08_decision_making.py
"""

import time

import brainpy.math as bm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neuromodels.networks.decision import (PARAMS, drift, fit_weibull, fixed_points, nullclines,
                                           simulate_spiking_trial, simulate_trials, summarize, weibull)
from neuromodels.utils import COLORS, INK, PLOT_STYLE, save_figure

DT, ONSET, DURATION, MU0 = 0.1, 100.0, 2100.0, 30.0
COHERENCES = [0.0, 3.2, 6.4, 12.8, 25.6, 51.2]  # % (Roitman & Shadlen 2002)
N_TRIALS = 1000


def coherence_axis(ax):
    """Log coherence axis that is linear below 3.2 %, so c' = 0 sits next to the doubling levels."""
    ax.set_xscale("symlog", linthresh=3.2, linscale=0.4)
    ax.set_xticks(COHERENCES)
    ax.set_xticklabels([f"{c:g}" for c in COHERENCES])
    ax.set_xlim(-0.3, 70)
    ax.set_xlabel("motion coherence c' (%)")


def plot_example_trials(ax, coherence, n_show=4, seed=11):
    out = simulate_trials([coherence], n_show, mu0=MU0, duration=DURATION, onset=ONSET, dt=DT, seed=seed,
                          monitors=("r1", "r2"))
    ts = out["ts"] - ONSET
    for k in range(n_show):
        ax.plot(ts, out["r1"][:, k], color=COLORS[0], lw=1.0, label="population 1" if k == 0 else None)
        ax.plot(ts, out["r2"][:, k], color=COLORS[1], lw=1.0, label="population 2" if k == 0 else None)
        ax.plot(out["rt"][k], PARAMS.threshold, "o", ms=3.5, color=INK["primary"])
    ax.axhline(PARAMS.threshold, color=INK["muted"], lw=0.8, ls="--")
    ax.text(1480, PARAMS.threshold + 0.8, "threshold", color=INK["secondary"], fontsize=8, ha="right")
    ax.axvline(0.0, color=INK["muted"], lw=0.8)
    ax.set_title(f"c' = {coherence:g} %")
    ax.set_xlabel("time from stimulus onset (ms)")
    ax.set_xlim(-ONSET, 1500)


def plot_phase_plane(ax, coherence, mu0, trajectories=None):
    grid = np.linspace(0, 1, 41)
    S1, S2 = np.meshgrid(grid, grid)
    dS1, dS2 = drift(S1, S2, coherence, mu0)
    ax.streamplot(S1, S2, dS1, dS2, color=INK["grid"], density=1.0, linewidth=0.7, arrowsize=0.7)
    if trajectories is not None:
        for k in range(trajectories[0].shape[1]):
            ax.plot(trajectories[0][:, k], trajectories[1][:, k], color=INK["muted"], lw=0.7)
    curves = nullclines(coherence, mu0)
    ax.plot(*curves["S1"], color=COLORS[0], lw=1.8, label="dS1/dt = 0")
    ax.plot(*curves["S2"], color=COLORS[1], lw=1.8, label="dS2/dt = 0")
    points = fixed_points(coherence, mu0)
    for p in points:
        filled = p.kind == "stable"
        ax.plot(p.S1, p.S2, "o", ms=7, mec=INK["primary"], mfc=INK["primary"] if filled else "white", zorder=5)
    ax.plot([], [], "o", ms=6, mec=INK["primary"], mfc=INK["primary"], label="stable")
    ax.plot([], [], "o", ms=6, mec=INK["primary"], mfc="white", label="saddle")
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("S1")
    ax.set_ylabel("S2")
    return points


def plot_spiking_trial(out, coherence, onset, offset, n_show=80):
    """Raster of ``n_show`` cells per population (top) and population rates (bottom)."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    starts = dict(zip(("A", "B", "nonselective"), np.cumsum([0, *out["sizes"][:2]])))
    groups = [("A", COLORS[0]), ("B", COLORS[1]), ("nonselective", COLORS[2]), ("I", COLORS[3])]
    for row, (name, color) in enumerate(groups):
        spikes = out["I.spike"][:, :n_show] if name == "I" else out["E.spike"][:, starts[name]:starts[name] + n_show]
        steps, cells = np.nonzero(spikes)
        axes[0].scatter(out["ts"][steps], cells + row * n_show, s=1.5, marker=".", linewidths=0, color=color,
                        rasterized=True)
        axes[1].plot(out["ts"], out[f"rate_{name}"], color=color, lw=1.2, label=name)
    axes[0].set_yticks([(k + 0.5) * n_show for k in range(4)])
    axes[0].set_yticklabels([name for name, _ in groups])
    for ax in axes:
        ax.axvspan(onset, offset, color=INK["grid"], alpha=0.5, lw=0)
    axes[0].set_title(f"Spiking network (Wang 2002), c' = {coherence:g} %: stimulus in grey")
    axes[1].set_ylabel("population rate (Hz)")
    axes[1].set_xlabel("time (ms)")
    axes[1].legend(loc="upper left", ncol=4)
    fig.tight_layout()
    return fig


def main():
    bm.set_platform("cpu")
    bm.enable_x64()
    plt.rcParams.update(PLOT_STYLE)
    start = time.perf_counter()

    result = simulate_trials(COHERENCES, N_TRIALS, mu0=MU0, duration=DURATION, onset=ONSET, dt=DT, seed=0)
    stats = summarize(result)
    alpha, beta = fit_weibull(stats["coherence"], stats["n_correct"], stats["n_decided"])
    decided = result["choice"] > 0
    changes_of_mind = np.mean(np.where(result["S1_final"] > result["S2_final"], 1, 2)[decided]
                              != result["choice"][decided])

    # Example trials.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    plot_example_trials(axes[0], 0.0)
    plot_example_trials(axes[1], 25.6)
    axes[0].set_ylabel("population rate (Hz)")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    trials_path = save_figure(fig, "08_decision_making_trials.png")

    # Psychometric and chronometric curves.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    c, acc, n = stats["coherence"], stats["accuracy"], stats["n_decided"]
    fine = np.linspace(0, 70, 400)
    axes[0].plot(fine, weibull(fine, alpha, beta), color=COLORS[0], lw=1.2,
                 label=rf"Weibull fit ($\alpha$ = {alpha:.1f} %, $\beta$ = {beta:.2f})")
    axes[0].errorbar(c, acc, yerr=np.sqrt(acc * (1 - acc) / n), fmt="o", ms=5, color=COLORS[0], capsize=2)
    axes[0].axhline(0.5, color=INK["muted"], lw=0.8, ls="--")
    axes[0].set_ylim(0.4, 1.02)
    axes[0].set_ylabel("fraction correct")
    axes[0].set_title("Psychometric curve")
    axes[0].legend(loc="lower right")
    coherence_axis(axes[0])
    # At c' = 0 no choice is correct, so that point averages all trials; error points need 20 errors.
    zero = c == 0
    rt_correct = np.where(zero, stats["rt"], stats["rt_correct"])
    rt_correct_sem = np.where(zero, stats["rt_sem"], stats["rt_correct_sem"])
    errors = ~zero & (stats["n_decided"] - stats["n_correct"] >= 20)
    axes[1].errorbar(c, rt_correct, yerr=rt_correct_sem, fmt="o-", ms=5, color=COLORS[0], capsize=2,
                     label="correct trials (all at c' = 0)")
    axes[1].errorbar(c[errors], stats["rt_error"][errors], yerr=stats["rt_error_sem"][errors], fmt="s--", ms=4,
                     color=COLORS[1], capsize=2, label="error trials")
    axes[1].set_ylabel("decision time (ms)")
    axes[1].set_title(f"Chronometric curve (threshold {PARAMS.threshold:.0f} Hz)")
    axes[1].legend(loc="upper right")
    coherence_axis(axes[1])
    fig.tight_layout()
    psychometric_path = save_figure(fig, "08_decision_making_psychometric.png")

    # Phase planes: stimulus off, and stimulus on at c' = 0 with noisy trajectories.
    traj = simulate_trials([0.0], 8, mu0=MU0, duration=DURATION, onset=ONSET, dt=DT, seed=12,
                           monitors=("S1", "S2"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    points_off = plot_phase_plane(axes[0], 0.0, 0.0)
    axes[0].set_title("No stimulus: spontaneous state is stable")
    points_on = plot_phase_plane(axes[1], 0.0, MU0, (traj["S1"], traj["S2"]))
    axes[1].set_title(rf"Stimulus on ($\mu_0$ = {MU0:g} Hz, c' = 0): two attractors, one saddle")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    phase_path = save_figure(fig, "08_decision_making_phase_plane.png")

    # One trial of the spiking network: 1 s of stimulus, then 1 s without.
    spk_coherence, spk_onset, spk_offset = 25.6, 500.0, 1500.0
    spiking = simulate_spiking_trial(spk_coherence, mu0=40.0, onset=spk_onset, offset=spk_offset, duration=2500.0,
                                     dt=DT, seed=1)
    spiking_path = save_figure(plot_spiking_trial(spiking, spk_coherence, spk_onset, spk_offset),
                               "08_decision_making_spiking.png")

    print("Fixed points (S1, S2) at c' = 0:")
    for label, points in (("stimulus off", points_off), (f"mu0 = {MU0:g} Hz", points_on)):
        described = ", ".join(f"{p.kind} ({p.S1:.3f}, {p.S2:.3f})" for p in points)
        print(f"  {label}: {described}")
    print(f"{N_TRIALS} trials per coherence, decision threshold {PARAMS.threshold:.0f} Hz:")
    print("  c' (%)   correct   decision time: all / correct / error (ms)   undecided")
    for k, level in enumerate(c):
        print(f"  {level:5.1f}    {acc[k]:.3f}     {stats['rt'][k]:5.0f} / {stats['rt_correct'][k]:5.0f} / "
              f"{stats['rt_error'][k]:5.0f}                     {stats['undecided'][k]:.3f}")
    print(f"  (c' = 0: 'correct' means population 1 won; a fair coin lies within "
          f"{2 * 0.5 / np.sqrt(stats['n_decided'][0]):.3f} of 0.5 with 95 % probability)")
    print(f"  Weibull fit: alpha = {alpha:.2f} % (82 % correct), beta = {beta:.2f}")
    print(f"  final winner differs from threshold choice in {changes_of_mind:.1%} of trials")
    ts = spiking["ts"]
    late_stim = (ts >= spk_offset - 200) & (ts < spk_offset)
    delay = ts >= spk_offset + 500
    print(f"Spiking network, c' = {spk_coherence:g} %, mu0 = 40 Hz (rates in Hz, A / B / non-selective / I):")
    for label, window in (("spontaneous (200-500 ms)", (ts >= 200) & (ts < spk_onset)),
                          ("last 200 ms of stimulus", late_stim), ("delay (2000-2500 ms)", delay)):
        rates = [spiking[f"rate_{name}"][window].mean() for name in ("A", "B", "nonselective", "I")]
        print(f"  {label:25s} " + " / ".join(f"{r:.1f}" for r in rates))
    for path in (trials_path, psychometric_path, phase_path, spiking_path):
        print(f"Saved {path}")
    print(f"Finished in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
