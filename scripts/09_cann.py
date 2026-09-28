"""Days 5–6 — a continuous attractor neural network: bumps, persistence, and tracking.

Saves to outputs/:
  09_cann_persistence.png  a stimulus ignites a bump; it outlives the stimulus only below the
                           critical inhibition k_c, and its height follows Wu et al. (2008)
  09_cann_tracking.png     a moving stimulus drags the bump along with a steady lag, up to a
                           maximal speed beyond which the bump detaches
  09_cann_translation.png  bumps at different positions share one shape (translational invariance)

Runs in float64 with N = 256 neurons, a = 0.5 rad, J0 = 1, time in units of τ, and
RK4 at dt = 0.05 τ.

Run:
    python scripts/09_cann.py
"""

import brainpy.math as bm
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np

from neuromodels.networks.cann import (CANN1D, gaussian_input, max_tracking_speed, population_vector,
                                       ring_distance, stationary_bump, step_midpoints, tracking_lag)
from neuromodels.utils import COLORS, INK, PLOT_STYLE, run, save_figure

DT = 0.05
N = 256
HEATMAP = "Blues"
HALO = [patheffects.withStroke(linewidth=3.0, foreground="white")]  # keeps overlays readable on the heatmap


def heights(model, A=0.0):
    """Stable bump height of ``u`` for ``model``'s parameters (``A``: centred stimulus)."""
    return stationary_bump(model.k, model.rho, model.J0, model.a, A=A)[0]


def simulate(model, center, amplitude, duration):
    """Drive ``model`` with a Gaussian stimulus whose centre and height are functions of time."""
    t = step_midpoints(duration, DT)
    return run(model, gaussian_input(model.x, center(t), amplitude(t), model.a), DT, monitors=("u", "r"))


def flash(z0, height, off):
    """Stimulus fixed at ``z0`` that switches off at time ``off``."""
    return (lambda t: np.full_like(t, z0)), (lambda t: np.where(t < off, height, 0.0))


def show_activity(ax, ts, activity, vmax):
    """Space–time heatmap: time on x, preferred stimulus on y."""
    return ax.imshow(activity[::4].T, aspect="auto", origin="lower", extent=[0.0, ts[-1], -np.pi, np.pi],
                     cmap=HEATMAP, vmin=0.0, vmax=vmax, interpolation="nearest")


def break_at_seam(angle):
    """NaN where an angle jumps across ±π, so line plots do not draw a vertical stroke."""
    return np.where(np.abs(np.diff(angle, prepend=angle[0])) > np.pi, np.nan, angle)


def persistence():
    k_c = CANN1D(N).k_c
    u_c = stationary_bump(k_c, N / (2 * np.pi))[0]  # bump height at k = k_c
    off, duration = 10.0, 150.0
    ratios = [(0.5, COLORS[0]), (0.98, COLORS[1]), (1.02, COLORS[2]), (1.1, COLORS[3])]
    sweep = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 1.2, 1.3]
    runs = {ratio: simulate(CANN1D(N, k=ratio * k_c), *flash(0.0, 2 * u_c, off), duration) for ratio in sweep}

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), constrained_layout=True)
    vmax = max(runs[0.5]["u"].max(), runs[1.1]["u"].max()) / u_c
    for ax, ratio in zip(axes[0], (0.5, 1.1)):
        keep = runs[ratio]["ts"] <= 60.0
        image = show_activity(ax, runs[ratio]["ts"][keep], runs[ratio]["u"][keep] / u_c, vmax)
        ax.axvline(off, color=INK["muted"], linewidth=0.8, linestyle=":")
        ax.set(xlabel="time (τ)", ylabel="preferred stimulus x (rad)",
               title=f"k = {ratio} k_c: the bump {'persists' if ratio < 1 else 'decays'} after the stimulus (dotted)")
    fig.colorbar(image, ax=axes[0], label="u / u_c", shrink=0.9)

    ax = axes[1, 0]
    for ratio, color in ratios:
        ax.plot(runs[ratio]["ts"], runs[ratio]["u"].max(axis=1) / u_c, color=color, label=f"k = {ratio} k_c")
        height = stationary_bump(ratio * k_c, N / (2 * np.pi))[0]
        if np.isfinite(height):
            ax.axhline(height / u_c, color=INK["muted"], linewidth=0.8, linestyle="--")
    ax.axvline(off, color=INK["muted"], linewidth=0.8, linestyle=":")
    ax.set(xlabel="time (τ)", ylabel="bump height  max u / u_c", yscale="log", ylim=(1e-3, 20),
           title="Near k_c the decay stalls at the ghost of the fixed point")
    ax.legend(loc="lower left", title="dashed: stationary theory", title_fontsize=8)

    ax = axes[1, 1]
    kappa = np.linspace(0.15, 1.0, 400)
    root = np.sqrt(1.0 - kappa)
    ax.plot(kappa, (1 + root) / kappa, color=INK["secondary"], label="stable bump (Wu et al. 2008)")
    ax.plot(kappa, (1 - root) / kappa, color=INK["muted"], linestyle="--", label="unstable bump (ignition threshold)")
    ax.plot([0.15, 1.3], [0, 0], color=INK["secondary"], linewidth=0.8, label="silent state")
    ax.plot(sweep, [runs[ratio]["u"][-1].max() / u_c for ratio in sweep], "o", color=INK["primary"], markersize=4,
            label="simulation, 140 τ after the stimulus")
    ax.annotate("fold at k = k_c", xy=(1.0, 1.0), xytext=(1.04, 3.0), color=INK["secondary"], fontsize=8,
                arrowprops=dict(arrowstyle="-", color=INK["muted"], linewidth=0.8))
    ax.set(xlabel="global inhibition k / k_c", ylabel="stationary height u0 / u_c", title="Bumps exist only for k < k_c")
    ax.legend(loc="upper right")

    model = CANN1D(N)
    simulated, expected = runs[0.5]["u"][-1].max(), heights(model)
    print(f"N = {N}: k_c = {k_c:.4f}, bump height at k_c u_c = {u_c:.5f}")
    print(f"k = 0.5 k_c: bump height {simulated:.8f} vs Wu et al. {expected:.8f} (rel. error {simulated / expected - 1:.1e})")
    for ratio, _ in ratios:
        print(f"k = {ratio:4} k_c: max u at t = 150 τ is {runs[ratio]['u'][-1].max() / u_c:.3e} u_c")
    return save_figure(fig, "09_cann_persistence.png")


def lag_run(model, velocity, A, duration, t_move=20.0):
    """Ignite a bump at 0, then move a weak stimulus at ``velocity``.

    Returns end-of-step times, the monitors, the unwrapped lag, the decoded bump
    position, and the stimulus position.
    """
    center = lambda t: np.where(t < t_move, 0.0, velocity * (t - t_move))
    out = simulate(model, center, lambda t: np.where(t < 5.0, heights(model), A), duration)
    decoded = population_vector(out["r"], model.x)
    stimulus = center(out["ts"])
    # Unwrapped, a bump that falls more than π behind shows a growing lag instead of a jump.
    return out["ts"], out, np.unwrap(ring_distance(stimulus, decoded)), decoded, stimulus


def tracking():
    model = CANN1D(N)
    u0 = heights(model)
    A = 0.1 * u0  # weak input: the recurrent dynamics shape the bump, the stimulus only steers it
    U_A = heights(model, A)
    v_max = max_tracking_speed(model.tau, A, U_A, model.a)
    duration = 20.0 + 50.0 * U_A / A  # the lag relaxes with time constant ≳ τU/A, slower near the critical speed
    speeds = [(0.5, COLORS[0]), (1.3, COLORS[1]), (1.6, COLORS[2])]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
    ts, out, _, decoded, stimulus = lag_run(model, 0.5 * v_max, A, 320.0)
    image = show_activity(axes[0], ts, out["r"] / out["r"].max(), 1.0)
    axes[0].plot(ts, break_at_seam(ring_distance(stimulus, 0.0)), color=INK["secondary"], linewidth=1.0,
                 linestyle="--", label="stimulus", path_effects=HALO)
    axes[0].plot(ts, break_at_seam(decoded), color=COLORS[0], linewidth=1.2, label="population vector",
                 path_effects=HALO)
    axes[0].set(xlabel="time (τ)", ylabel="preferred stimulus x (rad)", title="Tracking at v = 0.5 v_max, A = 0.1 u0")
    axes[0].legend(loc="lower right")
    fig.colorbar(image, ax=axes[0], label="r / max r", shrink=0.9)

    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.25, 1.3, 1.35, 1.4, 1.6]
    lags, steady = {}, []
    for fraction in fractions:
        ts, _, lag, _, _ = lag_run(model, fraction * v_max, A, duration)
        lags[fraction] = (ts, lag)
        last = lag[ts > duration - 20.0]
        steady.append(last.mean() if np.ptp(last) < 1e-3 * abs(last.mean()) and last.mean() < np.pi / 2 else np.nan)
    steady = np.array(steady)

    for fraction, color in speeds:
        ts, lag = lags[fraction]
        axes[1].plot(ts, lag / model.a, color=color, label=f"v = {fraction} v_max")
    axes[1].axhline(np.sqrt(3), color=INK["muted"], linewidth=0.8, linestyle="--")
    axes[1].text(duration, np.sqrt(3) - 0.12, "√3 a: largest lag in first-order theory", color=INK["secondary"],
                 fontsize=7, ha="right", va="top")
    axes[1].set(xlabel="time (τ)", ylabel="lag  (z_stimulus − z_bump) / a", ylim=(-0.2, 6),
                title="The lag settles only below the critical speed")
    axes[1].legend(loc="upper left")

    theory_v = np.linspace(0.0, 1.0, 200)
    theory_s = [tracking_lag(f * v_max, model.tau, A, U_A, model.a) for f in theory_v]
    axes[2].plot(theory_v, np.array(theory_s) / model.a, color=INK["secondary"], label="first-order theory, U = U_A")
    axes[2].plot(fractions, steady / model.a, "o", color=INK["primary"], markersize=4, label="simulation")
    for fraction, color in speeds:  # the runs of the middle panel, in their colours
        value = steady[fractions.index(fraction)]
        if np.isfinite(value):
            axes[2].plot(fraction, value / model.a, "o", color=color, markersize=7)
    detached = [f for f, s in zip(fractions, steady) if np.isnan(s)]
    if detached:  # shade from halfway between the last tracked and the first detached speed
        edge = 0.5 * (min(detached) + max(f for f, s in zip(fractions, steady) if np.isfinite(s)))
        axes[2].axvspan(edge, max(fractions) + 0.05, color=INK["grid"], linewidth=0)
        axes[2].text(edge + 0.02, 0.15, "bump\ndetaches", color=INK["secondary"], fontsize=7)
    axes[2].set(xlabel="stimulus speed v / v_max (first-order v_max)", ylabel="steady lag s / a",
                title="Steady lag against speed")
    axes[2].legend(loc="upper left")

    half = steady[fractions.index(0.5)]
    print(f"Tracking with A = 0.1 u0: driven height U_A = {U_A / u0:.3f} u0, first-order v_max = {v_max:.4f} rad/τ")
    print(f"  v = 0.5 v_max: steady lag {half:.4f} rad vs first-order theory "
          f"{tracking_lag(0.5 * v_max, model.tau, A, U_A, model.a):.4f} rad")
    tracked = [f for f, s in zip(fractions, steady) if np.isfinite(s)]
    print(f"  steady tracking up to {max(tracked)} v_max in the sweep; detached from {min(detached, default=np.nan)} v_max")
    return save_figure(fig, "09_cann_tracking.png")


def translation():
    model = CANN1D(N)
    u0 = heights(model)
    positions = [(-2.0, COLORS[0]), (0.5, COLORS[1]), (3.0, COLORS[2])]
    fig, (ax_raw, ax_aligned) = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    errors = []
    for z0, color in positions:
        out = simulate(model, *flash(z0, u0, 10.0), 60.0)
        u = out["u"][-1]
        decoded = population_vector(out["r"][-1], model.x)
        errors.append(abs(ring_distance(decoded, z0)))
        ax_raw.plot(model.x, u / u0, color=color, label=f"stimulus at {z0:+.1f} rad")
        ax_aligned.plot(ring_distance(model.x, decoded), u / u0, "o", color=color, markersize=2.5,
                        label=f"bump from {z0:+.1f} rad")
    d = np.linspace(-np.pi, np.pi, 400)
    ax_aligned.plot(d, np.exp(-d ** 2 / (4 * model.a ** 2)), color=INK["secondary"], linewidth=1.0,
                    label="exp(−d²/4a²)")
    ax_raw.set(xlabel="preferred stimulus x (rad)", ylabel="u / u0", ylim=(-0.03, 1.3), yticks=np.linspace(0, 1, 6),
               title="Persistent bumps 50 τ after the stimulus")
    ax_raw.legend(loc="upper center", ncol=3)
    ax_aligned.set(xlabel="distance from decoded centre (rad)", ylabel="u / u0", title="One shape everywhere on the ring")
    ax_aligned.legend(loc="upper right")
    print(f"Population vector vs stimulus position (off-grid, 50 τ after removal): max error {max(errors):.1e} rad")
    return save_figure(fig, "09_cann_translation.png")


def main():
    bm.enable_x64()
    plt.rcParams.update(PLOT_STYLE)
    for path in (persistence(), tracking(), translation()):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
