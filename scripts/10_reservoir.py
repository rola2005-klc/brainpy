"""Days 5–6 — reservoir computing: an echo state network predicts the Mackey–Glass series.

Saves to outputs/:
  10_reservoir_mackey_glass.png  held-out forecasts 84 steps ahead, autonomous (closed-loop)
                                 generation, and forecast error against horizon
  10_reservoir_echo_state.png    the echo state property: two initial states converge under the
                                 same input for ρ(W) < 1 but not for ρ(W) ≫ 1, and forecast error
                                 against spectral radius

Reservoir: bp.dyn.Reservoir with 300 leaky-tanh units (leak rate 0.3, ρ(W) = 0.9), fed the
standardised series; readout: closed-form ridge regression (α = 1e-8) on [1, u(n), x(n)].
Mackey–Glass τ = 17, sampled every time unit; 3000 samples to train, 2000 held out.
Runs in float64.

Run:
    python scripts/10_reservoir.py
"""

import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

from neuromodels.networks.reservoir import (forecast, generate, harvest_states, mackey_glass, make_reservoir,
                                            nrmse)
from neuromodels.utils import COLORS, INK, PLOT_STYLE, save_figure

N_TRAIN, N_TOTAL, WASHOUT = 3000, 5000, 100
HORIZON = 84  # steps ahead, as in Jaeger & Haas (2004)
VALID = 0.1   # closed-loop generation counts as on track while |error| < 0.1 standard deviations


def standardised_series():
    """Mackey–Glass samples after a 500-sample transient, scaled with training-set statistics."""
    x = mackey_glass(N_TOTAL, transient=500)
    return (x - x[:N_TRAIN].mean()) / x[:N_TRAIN].std()


def persistence_nrmse(u, horizon):
    """Error of predicting ``u(n + horizon) = u(n)`` on the held-out part."""
    return nrmse(u[N_TRAIN + horizon:], u[N_TRAIN:-horizon])


def prediction(u):
    reservoir = make_reservoir(seed=0)
    states = harvest_states(reservoir, u)
    _, targets, predicted = forecast(u, states, HORIZON, N_TRAIN, washout=WASHOUT)
    weights, _, _ = forecast(u[:N_TRAIN], states[:N_TRAIN], 1, N_TRAIN, washout=WASHOUT)
    reservoir.state.value = bm.asarray(states[N_TRAIN - 1])  # the state right after the last training input
    generated = generate(reservoir, weights, u[N_TRAIN - 1], N_TOTAL - N_TRAIN)[:, 0]
    truth = u[N_TRAIN:]
    off_track = np.flatnonzero(np.abs(generated - truth) > VALID)
    valid = off_track[0] if off_track.size else len(truth)

    fig, axes = plt.subplots(3, 1, figsize=(9, 8.2), constrained_layout=True)
    window = np.arange(600)
    steps = N_TRAIN + HORIZON + window  # time index of each predicted sample
    axes[0].plot(steps, targets[window], color=INK["secondary"], linewidth=1.0, label="Mackey–Glass (held out)")
    axes[0].plot(steps, predicted[window], color=COLORS[0], linewidth=1.2, linestyle="--",
                 label=f"ESN forecast made {HORIZON} steps earlier")
    axes[0].set(xlabel="time step n", ylabel="u (standardised)",
                title=f"Direct {HORIZON}-step-ahead forecast: NRMSE {nrmse(targets, predicted):.3f} "
                      f"on {len(targets)} held-out steps")
    axes[0].legend(loc="upper right", ncol=2)

    shown = np.arange(min(len(truth), max(2 * valid, 400)))
    axes[1].plot(N_TRAIN + shown, truth[shown], color=INK["secondary"], linewidth=1.0, label="Mackey–Glass (held out)")
    axes[1].plot(N_TRAIN + shown, generated[shown], color=COLORS[1], linewidth=1.2, linestyle="--",
                 label="ESN running on its own output")
    axes[1].axvline(N_TRAIN + valid, color=INK["muted"], linewidth=0.8, linestyle=":")
    axes[1].text(N_TRAIN + valid - 4, 2.45, f"error > {VALID} sd after {valid} steps", color=INK["secondary"],
                 fontsize=8, ha="right")
    axes[1].set(xlabel="time step n", ylabel="u (standardised)", ylim=(-2.4, 2.9),
                title="Closed loop: each one-step prediction becomes the next input")
    axes[1].legend(loc="upper right", ncol=2)

    horizons = np.array([1, 2, 5, 10, 20, 50, 84, 120, 200, 300])
    errors = [nrmse(*forecast(u, states, h, N_TRAIN, washout=WASHOUT)[1:]) for h in horizons]
    axes[2].plot(horizons, errors, "o-", color=COLORS[0], markersize=4, label="ESN, direct forecast")
    axes[2].plot(horizons, [persistence_nrmse(u, h) for h in horizons], "s--", color=INK["muted"], markersize=4,
                 label="persistence  u(n + h) = u(n)")
    axes[2].set(xscale="log", yscale="log", xlabel="forecast horizon h (steps)", ylabel="held-out NRMSE",
                title="Forecast error grows with the horizon")
    axes[2].legend(loc="lower right")

    one_step = nrmse(*forecast(u, states, 1, N_TRAIN, washout=WASHOUT)[1:])
    print(f"Held-out NRMSE, 1 step ahead: {one_step:.1e} (persistence {persistence_nrmse(u, 1):.3f})")
    print(f"Held-out NRMSE, {HORIZON} steps ahead: {nrmse(targets, predicted):.3f} "
          f"(persistence {persistence_nrmse(u, HORIZON):.2f})")
    print(f"Closed loop: NRMSE over the first {HORIZON} steps {nrmse(truth[:HORIZON], generated[:HORIZON]):.1e}; "
          f"stays within {VALID} sd for {valid} steps")
    return save_figure(fig, "10_reservoir_mackey_glass.png")


def echo_state(u):
    radii = [(0.5, COLORS[0]), (0.9, COLORS[1]), (1.5, COLORS[2]), (3.0, COLORS[3])]
    start = np.random.default_rng(100).uniform(-1, 1, 300)
    steps = np.arange(1, 1501)

    fig, (ax_sep, ax_err) = plt.subplots(1, 2, figsize=(11, 3.9), constrained_layout=True)
    for radius, color in radii:
        reservoir = make_reservoir(spectral_radius=radius, seed=0)
        from_zero = harvest_states(reservoir, u[:1500])
        from_random = harvest_states(reservoir, u[:1500], initial_state=start)
        distance = np.linalg.norm(from_zero - from_random, axis=1) / np.sqrt(300)
        ax_sep.plot(steps, np.maximum(distance, 1e-17), color=color, label=f"ρ(W) = {radius}")
        print(f"ρ(W) = {radius}: per-unit distance between the two runs after 1000 steps {distance[999]:.1e}")
    ax_sep.set(yscale="log", ylim=(1e-17, 10), xlabel="time step n", ylabel="per-unit RMS distance |x − x′| / √N",
               title="Same input, two initial states")
    ax_sep.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9))

    sweep = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 3.0]
    errors = {}
    for radius in sweep:
        states = harvest_states(make_reservoir(spectral_radius=radius, seed=0), u)
        errors[radius] = nrmse(*forecast(u, states, HORIZON, N_TRAIN, washout=WASHOUT)[1:])
    ax_err.plot(sweep, [errors[r] for r in sweep], "o-", color=INK["primary"], markersize=4, linewidth=1.0,
                label="held-out NRMSE")
    for radius, color in radii:  # the runs of the left panel, in their colours
        ax_err.plot(radius, errors[radius], "o", color=color, markersize=7)
    ax_err.axvline(1.0, color=INK["muted"], linewidth=0.8, linestyle=":")
    ax_err.set(yscale="log", xlabel="spectral radius ρ(W)", ylabel=f"NRMSE, {HORIZON} steps ahead",
               title="Without echo states the readout cannot generalise")
    ax_err.legend(loc="upper left")
    best = min(errors, key=errors.get)
    print(f"{HORIZON}-step NRMSE against ρ(W): best {errors[best]:.3f} at ρ = {best}, {errors[3.0]:.2f} at ρ = 3")
    return save_figure(fig, "10_reservoir_echo_state.png")


def main():
    bm.enable_x64()
    plt.rcParams.update(PLOT_STYLE)
    u = standardised_series()
    for path in (prediction(u), echo_state(u)):
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
