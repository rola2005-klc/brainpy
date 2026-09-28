"""Beyond the course: does a mouse drink or eat? A rate model of need-based choice.

Reuses chapter 08's goal circuit (the reduced Wong-Wang model) with thirst and
hunger as its inputs and lets every reward lower its need.

Saves to outputs/:
  13_need_choice_landscape.png   goal-circuit phase planes at four need states
  13_need_choice_session.png     one session each at low and high noise: choices,
                                 needs and goal rates, trial by trial
  13_need_choice_statistics.png  need-appropriate choice, bout lengths, and the
                                 need imbalance at which the mouse switches goals

Runs in float64. The parameters are placeholders until the model is fitted to
the behaviour of Richman et al. (2023).

Run:
    python scripts/13_need_choice.py
"""

import time

import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np

from neuromodels.networks.decision import drift, nullclines
from neuromodels.networks.need_choice import (FOOD, GOAL_PARAMS, MISS, WATER, bout_lengths, goal_states,
                                              need_stimulus, shuffled_stay_probabilities, simulate_sessions,
                                              stay_probabilities, switch_needs)
from neuromodels.utils import COLORS, INK, PLOT_STYLE, save_figure

WATER_COLOR, FOOD_COLOR = COLORS[0], COLORS[1]
NOISE_LEVELS = [(0.02, "σ = 0.02 nA: switches when a need runs out"), (0.04, "σ = 0.04 nA: noise switches goals")]
N_SESSIONS, N_TRIALS = 32, 480  # 480 trials of 15 s: a two-hour session


def landscape():
    cases = [(0.1, 0.1, "sated: only the disengaged state"), (0.3, 0.3, "both needs moderate: engage or not"),
             (1.0, 1.0, "hungry and thirsty: two goals compete"), (1.0, 0.3, "mostly thirsty: only the water goal")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    grid = np.linspace(0.0, 0.8, 40)
    S_W, S_F = np.meshgrid(grid, grid)
    for ax, (thirst, hunger, title) in zip(axes, cases):
        mu0, coherence = need_stimulus(thirst, hunger)
        dW, dF = drift(S_W, S_F, coherence, mu0, GOAL_PARAMS)
        ax.streamplot(S_W, S_F, dW, dF, color=INK["grid"], density=0.8, linewidth=0.6, arrowsize=0.7)
        lines = nullclines(coherence, mu0, GOAL_PARAMS)
        ax.plot(*lines["S1"], color=WATER_COLOR, label="water goal: dS_W/dt = 0")
        ax.plot(*lines["S2"], color=FOOD_COLOR, label="food goal: dS_F/dt = 0")
        for point in goal_states(thirst, hunger):
            ax.plot(point.S1, point.S2, "o", markersize=8, markeredgecolor=INK["primary"],
                    markerfacecolor=INK["primary"] if point.kind == "stable" else "white")
        ax.set(xlim=(0, 0.8), ylim=(0, 0.8), xlabel="water-goal gating S_W", ylabel="food-goal gating S_F",
               title=f"T = {thirst:g}, H = {hunger:g}: {title}")
        ax.title.set_fontsize(9)
    axes[0].legend(loc="upper right")
    return save_figure(fig, "13_need_choice_landscape.png")


def session_figure(results):
    fig, axes = plt.subplots(3, 2, figsize=(13, 7), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [0.6, 1, 1]})
    for col, ((sigma, title), out) in enumerate(zip(NOISE_LEVELS, results)):
        trials = np.arange(1, N_TRIALS + 1)
        choice = out["choice"][:, 0]
        for goal, level, color, label in ((WATER, 1, WATER_COLOR, "drink"), (FOOD, 0, FOOD_COLOR, "eat")):
            chosen = trials[choice == goal]
            axes[0, col].plot(chosen, np.full(chosen.size, level), "|", color=color, markersize=9, label=label)
        missed = trials[choice == MISS]
        axes[0, col].plot(missed, np.full(missed.size, 0.5), "|", color=INK["muted"], markersize=5, label="miss")
        axes[0, col].set(yticks=[0, 0.5, 1], yticklabels=["eat", "miss", "drink"], ylim=(-0.5, 1.5), title=title)
        axes[1, col].plot(trials, out["thirst"][:, 0], color=WATER_COLOR, label="thirst T")
        axes[1, col].plot(trials, out["hunger"][:, 0], color=FOOD_COLOR, label="hunger H")
        axes[1, col].set(ylabel="need", ylim=(0, 1.05))
        axes[2, col].plot(trials, out["r_W"][:, 0], color=WATER_COLOR, linewidth=1, label="water goal")
        axes[2, col].plot(trials, out["r_F"][:, 0], color=FOOD_COLOR, linewidth=1, label="food goal")
        axes[2, col].axhline(GOAL_PARAMS.threshold, color=INK["muted"], linestyle="--", linewidth=0.8)
        axes[2, col].set(xlabel="trial (15 s each)", ylabel="goal rate at decision (Hz)")
    for row in range(3):
        axes[row, 1].legend(loc="upper right")
    return save_figure(fig, "13_need_choice_session.png")


def statistics_figure(results):
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), constrained_layout=True)
    edges = np.linspace(-0.6, 0.6, 13)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for (sigma, _), out, style in zip(NOISE_LEVELS, results, ("-", "--")):
        difference = (out["thirst"] - out["hunger"])[:-1].ravel()  # needs just before the next trial
        choice = out["choice"][1:].ravel()
        engaged = choice != MISS
        index = np.digitize(difference[engaged], edges) - 1
        p_water = [np.mean(choice[engaged][index == k] == WATER) if np.sum(index == k) >= 20 else np.nan
                   for k in range(len(centers))]
        axes[0].plot(centers, p_water, "o" + style, color=WATER_COLOR, markersize=4, label=f"σ = {sigma}")
    axes[0].axhline(0.5, color=INK["muted"], linewidth=0.8, linestyle=":")
    axes[0].axvline(0.0, color=INK["muted"], linewidth=0.8, linestyle=":")
    axes[0].set(xlabel="thirst − hunger before the trial", ylabel="P(drink | engaged)", ylim=(-0.02, 1.02),
                title="The larger need decides only at the extremes")
    axes[0].legend(loc="upper left")

    for (sigma, _), result, style in zip(NOISE_LEVELS, results, ("-", "--")):
        for goal, color, name in ((WATER, WATER_COLOR, "drink"), (FOOD, FOOD_COLOR, "eat")):
            lengths = np.sort(bout_lengths(result["choice"], goal))
            survival = 1.0 - np.arange(lengths.size) / lengths.size
            axes[1].step(lengths, survival, where="post", color=color, linestyle=style,
                         label=f"{name} bouts, σ = {sigma}")
    axes[1].set(xscale="log", yscale="log", xlabel="bout length (trials)", ylabel="P(bout ≥ length)",
                title="Bout lengths")
    axes[1].legend(loc="lower left")

    bins = np.linspace(-0.6, 0.6, 49)
    for ax, (sigma, _), result in zip(axes[2:], NOISE_LEVELS, results):
        switches = switch_needs(result["choice"], result["thirst"], result["hunger"])
        ax.hist(switches["to_food"], bins=bins, color=FOOD_COLOR, alpha=0.8, label="drink → eat")
        ax.hist(switches["to_water"], bins=bins, color=WATER_COLOR, alpha=0.8, label="eat → drink")
        ax.axvline(0.0, color=INK["muted"], linewidth=0.8, linestyle="--")
        ax.set(xlabel="thirst − hunger at the switch", ylabel="switches",
               title=f"σ = {sigma}: median {np.median(switches['to_food']):+.2f} / {np.median(switches['to_water']):+.2f}")
        ax.legend(loc="upper right")
    return save_figure(fig, "13_need_choice_statistics.png")


def main():
    bm.enable_x64()
    plt.rcParams.update(PLOT_STYLE)
    start = time.time()
    results = [simulate_sessions(N_SESSIONS, N_TRIALS, sigma=sigma, seed=3) for sigma, _ in NOISE_LEVELS]
    for (sigma, _), out in zip(NOISE_LEVELS, results):
        choice = out["choice"]
        stay, chance = stay_probabilities(choice), shuffled_stay_probabilities(choice)
        switches = switch_needs(choice, out["thirst"], out["hunger"])
        print(f"σ = {sigma} nA: stay after drink / eat {stay['water']:.3f} / {stay['food']:.3f} "
              f"(shuffled {chance['water']:.2f} / {chance['food']:.2f}); mean bout {bout_lengths(choice, WATER).mean():.1f} / "
              f"{bout_lengths(choice, FOOD).mean():.1f} trials; rewards {np.mean((choice == WATER).sum(0)):.0f} water, "
              f"{np.mean((choice == FOOD).sum(0)):.0f} food; final needs {out['thirst'][-1].mean():.2f} / "
              f"{out['hunger'][-1].mean():.2f}; switch at T − H median {np.median(switches['to_food']):+.3f} (to eat) / "
              f"{np.median(switches['to_water']):+.3f} (to drink)")
    for path in (landscape(), session_figure(results), statistics_figure(results)):
        print(f"Saved {path.relative_to(path.parents[1])}")
    print(f"Finished in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
