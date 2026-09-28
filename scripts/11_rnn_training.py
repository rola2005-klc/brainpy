"""Train a rate RNN to integrate noisy evidence and report its sign.

A 64-unit continuous-time ReLU network (tau = 100 ms, dt = 20 ms) learns the
two-alternative decision task of ``neuromodels.training.rnn`` by
backpropagation through time: hold the output at 0 while the evidence streams
in, then report the sign of its mean after the go signal. The script compares
the trained network with the ideal observer (loss floor, psychometric curve,
psychophysical kernel) and looks inside it with PCA of condition-averaged hidden
states. It runs in BrainPy's default float32.

Run from the repository root:
    python scripts/11_rnn_training.py
"""

import time

import brainpy.math as bm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, to_rgb

from neuromodels.training import rnn
from neuromodels.utils import COLORS, INK, PLOT_STYLE, save_figure

N_UNITS = 64
N_ITERATIONS = 1000
SEED = 0


def tint(color, strength: float):
    """Blend ``color`` with white; ``strength`` 1 keeps the color, 0 gives white."""
    return tuple(1 - strength * (1 - np.array(to_rgb(color))))


def coherence_colors(levels: np.ndarray) -> list:
    """Diverging colors: COLORS[0] for positive, COLORS[1] for negative, gray at zero, fainter when weaker."""
    magnitudes = np.unique(np.abs(levels[levels != 0]))
    strength = dict(zip(magnitudes, np.linspace(0.3, 1.0, magnitudes.size)))
    return [INK["muted"] if c == 0 else tint(COLORS[0] if c > 0 else COLORS[1], strength[abs(c)]) for c in levels]


def running_mean(x: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(x, np.ones(window) / window, mode="valid")


def zero_coherence_kernel(model, task, n_trials: int, seed: int) -> np.ndarray:
    """Psychophysical kernel of the network from ``n_trials`` zero-coherence trials."""
    trials = task.sample(np.random.default_rng(seed), coherence=np.zeros(n_trials))
    bm.random.seed(seed)
    outputs, _ = rnn.simulate(model, trials["inputs"], task.dt)
    return rnn.psychophysical_kernel(trials["inputs"], trials["coherence"], rnn.decide(outputs, task), task)


def plot_learning(task, history, result, kernel):
    loss_floor, zero_loss = task.bayes_loss(), task.zero_output_loss()
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), constrained_layout=True)

    ax = axes[0, 0]
    window = 50
    iterations = np.arange(1, history["loss"].size + 1)
    ax.plot(iterations, history["loss"], color=INK["grid"], lw=0.6)
    ax.plot(iterations[window - 1:], running_mean(history["loss"], window), color=COLORS[0], lw=1.5)
    for level, label, color, va in [(zero_loss, "output always 0", INK["muted"], "bottom"),
                                    (loss_floor, "Bayes floor", INK["secondary"], "top")]:
        ax.axhline(level, color=color, lw=0.8)
        ax.text(iterations[-1], level, label, color=INK["secondary"], fontsize=7, ha="right", va=va)
    ax.set(xlabel="iteration", ylabel="masked MSE", ylim=(0.09, 0.2),
           title=f"Training loss (gray: per batch; blue: {window}-iteration mean)")

    ax = axes[0, 1]
    ideal = task.ideal_accuracy(np.asarray(task.coherences)).mean()  # |c| is drawn uniformly
    ax.plot(history["eval_iteration"], history["eval_accuracy"], "o-", color=COLORS[0], ms=4, lw=1.5)
    for level, label, color in [(ideal, "ideal observer", INK["secondary"]), (0.5, "chance", INK["muted"])]:
        ax.axhline(level, color=color, lw=0.8)
        ax.text(history["eval_iteration"][-1], level, label, color=INK["secondary"], fontsize=7,
                ha="right", va="bottom")
    ax.set(xlabel="iteration", ylabel="fraction correct", ylim=(0.4, 0.9), title="Held-out accuracy (512 trials)")

    ax = axes[1, 0]
    levels, p_plus = rnn.psychometric(result["coherence"], result["choice"])
    fine = np.linspace(levels[0], levels[-1], 400)
    ideal = np.where(fine >= 0, task.ideal_accuracy(fine), 1 - task.ideal_accuracy(fine))
    ax.plot(fine, ideal, color=INK["secondary"], lw=1.0, label="ideal observer (closed form)")
    ax.plot(levels, p_plus, "o", color=COLORS[0], ms=5, label="trained RNN")
    ax.set(xlabel="signed coherence c", ylabel='P(choice "+")', title="Psychometric curve")
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    onset = (np.arange(kernel.size) + 0.5) * task.dt
    ax.axhline(0.0, color=INK["grid"], lw=0.8)
    ax.axhline(task.ideal_kernel(), color=INK["secondary"], lw=0.8, label="perfect integrator")
    ax.plot(onset, kernel, "o-", color=COLORS[0], ms=3, lw=1.0, label="trained RNN")
    ax.set(xlabel="time from stimulus onset (ms)", ylabel='evidence before "+" minus "-" choice',
           title="Psychophysical kernel (c = 0 trials)")
    ax.legend(loc="lower right")
    return fig


def plot_dynamics(task, result):
    levels, projections, explained = rnn.condition_averaged_pcs(result["rates"], result["coherence"])
    colors = coherence_colors(levels)
    ts = (np.arange(task.n_steps) + 1) * task.dt  # end-of-step times
    go = task.fixation + task.stimulus
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)

    for ax in axes[:2]:
        ax.axvspan(task.fixation, go, color=INK["grid"], alpha=0.6, lw=0)
    for level, color in zip(levels, colors):
        trials = result["coherence"] == level
        axes[0].plot(ts, result["outputs"][:, trials, 0].mean(axis=1), color=color, lw=1.2)
    axes[0].set(xlabel="time (ms)", ylabel="readout z", title="Output: hold, then report")
    axes[0].text(task.fixation + task.stimulus / 2, axes[0].get_ylim()[1], "stimulus", color=INK["secondary"],
                 fontsize=7, ha="center", va="top")

    for projection, color in zip(projections, colors):
        axes[1].plot(ts, projection[:, 0], color=color, lw=1.2)
    axes[1].set(xlabel="time (ms)", ylabel="PC 1", title="Hidden state integrates evidence")

    go_index, onset_index = task.stimulus_steps.stop - 1, task.stimulus_steps.start - 1
    for projection, color in zip(projections, colors):
        axes[2].plot(projection[:, 0], projection[:, 1], color=color, lw=1.2)
        axes[2].plot(*projection[go_index, :2], "o", color=color, ms=4)
    # All conditions coincide until the stimulus starts, so these two points are shared.
    for index, label, face in [(0, "trial start", "white"), (onset_index, "stimulus onset", INK["primary"])]:
        point = projections[:, index, :2].mean(axis=0)
        axes[2].plot(*point, "o", color=INK["primary"], mfc=face, ms=4)
        axes[2].annotate(label, point, xytext=(8, -2), textcoords="offset points", fontsize=7,
                         color=INK["secondary"], va="top")
    axes[2].set(xlabel=f"PC 1 ({explained[0]:.0%} var.)", ylabel=f"PC 2 ({explained[1]:.0%} var.)",
                title="State space (colored dots: go signal)")

    swatches = plt.cm.ScalarMappable(cmap=ListedColormap(colors),
                                     norm=BoundaryNorm(np.arange(levels.size + 1), levels.size))
    cbar = fig.colorbar(swatches, ax=axes, fraction=0.03, pad=0.02, ticks=np.arange(levels.size) + 0.5)
    cbar.set_ticklabels([f"{c:+.2f}" if c else "0" for c in levels])
    cbar.set_label("signed coherence c")
    cbar.outline.set_visible(False)
    return fig, explained


def main():
    plt.rcParams.update(PLOT_STYLE)
    task = rnn.DecisionTask()
    model = rnn.RateRNN(2, N_UNITS, 1, seed=SEED)

    start = time.time()
    history = rnn.train(model, task, n_iterations=N_ITERATIONS, batch_size=64, seed=SEED, eval_every=100)
    print(f"Trained {N_UNITS} units for {N_ITERATIONS} iterations in {time.time() - start:.0f} s")

    result = rnn.evaluate(model, task, n_per_level=400, seed=1)
    kernel = zero_coherence_kernel(model, task, n_trials=4000, seed=2)
    half = kernel.size // 2
    print(f"Loss: {history['loss'][:10].mean():.3f} (first 10) -> {history['loss'][-50:].mean():.3f} (last 50); "
          f"always-0 output {task.zero_output_loss():.3f}, Bayes floor {task.bayes_loss():.3f}")
    print(f"Held-out accuracy (c != 0): network {result['accuracy']:.3f}, "
          f"ideal observer on the same trials {result['ideal_accuracy']:.3f}; "
          f"P(choice +) at c = 0: {np.mean(result['choice'][result['coherence'] == 0] > 0):.3f}")
    print(f"Psychophysical kernel, first / second half of the stimulus: {kernel[:half].mean():.3f} / "
          f"{kernel[half:].mean():.3f} (perfect integrator {task.ideal_kernel():.3f})")

    fig = plot_learning(task, history, result, kernel)
    print(f"Saved {save_figure(fig, '11_rnn_training_learning.png')}")
    fig, explained = plot_dynamics(task, result)
    print(f"PC 1 and PC 2 explain {explained[0]:.0%} and {explained[1]:.0%} of condition-averaged variance")
    print(f"Saved {save_figure(fig, '11_rnn_training_dynamics.png')}")
    plt.close("all")


if __name__ == "__main__":
    main()
