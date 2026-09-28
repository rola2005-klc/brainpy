"""Train a spiking network with surrogate gradients to classify latency-coded spike patterns.

64 leaky integrate-and-fire neurons (tau_mem = 10 ms, tau_syn = 5 ms, dt = 1 ms)
learn to separate four classes of spike patterns in which every input fires once
per trial, so that the class is carried by spike timing alone. Spikes are a
Heaviside step in the forward pass and take the SuperSpike fast-sigmoid
derivative in the backward pass (``neuromodels.training.snn``). The script plots
the surrogate derivative, the learning curves and spike rasters before and after
training, and checks that the true Heaviside derivative gives the input weights
no gradient at all. It runs in BrainPy's default float32.

Run from the repository root:
    python scripts/12_snn_training.py
"""

import time

import brainpy.math as bm
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from neuromodels.training import snn
from neuromodels.utils import COLORS, INK, PLOT_STYLE, save_figure

N_HIDDEN = 64
N_ITERATIONS = 400
SEED = 0


def running_mean(x: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(x, np.ones(window) / window, mode="valid")


def input_gradient_norm(task, spikes, labels, spk_fun) -> float:
    """Norm of dLoss/dW_in at initialisation for a given spike function (same seed, same weights)."""
    model = snn.SpikingClassifier(task.n_inputs, N_HIDDEN, task.n_classes, spk_fun=spk_fun, seed=SEED)

    def loss():
        readout, _ = snn.simulate(model, spikes, task.dt)
        return snn.classification_loss(readout, labels)[0]

    return float(jnp.linalg.norm(bm.grad(loss, grad_vars=model.i2h.W)()))


def count_classifier_accuracy(task, rng, n_train: int = 2000, n_test: int = 1000) -> float:
    """Nearest-centroid classifier on input spike counts: the baseline a rate code would get."""
    spikes, labels = task.sample(rng, n_train)
    centroids = np.stack([spikes.sum(axis=0)[labels == k].mean(axis=0) for k in range(task.n_classes)])
    spikes, labels = task.sample(rng, n_test)
    distance = ((spikes.sum(axis=0)[:, None, :] - centroids[None]) ** 2).sum(axis=-1)
    return float(np.mean(distance.argmin(axis=1) == labels))


def plot_learning(task, history, spk_fun):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1), constrained_layout=True)

    ax = axes[0]
    v = np.linspace(-1, 1, 401)
    ax.plot(v, v >= 0, color=INK["secondary"], lw=1.0, label="forward: H(V − ϑ)")
    ax.plot(v, np.zeros_like(v), color=INK["muted"], lw=1.0, label="true dH/dV (0; δ at ϑ)")
    ax.plot(v, snn.spike_derivative(spk_fun, v), color=COLORS[2], lw=1.5, label="backward: surrogate dS/dV")
    ax.set(xlabel="V − ϑ (threshold units)", ylabel="value", ylim=(-0.05, 1.3), title="Spike nonlinearity")
    ax.legend(loc="upper left")

    iterations = np.arange(1, history["loss"].size + 1)
    window = 20
    ax = axes[1]
    ax.plot(iterations, history["loss"], color=INK["grid"], lw=0.6)
    ax.plot(iterations[window - 1:], running_mean(history["loss"], window), color=COLORS[0], lw=1.5)
    ax.axhline(np.log(task.n_classes), color=INK["muted"], lw=0.8)
    ax.text(iterations[-1], np.log(task.n_classes), "chance: ln 4", color=INK["secondary"], fontsize=7,
            ha="right", va="bottom")
    ax.set(xlabel="iteration", ylabel="cross-entropy + rate penalty", title="Training loss")

    ax = axes[2]
    ax.plot(iterations[window - 1:], running_mean(history["accuracy"], window), color=COLORS[0], lw=1.5,
            label="training batches")
    ax.plot(history["eval_iteration"], history["eval_accuracy"], "o-", color=COLORS[1], ms=3.5, lw=1.2,
            label="held-out (256 trials)")
    ax.axhline(1 / task.n_classes, color=INK["muted"], lw=0.8)
    ax.text(iterations[-1], 1 / task.n_classes, "chance", color=INK["secondary"], fontsize=7, ha="right",
            va="bottom")
    ax.set(xlabel="iteration", ylabel="fraction correct", ylim=(0, 1.05), title="Accuracy")
    ax.legend(loc="lower right")

    ax = axes[3]
    ax.plot(iterations[window - 1:], running_mean(history["rate"], window), color=COLORS[0], lw=1.5)
    ax.set(xlabel="iteration", ylabel="mean rate (Hz)", ylim=(0, None), title="Hidden-layer firing rate")
    return fig


def neuron_order(hidden_spikes: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Sort neurons by preferred class, then by mean first-spike time; silent neurons last."""
    counts = np.stack([hidden_spikes[:, labels == k].sum(axis=(0, 1)) for k in range(n_classes)])
    preferred = np.where(counts.sum(axis=0) > 0, counts.argmax(axis=0), n_classes)
    fired = hidden_spikes.any(axis=0)  # (trial, neuron)
    first_sum = np.where(fired, hidden_spikes.argmax(axis=0), 0).sum(axis=0)
    mean_first = np.where(fired.any(axis=0), first_sum / np.maximum(fired.sum(axis=0), 1), np.inf)
    return np.lexsort((mean_first, preferred))


def plot_rasters(task, examples, before, after, readout, order):
    ts = (np.arange(task.n_steps) + 1) * task.dt  # end-of-step times
    input_order = np.argsort(task.latencies[0])
    fig, axes = plt.subplots(4, task.n_classes, figsize=(11, 8.2), sharex=True, constrained_layout=True)
    rows = [(examples, input_order, "input (sorted by class-1 latency)"),
            (before, order, "hidden, before training"),
            (after, order, "hidden, after training")]
    for k in range(task.n_classes):
        for row, (spikes, sort, label) in enumerate(rows):
            ax = axes[row, k]
            t_index, unit = np.nonzero(spikes[:, k, sort])
            ax.plot(ts[t_index], unit, ".", color=INK["primary"], ms=1.8)
            ax.set_ylim(-1, spikes.shape[-1])
            if k == 0:
                ax.set_ylabel(label, fontsize=8)
        axes[0, k].set_title(f"class {k + 1} example")
        ax = axes[3, k]
        for j in range(task.n_classes):
            ax.plot(ts, readout[:, k, j], color=COLORS[j], lw=1.3, label=f"readout {j + 1}")
        ax.set(xlabel="time (ms)", ylim=(1.1 * readout.min(), 1.1 * readout.max()))
    axes[3, 0].set_ylabel("readout after training", fontsize=8)
    axes[3, -1].legend(loc="upper left", fontsize=7)
    return fig


def main():
    plt.rcParams.update(PLOT_STYLE)
    task = snn.LatencyTask()
    spk_fun = bm.surrogate.InvSquareGrad(alpha=10.0)
    model = snn.SpikingClassifier(task.n_inputs, N_HIDDEN, task.n_classes, spk_fun=spk_fun, seed=SEED)

    rng = np.random.default_rng(SEED + 1)
    batch, batch_labels = task.sample(rng, 64)
    surrogate_norm = input_gradient_norm(task, batch, batch_labels, spk_fun)
    true_norm = input_gradient_norm(task, batch, batch_labels, snn.heaviside)
    print(f"|dLoss/dW_in| at initialisation: surrogate {surrogate_norm:.3g}, true Heaviside derivative {true_norm:.3g}")

    examples, example_labels = task.sample(rng, labels=np.arange(task.n_classes))
    before = np.asarray(snn.simulate(model, examples, task.dt)[1])
    start = time.time()
    history = snn.train(model, task, n_iterations=N_ITERATIONS, batch_size=64, seed=SEED, eval_every=25)
    print(f"Trained {N_HIDDEN} LIF neurons for {N_ITERATIONS} iterations in {time.time() - start:.0f} s")
    readout, after = (np.asarray(x) for x in snn.simulate(model, examples, task.dt))

    test, test_labels = task.sample(np.random.default_rng(SEED + 2), 1000)
    accuracy = np.mean(snn.predict(model, test, task.dt) == test_labels)
    hidden = np.asarray(snn.simulate(model, test, task.dt)[1])
    print(f"Loss: {history['loss'][:10].mean():.3f} (first 10) -> {history['loss'][-20:].mean():.3f} (last 20); "
          f"chance ln {task.n_classes} = {np.log(task.n_classes):.3f}")
    print(f"Held-out accuracy: {accuracy:.3f} (chance {1 / task.n_classes:.2f}); nearest-centroid on input "
          f"spike counts: {count_classifier_accuracy(task, np.random.default_rng(SEED + 3)):.3f}")
    print(f"Hidden firing rate: {history['rate'][:10].mean():.1f} Hz (first 10 batches) -> "
          f"{hidden.mean() * 1e3 / task.dt:.1f} Hz after training; "
          f"silent neurons {np.mean(~hidden.any(axis=(0, 1))):.0%}")

    fig = plot_learning(task, history, spk_fun)
    print(f"Saved {save_figure(fig, '12_snn_training_learning.png')}")
    order = neuron_order(hidden, test_labels, task.n_classes)
    fig = plot_rasters(task, examples, before, after, readout, order)
    print(f"Saved {save_figure(fig, '12_snn_training_rasters.png')}")
    plt.close("all")


if __name__ == "__main__":
    main()
