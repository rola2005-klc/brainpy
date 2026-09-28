"""Training a spiking network with surrogate gradients on latency-coded spike patterns.

Network
-------
Input spikes drive current-based exponential synapses (``bp.dyn.Expon``,
``tau_syn``) onto a hidden layer of leaky integrate-and-fire neurons
(``bp.dyn.Lif`` in ``bm.TrainingMode``) with, in dimensionless units,

    tau_mem dV/dt = -V + I_syn,     spike when V >= 1,  then V <- V - 1,

and the hidden spikes drive a non-spiking leaky readout (``bp.dyn.Integrator``)
through a second synaptic stage. The score of class ``k`` is the maximum of
readout trace ``k`` over the trial; the loss is softmax cross-entropy plus a
penalty on the squared spike counts of the hidden neurons.

Surrogate gradients
-------------------
A spike is ``S = H(V - 1)`` with ``H`` the Heaviside step, whose derivative is
zero everywhere except at threshold, where it is a Dirac delta. Backpropagation
through ``H`` therefore yields zero gradient for every weight upstream of a
spiking layer. Surrogate-gradient learning (Neftci, Mostafa & Zenke 2019) keeps
``H`` in the forward pass and substitutes a smooth pseudo-derivative in the
backward pass; here the fast-sigmoid derivative of SuperSpike (Zenke & Ganguli
2018), ``dS/dV ~ 1 / (beta |V - 1| + 1)^2``, i.e.
``bm.surrogate.InvSquareGrad(alpha=beta)``. The pseudo-derivative is positive
below threshold, so silent neurons still receive credit and can learn to fire;
its precise shape matters little in practice (Zenke & Vogels 2021). The loop
itself (``bm.for_loop`` inside ``bm.grad``, Adam) is the same BPTT used for rate
networks.

Task
----
Each class is a template of one spike per input channel at a class-specific
latency (drawn once from a fixed seed). A trial jitters every template spike and
adds class-independent Poisson background spikes. Every channel carries exactly
one template spike per trial in every class, so input spike counts carry no
class information: only relative timing does, as in the tempotron task of
Gütig & Sompolinsky (2006).

References
----------
Neftci EO, Mostafa H, Zenke F (2019) Surrogate gradient learning in spiking
neural networks: bringing the power of gradient-based optimization to spiking
neural networks. IEEE Signal Process Mag 36(6):51-63.
Zenke F, Ganguli S (2018) SuperSpike: supervised learning in multilayer spiking
neural networks. Neural Comput 30:1514-1541.
Zenke F, Vogels TP (2021) The remarkable robustness of surrogate gradient
learning for instilling complex function in spiking neural networks. Neural
Comput 33:899-925.
Gütig R, Sompolinsky H (2006) The tempotron: a neuron that learns spike
timing-based decisions. Nat Neurosci 9:420-428.
"""

from typing import Callable, Dict, Optional, Tuple

import brainpy as bp
import brainpy.math as bm
import jax.numpy as jnp
import numpy as np

from neuromodels.utils import n_steps

__all__ = [
    "LatencyTask",
    "SpikingClassifier",
    "heaviside",
    "spike_derivative",
    "simulate",
    "classification_loss",
    "train",
    "predict",
]


class LatencyTask:
    """Spatiotemporal spike patterns whose class is encoded only in spike latencies (times in ms)."""

    def __init__(self, n_inputs: int = 40, n_classes: int = 4, duration: float = 100.0, dt: float = 1.0,
                 latency_range: Tuple[float, float] = (5.0, 75.0), jitter: float = 5.0,
                 background_rate: float = 10.0, seed: int = 0):
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.duration = duration
        self.dt = dt
        self.jitter = jitter
        self.background_rate = background_rate  # Hz
        self.latencies = np.random.default_rng(seed).uniform(*latency_range, size=(n_classes, n_inputs))

    @property
    def n_steps(self) -> int:
        """Number of time steps in a trial."""
        return n_steps(self.duration, self.dt)

    def sample(self, rng: np.random.Generator, batch_size: Optional[int] = None,
               labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Draw trials: spikes (time, trial, input) as 0/1 floats and integer labels (trial,)."""
        if labels is None:
            labels = rng.integers(0, self.n_classes, batch_size)
        labels = np.asarray(labels)
        times = self.latencies[labels] + self.jitter * rng.standard_normal((labels.size, self.n_inputs))
        # Clipping keeps every template spike inside the trial, so counts stay class-independent.
        index = np.clip(np.floor(times / self.dt).astype(int), 0, self.n_steps - 1)
        spikes = rng.random((self.n_steps, labels.size, self.n_inputs)) < self.background_rate * self.dt / 1e3
        trial, channel = np.meshgrid(np.arange(labels.size), np.arange(self.n_inputs), indexing="ij")
        spikes[index, trial, channel] = True
        return spikes.astype(float), labels


def heaviside(x):
    """Hard spike threshold with the true derivative: zero wherever it is defined."""
    return jnp.asarray(x >= 0, dtype=bm.float_)


def spike_derivative(spk_fun: Callable, x) -> np.ndarray:
    """dS/dx that backpropagation sees for the spike function ``spk_fun``, element-wise."""
    return np.asarray(bm.vector_grad(spk_fun)(jnp.asarray(x, dtype=bm.float_)))


class SpikingClassifier(bp.DynamicalSystem):
    """Input spikes -> synapses -> LIF hidden layer -> synapses -> leaky readout, in ``bm.TrainingMode``.

    ``input_gain`` and ``output_gain`` scale the initial weights ``N(0, gain^2 / fan_in)``.
    ``spk_fun`` defaults to the SuperSpike surrogate with steepness ``beta``;
    passing :func:`heaviside` gives the same forward pass with true gradients.
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int, tau_mem: float = 10.0, tau_syn: float = 5.0,
                 tau_out: float = 20.0, beta: float = 10.0, input_gain: float = 3.0, output_gain: float = 1.0,
                 spk_fun: Optional[Callable] = None, seed: int = 0, mode: Optional[bm.Mode] = None):
        super().__init__(mode=bm.training_mode if mode is None else mode)
        spk_fun = bm.surrogate.InvSquareGrad(alpha=beta) if spk_fun is None else spk_fun
        w_in = bp.init.Normal(scale=input_gain / np.sqrt(n_in), seed=seed)
        w_out = bp.init.Normal(scale=output_gain / np.sqrt(n_hidden), seed=seed + 1)
        self.i2h = bp.dnn.Dense(n_in, n_hidden, W_initializer=w_in, b_initializer=None, mode=self.mode)
        self.syn_hidden = bp.dyn.Expon(n_hidden, tau=tau_syn, mode=self.mode)
        # Dimensionless voltage: rest and reset at 0, threshold at 1; the soft reset subtracts
        # the threshold. The surrogate gradient also flows through that reset term.
        self.hidden = bp.dyn.Lif(n_hidden, V_rest=0.0, V_reset=0.0, V_th=1.0, tau=tau_mem, spk_fun=spk_fun,
                                 V_initializer=bp.init.ZeroInit(), mode=self.mode)
        self.h2o = bp.dnn.Dense(n_hidden, n_out, W_initializer=w_out, b_initializer=None, mode=self.mode)
        self.syn_out = bp.dyn.Expon(n_out, tau=tau_syn, mode=self.mode)
        self.readout = bp.dyn.Integrator(n_out, tau=tau_out, mode=self.mode)

    def reset_state(self, batch_size=None, **kwargs):
        pass  # the layers reset their own states

    def update(self, spikes):
        hidden_spikes = self.hidden(self.syn_hidden(self.i2h(spikes)))
        return self.readout(self.syn_out(self.h2o(hidden_spikes)))


def simulate(model: SpikingClassifier, spikes, dt: float, reset: bool = True):
    """Unroll ``model`` over time-major input ``spikes`` (time, trial, n_in).

    Returns the readout traces (time, trial, n_out) and the hidden spikes
    (time, trial, n_hidden). ``dt`` is set only for the duration of the call.
    """
    spikes = bm.as_jax(spikes)
    if reset:
        model.reset(spikes.shape[1])

    def step(i, s):
        bp.share.save(t=i * dt)
        return model(s), model.hidden.spike.value

    with bm.environment(dt=float(dt)):
        return bm.for_loop(step, (np.arange(spikes.shape[0]), spikes))


def classification_loss(readout, labels, hidden_spikes=None, rate_penalty: float = 0.0):
    """Softmax cross-entropy on the per-class peak of the readout over time; also returns the scores.

    With ``rate_penalty > 0`` the mean squared spike count per hidden neuron and
    trial is added, which keeps the hidden layer sparse: cross-entropy alone keeps
    rewarding larger margins, which the network buys with more spikes.
    """
    scores = bm.max(readout, axis=0)
    loss = bp.losses.cross_entropy_loss(scores, labels)
    if rate_penalty:
        loss = loss + rate_penalty * bm.mean(bm.sum(hidden_spikes, axis=0) ** 2)
    return loss, scores


def predict(model: SpikingClassifier, spikes, dt: float) -> np.ndarray:
    """Predicted class per trial: the readout with the highest peak."""
    readout, _ = simulate(model, spikes, dt)
    return np.asarray(bm.argmax(bm.max(readout, axis=0), axis=-1))


def train(model: SpikingClassifier, task: LatencyTask, n_iterations: int = 300, batch_size: int = 64,
          lr: float = 5e-3, lr_decay: float = 0.1, rate_penalty: float = 0.02, seed: int = 0,
          eval_every: int = 25, n_eval: int = 256) -> Dict[str, np.ndarray]:
    """Train ``model`` on ``task`` with surrogate-gradient BPTT and Adam.

    The learning rate decays exponentially from ``lr`` to ``lr * lr_decay`` over
    training. ``seed`` seeds trial sampling. Returns per-iteration ``loss``,
    training-batch ``accuracy`` and mean hidden ``rate`` (Hz), and the held-out
    accuracy on a fixed batch of ``n_eval`` trials every ``eval_every`` iterations.
    """
    rng = np.random.default_rng(seed)
    held_out, held_out_labels = task.sample(np.random.default_rng([seed, 1]), n_eval)
    train_vars = model.train_vars().unique()
    schedule = bp.optim.ExponentialDecayLR(lr, decay_steps=n_iterations, decay_rate=lr_decay)
    optimizer = bp.optim.Adam(lr=schedule, train_vars=train_vars)

    def loss_fn(spikes, labels):
        readout, hidden_spikes = simulate(model, spikes, task.dt)
        loss, scores = classification_loss(readout, labels, hidden_spikes, rate_penalty)
        return loss, (scores, hidden_spikes)

    grad_fn = bm.grad(loss_fn, grad_vars=train_vars, return_value=True, has_aux=True)

    @bm.jit
    def train_step(spikes, labels):
        grads, loss, (scores, hidden_spikes) = grad_fn(spikes, labels)
        optimizer.update(grads)
        accuracy = bm.mean(bm.argmax(scores, axis=-1) == labels)
        return loss, accuracy, bm.mean(hidden_spikes) * 1e3 / task.dt

    classify = bm.jit(lambda spikes: bm.argmax(bm.max(simulate(model, spikes, task.dt)[0], axis=0), axis=-1))
    history = {"loss": [], "accuracy": [], "rate": [], "eval_iteration": [], "eval_accuracy": []}

    def record(iteration):
        history["eval_iteration"].append(iteration)
        history["eval_accuracy"].append(float(np.mean(np.asarray(classify(held_out)) == held_out_labels)))

    for it in range(n_iterations):
        if eval_every and it % eval_every == 0:
            record(it)
        spikes, labels = task.sample(rng, batch_size)
        for key, value in zip(("loss", "accuracy", "rate"), train_step(spikes, labels)):
            history[key].append(float(value))
    if eval_every:
        record(n_iterations)
    return {key: np.array(value) for key, value in history.items()}
