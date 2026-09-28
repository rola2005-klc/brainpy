"""Training a rate RNN on a perceptual decision task with backpropagation through time.

Model
-----
Following Song, Yang & Wang (2016), ``N`` rate units with synaptic currents ``x``
obey

    tau dx/dt = -x + W_rec r + W_in u + b + sqrt(2 tau) sigma_rec xi(t),   r = [x]_+,

and a linear readout ``z = W_out r + b_out`` reports the decision. Euler steps of
size ``dt`` give, with ``a = dt / tau``,

    x <- (1 - a) x + a (W_rec r + W_in u + b) + sqrt(2 a) sigma_rec N(0, 1),

where the ``sqrt(2 a)`` factor keeps the stationary noise variance independent
of ``dt``.

Task
----
A two-alternative evidence-integration task in the spirit of random-dot motion
discrimination (Gold & Shadlen 2007). Input 0 is a fixation cue whose offset is
the go signal; input 1 carries momentary evidence ``c + noise * N(0, 1)`` during
the stimulus epoch. The network must output 0 until the go signal and
``sign(c)`` afterwards. The summed evidence is a sufficient statistic for the
sign of ``c``, so the ideal observer answers ``sign(sum of evidence)`` and is
correct with probability ``Phi(|c| sqrt(n) / noise)`` after ``n`` stimulus
steps. That bound calibrates the trained network.

Training
--------
The trial is unrolled with ``bm.for_loop`` and the masked squared error is
differentiated with ``bm.grad``: reverse-mode differentiation through the
unrolled loop is backpropagation through time (Werbos 1990). Adam (Kingma & Ba
2015) updates the weights on a fresh batch every iteration, so the network never
sees the same trial twice. ``bp.BPTT`` packages the same steps behind ``fit``,
but in BrainPy 2.8.2 it leaves the global ``dt`` set to the trainer's value; the
explicit loop here scopes ``dt`` with ``bm.environment`` instead.

Analysis
--------
Psychometric curves against the ideal observer, principal components of
condition-averaged hidden states (as in Mante et al. 2013), and the
psychophysical kernel: the evidence fluctuation preceding "+" choices minus that
preceding "-" choices at each stimulus step (Kiani, Hanks & Shadlen 2008). The
kernel is flat for a perfect integrator, rises toward the end of the stimulus
for a leaky one and falls for bounded integration.

References
----------
Song HF, Yang GR, Wang X-J (2016) Training excitatory-inhibitory recurrent
neural networks for cognitive tasks: a simple and flexible framework. PLoS
Comput Biol 12:e1004792.
Werbos PJ (1990) Backpropagation through time: what it does and how to do it.
Proc IEEE 78:1550-1560.
Kingma DP, Ba J (2015) Adam: a method for stochastic optimization. ICLR.
Gold JI, Shadlen MN (2007) The neural basis of decision making. Annu Rev
Neurosci 30:535-574.
Mante V, Sussillo D, Shenoy KV, Newsome WT (2013) Context-dependent computation
by recurrent dynamics in prefrontal cortex. Nature 503:78-84.
Kiani R, Hanks TD, Shadlen MN (2008) Bounded integration in parietal cortex
underlies decisions even when viewing duration is dictated by the environment.
J Neurosci 28:3017-3029.
"""

from dataclasses import dataclass
from math import erf
from typing import Any, Callable, Dict, Optional, Tuple

import brainpy as bp
import brainpy.math as bm
import numpy as np

from neuromodels.utils import n_steps

__all__ = [
    "DecisionTask",
    "RateRNN",
    "simulate",
    "masked_mse",
    "decide",
    "train",
    "evaluate",
    "psychometric",
    "condition_averaged_pcs",
    "psychophysical_kernel",
]


@dataclass(frozen=True)
class DecisionTask:
    """Epoch durations (ms) and evidence statistics of the two-alternative task."""

    dt: float = 20.0
    fixation: float = 100.0
    stimulus: float = 800.0
    decision: float = 200.0
    coherences: Tuple[float, ...] = (0.02, 0.05, 0.1, 0.2, 0.4)
    noise: float = 1.0

    @property
    def stimulus_steps(self) -> slice:
        """Time-step indices of the stimulus epoch."""
        start = n_steps(self.fixation, self.dt)
        return slice(start, start + n_steps(self.stimulus, self.dt))

    @property
    def decision_steps(self) -> slice:
        """Time-step indices of the decision epoch."""
        start = self.stimulus_steps.stop
        return slice(start, start + n_steps(self.decision, self.dt))

    @property
    def n_steps(self) -> int:
        """Total number of time steps in a trial."""
        return self.decision_steps.stop

    def signed_coherences(self, include_zero: bool = True) -> np.ndarray:
        """All signed coherence levels in ascending order."""
        c = np.asarray(self.coherences, dtype=float)
        return np.sort(np.concatenate([-c, [0.0] if include_zero else [], c]))

    def sample(self, rng: np.random.Generator, batch_size: Optional[int] = None,
               coherence: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Draw trials as time-major arrays.

        Returns ``inputs`` (time, trial, 2), ``targets`` (time, trial, 1), ``mask``
        (time, trial) and the signed ``coherence`` (trial,). Without ``coherence``,
        ``batch_size`` levels are drawn from ``coherences`` with random signs.
        """
        if coherence is None:
            coherence = rng.choice(self.coherences, batch_size) * rng.choice([-1.0, 1.0], batch_size)
        coherence = np.asarray(coherence, dtype=float)
        stim, dec = self.stimulus_steps, self.decision_steps
        inputs = np.zeros((self.n_steps, coherence.size, 2))
        inputs[:dec.start, :, 0] = 1.0  # fixation cue; its offset is the go signal
        noise = rng.standard_normal((stim.stop - stim.start, coherence.size))
        inputs[stim, :, 1] = coherence + self.noise * noise
        targets = np.zeros((self.n_steps, coherence.size, 1))
        targets[dec, :, 0] = np.sign(coherence)
        mask = np.ones((self.n_steps, coherence.size))
        return {"inputs": inputs, "targets": targets, "mask": mask, "coherence": coherence}

    def ideal_accuracy(self, coherence) -> np.ndarray:
        """P(correct) of the ideal observer, ``Phi(|c| sqrt(n_stimulus) / noise)``."""
        n = self.stimulus_steps.stop - self.stimulus_steps.start
        z = np.abs(np.asarray(coherence, dtype=float)) * np.sqrt(n) / self.noise
        return 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))

    def zero_output_loss(self) -> float:
        """Loss of a network that always outputs 0: the fraction of decision-epoch steps."""
        return (self.decision_steps.stop - self.decision_steps.start) / self.n_steps

    def bayes_loss(self, n_samples: int = 200_000, seed: int = 0) -> float:
        """Lowest achievable loss on trials from :meth:`sample`, by Monte Carlo.

        Outputting 0 before the go signal costs nothing. In the decision epoch the
        squared error is minimised by the posterior mean ``E[sign(c) | S]`` given
        the summed evidence ``S ~ N(n c, n noise^2)``; its expected squared error,
        weighted by the fraction of decision steps, is the floor. A network sits
        slightly above it because its output cannot jump at the go signal.
        """
        rng = np.random.default_rng(seed)
        n = self.stimulus_steps.stop - self.stimulus_steps.start
        c = np.asarray(self.coherences, dtype=float)
        levels = np.concatenate([-c, c])  # sample() draws these with equal probability
        true = rng.choice(levels, n_samples)
        total = n * true + np.sqrt(n) * self.noise * rng.standard_normal(n_samples)
        log_like = -(total[:, None] - n * levels) ** 2 / (2 * n * self.noise ** 2)
        weight = np.exp(log_like - log_like.max(axis=1, keepdims=True))
        posterior_mean = weight @ np.sign(levels) / weight.sum(axis=1)
        return self.zero_output_loss() * float(np.mean((np.sign(true) - posterior_mean) ** 2))

    def ideal_kernel(self) -> float:
        """Psychophysical kernel of the ideal observer at zero coherence.

        With ``S`` the sum of ``n`` i.i.d. N(0, noise^2) samples,
        ``E[x_t | S > 0] = noise sqrt(2 / (pi n))`` for every ``t``; the kernel is
        twice that and does not depend on ``t``.
        """
        n = self.stimulus_steps.stop - self.stimulus_steps.start
        return 2.0 * self.noise * np.sqrt(2.0 / (np.pi * n))


class RateRNN(bp.DynamicalSystem):
    """Continuous-time rate network with a linear readout (see the module docstring).

    Built in ``bm.TrainingMode`` by default, so the ``bp.dnn.Dense`` weights are
    ``bm.TrainVar`` objects and the state carries a batch axis. The update reads
    ``dt`` from ``bp.share``; :func:`simulate` scopes it.
    """

    def __init__(self, n_in: int, n_rec: int, n_out: int, tau: float = 100.0, sigma_rec: float = 0.05,
                 gain: float = 0.9, activation: Callable = bm.relu, seed: int = 0, mode: Optional[bm.Mode] = None):
        super().__init__(mode=bm.training_mode if mode is None else mode)
        self.n_rec = n_rec
        self.tau = tau
        self.sigma_rec = sigma_rec
        self.activation = activation
        # 1/sqrt(fan-in) scaling keeps initial currents O(1); a recurrent gain below 1
        # starts the dynamics in a stable regime, away from exploding gradients.
        w_in = bp.init.Normal(scale=1 / np.sqrt(n_in), seed=seed)
        w_rec = bp.init.Normal(scale=gain / np.sqrt(n_rec), seed=seed + 1)
        w_out = bp.init.Normal(scale=1 / np.sqrt(n_rec), seed=seed + 2)
        self.w_in = bp.dnn.Dense(n_in, n_rec, W_initializer=w_in, b_initializer=None, mode=self.mode)
        self.w_rec = bp.dnn.Dense(n_rec, n_rec, W_initializer=w_rec, mode=self.mode)
        self.w_out = bp.dnn.Dense(n_rec, n_out, W_initializer=w_out, mode=self.mode)
        self.reset_state(self.mode)

    def reset_state(self, batch_size=None, **kwargs):
        batch_size = self.mode if batch_size is None else batch_size
        self.x = bp.init.variable_(bm.zeros, self.n_rec, batch_size)

    @property
    def rates(self):
        """Firing rates ``r = f(x)`` of the recurrent units."""
        return self.activation(self.x.value)

    def update(self, u):
        a = bp.share.load("dt") / self.tau
        noise = self.sigma_rec * np.sqrt(2 * a) * bm.random.randn(*self.x.shape)
        drive = self.w_rec(self.rates) + self.w_in(u)
        self.x.value = (1 - a) * self.x.value + a * drive + noise
        return self.w_out(self.rates)


def simulate(model: RateRNN, inputs, dt: float, reset: bool = True):
    """Unroll ``model`` over time-major ``inputs`` (time, trial, n_in).

    Returns the readout (time, trial, n_out) and the rates (time, trial, n_rec)
    after each step. Works eagerly and inside ``bm.jit`` / ``bm.grad``; ``dt`` is
    set only for the duration of the call.
    """
    inputs = bm.as_jax(inputs)
    if reset:
        model.reset(inputs.shape[1])

    def step(i, u):
        bp.share.save(t=i * dt)
        return model(u), model.rates

    with bm.environment(dt=float(dt)):
        return bm.for_loop(step, (np.arange(inputs.shape[0]), inputs))


def masked_mse(outputs, targets, mask):
    """Squared error summed over outputs and averaged over unmasked (time, trial) entries."""
    err = bm.sum((outputs - targets) ** 2, axis=-1)
    return bm.sum(err * mask) / bm.sum(mask)


def decide(outputs, task: DecisionTask) -> np.ndarray:
    """Choice (+1 or -1) per trial: the sign of the mean readout over the decision epoch."""
    z = np.asarray(outputs)[task.decision_steps, :, 0].mean(axis=0)
    return np.where(z >= 0, 1.0, -1.0)


def _accuracy(choice: np.ndarray, coherence: np.ndarray) -> float:
    """Fraction correct over trials with a defined answer (non-zero coherence)."""
    signed = coherence != 0
    return float(np.mean(choice[signed] == np.sign(coherence[signed])))


def train(model: RateRNN, task: DecisionTask, n_iterations: int = 1000, batch_size: int = 64, lr: float = 1e-2,
          lr_decay: float = 0.02, seed: int = 0, eval_every: int = 100, n_eval: int = 512) -> Dict[str, np.ndarray]:
    """Train ``model`` on ``task`` with BPTT and Adam.

    The learning rate decays exponentially from ``lr`` to ``lr * lr_decay`` over
    training; the late small steps average out minibatch noise, which otherwise
    leaves a choice bias and a leaky (recency-weighted) integrator. ``seed``
    seeds trial sampling (NumPy) and the recurrent noise (``bm.random``). Every
    ``eval_every`` iterations the accuracy on a fixed held-out batch of
    ``n_eval`` trials is recorded. Returns per-iteration ``loss`` and the arrays
    ``eval_iteration`` / ``eval_accuracy``.
    """
    bm.random.seed(seed)
    rng = np.random.default_rng(seed)
    held_out = task.sample(np.random.default_rng([seed, 1]), n_eval)
    train_vars = model.train_vars().unique()
    schedule = bp.optim.ExponentialDecayLR(lr, decay_steps=n_iterations, decay_rate=lr_decay)
    optimizer = bp.optim.Adam(lr=schedule, train_vars=train_vars)

    def loss_fn(inputs, targets, mask):
        outputs, _ = simulate(model, inputs, task.dt)
        return masked_mse(outputs, targets, mask)

    grad_fn = bm.grad(loss_fn, grad_vars=train_vars, return_value=True)

    @bm.jit
    def train_step(inputs, targets, mask):
        grads, loss = grad_fn(inputs, targets, mask)
        optimizer.update(grads)
        return loss

    predict = bm.jit(lambda inputs: simulate(model, inputs, task.dt)[0])
    losses, eval_iteration, eval_accuracy = [], [], []

    def record(iteration):
        eval_iteration.append(iteration)
        eval_accuracy.append(_accuracy(decide(predict(held_out["inputs"]), task), held_out["coherence"]))

    for it in range(n_iterations):
        if eval_every and it % eval_every == 0:
            record(it)
        batch = task.sample(rng, batch_size)
        losses.append(float(train_step(batch["inputs"], batch["targets"], batch["mask"])))
    if eval_every:
        record(n_iterations)
    return {"loss": np.array(losses), "eval_iteration": np.array(eval_iteration),
            "eval_accuracy": np.array(eval_accuracy)}


def evaluate(model: RateRNN, task: DecisionTask, n_per_level: int = 200, seed: int = 1) -> Dict[str, Any]:
    """Run held-out trials at every signed coherence (zero included) and score them.

    ``seed`` seeds the trials and the recurrent noise. Returns the trials,
    ``outputs``, ``rates``, the network's ``choice`` and the ideal observer's
    ``ideal_choice`` on the same evidence, and both accuracies over the non-zero
    coherences.
    """
    coherence = np.repeat(task.signed_coherences(), n_per_level)
    trials = task.sample(np.random.default_rng(seed), coherence=coherence)
    bm.random.seed(seed)
    outputs, rates = simulate(model, trials["inputs"], task.dt)
    choice = decide(outputs, task)
    evidence = trials["inputs"][task.stimulus_steps, :, 1].sum(axis=0)
    ideal_choice = np.where(evidence >= 0, 1.0, -1.0)
    return {**trials, "outputs": np.asarray(outputs), "rates": np.asarray(rates), "choice": choice,
            "ideal_choice": ideal_choice, "accuracy": _accuracy(choice, coherence),
            "ideal_accuracy": _accuracy(ideal_choice, coherence)}


def psychometric(coherence: np.ndarray, choice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fraction of "+" choices at each signed coherence level."""
    levels = np.unique(coherence)
    return levels, np.array([np.mean(choice[coherence == c] > 0) for c in levels])


def condition_averaged_pcs(rates: np.ndarray, coherence: np.ndarray, n_components: int = 2):
    """Project trial-averaged rates for each signed coherence onto their top principal components.

    Returns ``levels``, projections of shape (level, time, component) and the
    fraction of variance each component explains. Each component is oriented so
    that it increases with coherence at the last time step.
    """
    levels = np.unique(coherence)
    means = np.stack([rates[:, coherence == c].mean(axis=1) for c in levels])  # (level, time, unit)
    centre = means.reshape(-1, means.shape[-1]).mean(axis=0)
    _, s, vt = np.linalg.svd((means - centre).reshape(-1, means.shape[-1]), full_matrices=False)
    projections = (means - centre) @ vt[:n_components].T
    orientation = np.sign(np.sum((levels - levels.mean())[:, None] * projections[:, -1], axis=0))
    projections *= np.where(orientation == 0, 1.0, orientation)
    return levels, projections, (s ** 2 / np.sum(s ** 2))[:n_components]


def psychophysical_kernel(inputs: np.ndarray, coherence: np.ndarray, choice: np.ndarray, task: DecisionTask,
                          max_coherence: float = 0.0) -> np.ndarray:
    """Mean evidence fluctuation before "+" choices minus before "-" choices, per stimulus step.

    Only trials with ``|c| <= max_coherence`` enter (zero-coherence trials by
    default). Subtracting each trial's mean evidence ``c`` leaves the noise, so
    pooling several coherences measures how fluctuations at each step sway the
    choice rather than how ``c`` does.
    """
    fluctuation = inputs[task.stimulus_steps, :, 1] - coherence
    use = np.abs(coherence) <= max_coherence
    plus, minus = use & (choice > 0), use & (choice < 0)
    return fluctuation[:, plus].mean(axis=1) - fluctuation[:, minus].mean(axis=1)
