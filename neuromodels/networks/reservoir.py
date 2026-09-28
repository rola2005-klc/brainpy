r"""Reservoir computing with an echo state network (course Days 5–6).

An echo state network (ESN; Jaeger 2001) drives a fixed random recurrent network,
the reservoir, with an input signal and trains only a linear readout. With
leaky-integrator tanh units (Jaeger et al. 2007; Lukoševičius 2012):

.. math::
    x(n) = (1 - \alpha)\, x(n-1) + \alpha \tanh\big(W_{in} u(n) + W x(n-1) + b\big)

    y(n) = W_{out} [1; u(n); x(n)]

``W`` is a random matrix rescaled to a chosen spectral radius ρ(W) = max |λᵢ(W)|.
The readout is ridge regression in closed form. With the rows of Φ equal to
``[1, u(n), x(n)]`` and the rows of Y equal to the targets,

.. math::
    W_{out} = \arg\min_W \|\Phi W - Y\|^2 + \beta \|D W\|^2 = (\Phi^T \Phi + \beta D)^{-1} \Phi^T Y,

where D is the identity, optionally with the bias entry zeroed.

Echo state property (ESP): after a washout the state depends only on the input
history, so runs from different initial states converge under the same input.
The largest singular value σ_max(W) < 1 guarantees it, because tanh is
1-Lipschitz and the update then contracts. ρ(W) < 1 is the usual working
criterion. For zero input the ESP fails once the linearization at the origin,
(1 − α)I + αW, has spectral radius above 1. A large random W with ρ(W) > 1 has
real eigenvalues close to +ρ(W), so in practice this happens as soon as
ρ(W) > 1 (Jaeger 2001; Jaeger et al. 2007; Yildiz, Jaeger & Kiebel 2012).

The reservoir is BrainPy's ``bp.dyn.Reservoir`` (activation type "internal",
the update above), built from weights drawn with a seeded NumPy generator.
BrainPy multiplies row vectors (``x @ W``), so its ``Wrec`` is the transpose of
``W`` above, with the same spectral radius. The readout is NumPy.
``tests/test_reservoir.py`` checks that ``bp.RidgeTrainer`` finds the same
weights.

Benchmark: the Mackey–Glass delay equation (Mackey & Glass 1977), chaotic for
τ = 17 and the standard ESN test series (Jaeger & Haas 2004):

.. math::
    \frac{dx}{dt} = \frac{\beta\, x(t-\tau)}{1 + x(t-\tau)^n} - \gamma\, x(t),
    \qquad \beta = 0.2,\ \gamma = 0.1,\ n = 10,\ \tau = 17.

References
----------
Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent
neural networks. GMD Report 148, German National Research Center for
Information Technology.
Jaeger, H. & Haas, H. (2004). Harnessing nonlinearity: predicting chaotic systems
and saving energy in wireless communication. Science 304, 78–80.
Jaeger, H., Lukoševičius, M., Popovici, D. & Siewert, U. (2007). Optimization and
applications of echo state networks with leaky-integrator neurons. Neural
Networks 20, 335–352.
Lukoševičius, M. (2012). A practical guide to applying echo state networks. In
Neural Networks: Tricks of the Trade, 2nd ed., LNCS 7700, 659–686. Springer.
Mackey, M. C. & Glass, L. (1977). Oscillation and chaos in physiological control
systems. Science 197, 287–289.
Yildiz, I. B., Jaeger, H. & Kiebel, S. J. (2012). Re-visiting the echo state
property. Neural Networks 35, 1–9.
"""

from typing import Optional, Tuple

import brainpy as bp
import brainpy.math as bm
import numpy as np

from neuromodels.utils import run


def mackey_glass(n_samples: int, tau: float = 17.0, beta: float = 0.2, gamma: float = 0.1, n: int = 10,
                 x0: float = 1.2, dt: float = 0.1, sample_every: int = 10, transient: int = 0) -> np.ndarray:
    """Mackey–Glass series sampled every ``dt * sample_every`` time units.

    Integrates with classical RK4 at step ``dt`` from the constant history
    ``x(t ≤ 0) = x0``. The half-step stages need ``x(t − τ + dt/2)``, which
    falls between stored points; cubic Hermite interpolation from the stored
    values and slopes keeps it fourth-order accurate, so the scheme as a whole
    stays fourth order. The first ``transient`` samples are dropped.
    """
    lag = int(round(tau / dt))
    if abs(lag * dt - tau) > 1e-9 * tau:
        raise ValueError("tau must be a whole number of steps dt")
    total = (transient + n_samples) * sample_every
    x = np.empty(total + 1)
    slope = np.empty(total + 1)

    def rhs(now, delayed):
        return beta * delayed / (1.0 + delayed ** n) - gamma * now

    def past(k):  # x at grid index k; the history is constant for k ≤ 0
        return x0 if k <= 0 else x[k]

    def past_midpoint(k):  # x at t_k + dt/2, by cubic Hermite interpolation on [t_k, t_k+1]
        return x0 if k < 0 else 0.5 * (x[k] + x[k + 1]) + dt * (slope[k] - slope[k + 1]) / 8.0

    x[0] = x0
    slope[0] = rhs(x0, x0)  # right-hand slope at t = 0; the history itself is flat
    for i in range(total):
        now, mid = x[i], past_midpoint(i - lag)
        k1 = rhs(now, past(i - lag))
        k2 = rhs(now + 0.5 * dt * k1, mid)
        k3 = rhs(now + 0.5 * dt * k2, mid)
        k4 = rhs(now + dt * k3, past(i - lag + 1))
        x[i + 1] = now + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        slope[i + 1] = rhs(x[i + 1], past(i + 1 - lag))
    return x[transient * sample_every::sample_every][:n_samples]


def make_reservoir(num_units: int = 300, num_inputs: int = 1, spectral_radius: Optional[float] = 0.9,
                   leak_rate: float = 0.3, input_scaling: float = 1.0, bias_scaling: float = 0.2, seed: int = 0,
                   **kwargs) -> bp.dyn.Reservoir:
    """Leaky-tanh reservoir (``bp.dyn.Reservoir``) with dense uniform random weights.

    ``W_in ~ U(±input_scaling)``, ``b ~ U(±bias_scaling)`` and ``W ~ U(±1)``, which
    BrainPy rescales to ``spectral_radius`` (``None`` keeps it as drawn). The
    weights come from ``numpy.random.default_rng(seed)``, so the reservoir does not
    depend on BrainPy's global random state. Extra keyword arguments go to
    ``bp.dyn.Reservoir``, e.g. ``mode=bm.batching_mode`` for BrainPy's trainers.
    """
    rng = np.random.default_rng(seed)
    W_in = rng.uniform(-input_scaling, input_scaling, (num_inputs, num_units))
    W = rng.uniform(-1.0, 1.0, (num_units, num_units))
    b = rng.uniform(-bias_scaling, bias_scaling, num_units)
    return bp.dyn.Reservoir(num_inputs, num_units, leaky_rate=leak_rate, activation="tanh",
                            Win_initializer=bm.asarray(W_in), Wrec_initializer=bm.asarray(W),
                            b_initializer=bm.asarray(b), in_connectivity=1.0, rec_connectivity=1.0,
                            spectral_radius=spectral_radius, **kwargs)


def harvest_states(reservoir: bp.dyn.Reservoir, inputs: np.ndarray, initial_state=None) -> np.ndarray:
    """Drive ``reservoir`` with ``inputs`` (time first) and return its states, one row per step.

    Starts from the zero state, or from ``initial_state``. The reservoir is a
    discrete-time map, so each input sample is one step (``dt = 1``), and the
    state after the last step stays in ``reservoir.state`` for :func:`generate`.
    """
    inputs = np.asarray(inputs, dtype=float)
    if inputs.ndim == 1:
        inputs = inputs[:, None]
    bp.reset_state(reservoir)
    if initial_state is not None:
        reservoir.state.value = bm.asarray(initial_state, dtype=reservoir.state.dtype)
    return run(reservoir, inputs, dt=1.0, monitors=("state",), reset=False)["state"]


def ridge_regression(features: np.ndarray, targets: np.ndarray, alpha: float, penalize_bias: bool = False) -> np.ndarray:
    """Closed-form ridge regression with an intercept.

    Solves the normal equations ``(ΦᵀΦ + α D) W = ΦᵀY`` with ``Φ = [1, features]``
    and ``D = I``, except that the bias entry of ``D`` is 0 unless
    ``penalize_bias``. Returns ``W`` with shape ``(1 + n_features, n_targets)``, or
    ``(1 + n_features,)`` for 1-D targets. Row 0 holds the bias.
    """
    design = np.hstack([np.ones((len(features), 1)), np.asarray(features, dtype=float)])
    penalty = np.full(design.shape[1], float(alpha))
    if not penalize_bias:
        penalty[0] = 0.0
    return np.linalg.solve(design.T @ design + np.diag(penalty), design.T @ np.asarray(targets, dtype=float))


def linear_readout(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Readout ``W[0] + features @ W[1:]`` for weights from :func:`ridge_regression`."""
    return weights[0] + np.asarray(features) @ weights[1:]


def nrmse(target: np.ndarray, prediction: np.ndarray) -> float:
    """Root-mean-square error divided by the standard deviation of ``target``."""
    target, prediction = np.asarray(target), np.asarray(prediction)
    return float(np.sqrt(np.mean((target - prediction) ** 2) / np.var(target)))


def forecast(inputs: np.ndarray, states: np.ndarray, horizon: int, n_train: int, washout: int = 100,
             alpha: float = 1e-8) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit and test a direct ``horizon``-step-ahead predictor ``u(n) → u(n + horizon)``.

    Features at step ``n`` are ``[u(n), x(n)]``. The readout is fitted on
    ``washout ≤ n < n_train − horizon``, so every training target lies within the
    first ``n_train`` samples, and tested on ``n ≥ n_train``. Returns the weights
    and the held-out targets and predictions.
    """
    u = np.asarray(inputs, dtype=float).reshape(len(inputs), -1)
    steps = np.arange(len(u) - horizon)
    features, targets = np.hstack([u, states])[steps], u[steps + horizon]
    train = (steps >= washout) & (steps < n_train - horizon)
    test = steps >= n_train
    weights = ridge_regression(features[train], targets[train], alpha)
    return weights, targets[test], linear_readout(features[test], weights)


def generate(reservoir: bp.dyn.Reservoir, weights: np.ndarray, last_input, n_steps: int) -> np.ndarray:
    """Run a trained one-step predictor autonomously, feeding each output back as the next input.

    Starts from ``reservoir.state``, which must be the state right after
    ``last_input``, as :func:`harvest_states` leaves it. Row ``i`` of the result
    predicts the sample ``i + 1`` steps after ``last_input``.
    """
    u = np.atleast_1d(np.asarray(last_input, dtype=float))
    x = np.asarray(reservoir.state.value)
    outputs = []
    for _ in range(n_steps):
        u = np.atleast_1d(linear_readout(np.concatenate([u, x]), weights))
        outputs.append(u)
        x = np.asarray(reservoir.update(bm.asarray(u, dtype=reservoir.state.dtype)))
    return np.array(outputs)
