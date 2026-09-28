"""Drink or eat? A rate model of need-based goal selection.

Richman et al. (2023) gave mice that were both hungry and thirsty a free choice
between water and food on every trial. The mice collected need-appropriate
rewards by grouping their choices into persistent bouts with stochastic
transitions between them. This module asks whether the two-population decision
circuit of chapter 08 (the reduced model of Wong & Wang 2006) produces that
behaviour when its inputs are slow need signals instead of motion evidence.

Circuit
-------
Goal populations W (drink) and F (eat) have NMDA gating variables S_W, S_F:

    dS_i/dt = -S_i / tau_S + (1 - S_i) gamma H(x_i)
    x_W = J_same S_W - J_cross S_F + I_0 + J_ext mu_max T + n_W
    x_F = J_same S_F - J_cross S_W + I_0 + J_ext mu_max H + n_F

T and H in [0, 1] are thirst and hunger, and n_i is Ornstein-Uhlenbeck noise
with correlation time tau_AMPA (stationary SD sigma / sqrt 2, as in chapter 08).
With J_same = 0.24 nA, below the decision network's 0.2609 nA, a goal state
exists only while a need drives it: without need only the disengaged state is
stable; with both needs high, two goal attractors compete and noise switches
between them; with one need much larger, only its goal survives.

Behaviour
---------
A trial arrives every ``iti`` ms. The mouse takes the goal whose rate
(noise-free, as in chapter 08) exceeds ``threshold`` (the larger if both do)
and misses the trial otherwise. Each reward lowers its need,
T -> T - delta_water or H -> H - delta_food: the homeostatic feedback that
eventually ends the session in satiety.

Status
------
Not fitted to data. The circuit parameters are Wong & Wang's except J_same and
the noise: sigma = 0.04 nA (twice theirs) gives bouts of about ten trials; with
their 0.02 nA the goal attractors almost never switch by noise. mu_max, sigma,
the deltas and iti (15 s) are placeholders, to be fitted to the behaviour of
Richman et al. (2023).

References
----------
Richman, E. B., Ticea, N., Allen, W. E., Deisseroth, K. & Luo, L. (2023). Neural
    landscape diffusion resolves conflicts between needs across time. Nature 623,
    571-579.
Wong, K.-F. & Wang, X.-J. (2006). A recurrent network mechanism of time
    integration in perceptual decisions. J. Neurosci. 26, 1314-1328.
"""

import dataclasses
from typing import Dict, List

import brainpy as bp
import brainpy.math as bm
import jax.numpy as jnp
import numpy as np

from neuromodels.networks.decision import PARAMS, FixedPoint, fixed_points, transfer
from neuromodels.utils import run

MISS, WATER, FOOD = 0, 1, 2
GOAL_PARAMS = dataclasses.replace(PARAMS, J_same=0.24)  # weaker recurrence: goals need a need


def need_stimulus(thirst: float, hunger: float, mu_max: float = 40.0):
    """Map needs onto chapter 08's stimulus terms: mean strength mu0 (Hz) and coherence c' (%).

    The need drives are mu_max T and mu_max H, which chapter 08 writes as
    mu0 (1 ± c'/100).
    """
    mu_water, mu_food = mu_max * thirst, mu_max * hunger
    mu0 = 0.5 * (mu_water + mu_food)
    coherence = 0.0 if mu0 == 0 else 100.0 * (mu_water - mu_food) / (mu_water + mu_food)
    return mu0, coherence


def goal_states(thirst: float, hunger: float, mu_max: float = 40.0, params=GOAL_PARAMS) -> List[FixedPoint]:
    """Fixed points of the noise-free goal circuit at fixed needs (S1 = S_W, S2 = S_F)."""
    mu0, coherence = need_stimulus(thirst, hunger, mu_max)
    return fixed_points(coherence, mu0, params)


class NeedChoice(bp.DynamicalSystem):
    """Goal-selection circuit with homeostatic needs; one unit per simulated session.

    ``update()`` is one trial: it integrates the goal circuit for ``iti`` ms in
    steps of ``dt`` (Euler for S, exact Ornstein-Uhlenbeck steps for the noise),
    reads out the choice and consumes the reward. Monitor ``choice``
    (:data:`MISS`, :data:`WATER`, :data:`FOOD`), ``thirst``, ``hunger``,
    ``r_W`` and ``r_F``. The noise comes from the model's own generator, reseeded
    by ``reset_state``, so every run from reset is identical.
    """

    def __init__(self, num_sessions: int, thirst=1.0, hunger=1.0, params=GOAL_PARAMS, mu_max: float = 40.0,
                 sigma: float = 0.04, delta_water: float = 0.01, delta_food: float = 0.01,
                 iti: float = 15000.0, dt: float = 1.0, threshold: float = 15.0, seed: int = 0):
        super().__init__()
        self.num = num_sessions
        self.params, self.mu_max, self.sigma = params, mu_max, sigma
        self.delta_water, self.delta_food = delta_water, delta_food
        self.iti, self.dt, self.threshold, self.seed = iti, dt, threshold, seed
        self.fine_steps = int(round(iti / dt))
        self.thirst0 = jnp.broadcast_to(jnp.asarray(thirst, dtype=bm.float_), (num_sessions,))
        self.hunger0 = jnp.broadcast_to(jnp.asarray(hunger, dtype=bm.float_), (num_sessions,))
        # Exact OU step: n <- decay n + kick N(0, 1), with stationary SD sigma / sqrt(2).
        self.noise_decay = float(np.exp(-dt / params.tau_ampa))
        self.noise_kick = float(sigma / np.sqrt(2.0) * np.sqrt(1.0 - self.noise_decay ** 2))
        self.rng = bm.random.RandomState(seed)
        zeros = lambda: bm.Variable(jnp.zeros(num_sessions, dtype=bm.float_))
        self.S_W, self.S_F, self.n_W, self.n_F = zeros(), zeros(), zeros(), zeros()
        self.r_W, self.r_F = zeros(), zeros()
        self.thirst, self.hunger = bm.Variable(self.thirst0), bm.Variable(self.hunger0)
        self.choice = bm.Variable(jnp.zeros(num_sessions, dtype=jnp.int32))
        self.reset_state()

    def reset_state(self, batch_size=None, **kwargs):
        spontaneous = goal_states(0.0, 0.0, self.mu_max, self.params)[0].S1  # the only fixed point without need
        for S in (self.S_W, self.S_F):
            S.value = jnp.full(self.num, spontaneous, dtype=bm.float_)
        for var in (self.n_W, self.n_F, self.r_W, self.r_F):
            var.value = jnp.zeros(self.num, dtype=bm.float_)
        self.thirst.value, self.hunger.value = self.thirst0, self.hunger0
        self.choice.value = jnp.zeros(self.num, dtype=jnp.int32)
        self.rng.seed(self.seed)

    def mean_inputs(self, S_W, S_F):
        """Noise-free input currents (nA) to the water and food populations."""
        p = self.params
        drive = p.J_ext * self.mu_max
        x_W = p.J_same * S_W - p.J_cross * S_F + p.I_0 + drive * self.thirst.value
        x_F = p.J_same * S_F - p.J_cross * S_W + p.I_0 + drive * self.hunger.value
        return x_W, x_F

    def _fine_step(self, i):
        p = self.params
        x_W, x_F = self.mean_inputs(self.S_W.value, self.S_F.value)
        rate = p.gamma * 1e-3  # H in Hz, time in ms
        dS_W = -self.S_W.value / p.tau_s + (1 - self.S_W.value) * rate * transfer(x_W + self.n_W.value, p)
        dS_F = -self.S_F.value / p.tau_s + (1 - self.S_F.value) * rate * transfer(x_F + self.n_F.value, p)
        self.S_W.value = self.S_W.value + self.dt * dS_W
        self.S_F.value = self.S_F.value + self.dt * dS_F
        self.n_W.value = self.noise_decay * self.n_W.value + self.noise_kick * self.rng.randn(self.num)
        self.n_F.value = self.noise_decay * self.n_F.value + self.noise_kick * self.rng.randn(self.num)

    def update(self, offered=1.0):
        bm.for_loop(self._fine_step, jnp.arange(self.fine_steps))
        x_W, x_F = self.mean_inputs(self.S_W.value, self.S_F.value)
        r_W, r_F = transfer(x_W, self.params), transfer(x_F, self.params)
        engaged = (jnp.maximum(r_W, r_F) >= self.threshold) & (offered > 0)
        choice = jnp.where(engaged, jnp.where(r_W >= r_F, WATER, FOOD), MISS).astype(self.choice.value.dtype)
        self.thirst.value = jnp.maximum(self.thirst.value - self.delta_water * (choice == WATER), 0.0)
        self.hunger.value = jnp.maximum(self.hunger.value - self.delta_food * (choice == FOOD), 0.0)
        self.r_W.value, self.r_F.value, self.choice.value = r_W, r_F, choice
        return choice


def simulate_sessions(num_sessions: int, n_trials: int, thirst=1.0, hunger=1.0, seed: int = 0,
                      **kwargs) -> Dict[str, np.ndarray]:
    """Run ``num_sessions`` independent sessions of ``n_trials`` trials each.

    Returns trial-by-session arrays ``choice``, ``thirst`` and ``hunger`` (needs
    after each trial's reward), the rates ``r_W``, ``r_F`` at each decision, and
    ``ts``, the end of each trial in ms.
    """
    model = NeedChoice(num_sessions, thirst, hunger, seed=seed, **kwargs)
    return run(model, np.ones(n_trials), dt=model.iti, monitors=("choice", "thirst", "hunger", "r_W", "r_F"))


def engaged_sequences(choices: np.ndarray) -> List[np.ndarray]:
    """Per session (columns), the sequence of non-miss choices in trial order."""
    choices = np.asarray(choices)
    return [column[column != MISS] for column in choices.T]


def stay_probabilities(choices: np.ndarray) -> Dict[str, float]:
    """P(next engaged choice repeats the last one), after water (``"water"``) and after food (``"food"``).

    Misses do not break a bout: consecutive engaged choices are compared.
    """
    stays = {WATER: [0, 0], FOOD: [0, 0]}
    for sequence in engaged_sequences(choices):
        for previous, following in zip(sequence[:-1], sequence[1:]):
            stays[previous][0] += previous == following
            stays[previous][1] += 1
    return {name: (stays[goal][0] / stays[goal][1] if stays[goal][1] else float("nan"))
            for name, goal in (("water", WATER), ("food", FOOD))}


def bout_lengths(choices: np.ndarray, goal: int) -> np.ndarray:
    """Lengths of runs of consecutive engaged choices of ``goal``, pooled over sessions."""
    lengths = []
    for sequence in engaged_sequences(choices):
        if sequence.size == 0:
            continue
        edges = np.flatnonzero(np.diff(sequence)) + 1
        for run_ in np.split(sequence, edges):
            if run_[0] == goal:
                lengths.append(run_.size)
    return np.asarray(lengths)


def shuffled_stay_probabilities(choices: np.ndarray, seed: int = 0) -> Dict[str, float]:
    """Stay probabilities after shuffling each session's engaged choices: persistence by chance alone."""
    rng = np.random.default_rng(seed)
    columns = []
    length = max((len(s) for s in engaged_sequences(choices)), default=0)
    for sequence in engaged_sequences(choices):
        padded = np.full(length, MISS)
        padded[:sequence.size] = rng.permutation(sequence)
        columns.append(padded)
    return stay_probabilities(np.stack(columns, axis=1)) if columns else {"water": np.nan, "food": np.nan}


def switch_needs(choices: np.ndarray, thirst: np.ndarray, hunger: np.ndarray) -> Dict[str, np.ndarray]:
    """Need difference T - H at each switch between goals, split by direction.

    ``thirst`` and ``hunger`` are the monitored needs after each trial. A switch
    is an engaged choice that differs from the previous engaged choice; its need
    difference is taken just before that trial. With hysteresis, water-to-food
    switches happen below T - H = 0 and food-to-water switches above it; noise
    alone would centre both on 0.
    """
    to_food, to_water = [], []
    for s in range(choices.shape[1]):
        engaged = np.flatnonzero(choices[:, s] != MISS)
        for before, now in zip(engaged[:-1], engaged[1:]):
            if choices[before, s] == choices[now, s]:
                continue
            difference = thirst[now - 1, s] - hunger[now - 1, s]
            (to_food if choices[now, s] == FOOD else to_water).append(difference)
    return {"to_food": np.asarray(to_food), "to_water": np.asarray(to_water)}
