"""Reduced two-population model of perceptual decisions (course Days 5–6).

Wong & Wang (2006) reduced the spiking decision network of Wang (2002) to two
variables: the NMDA gating S1, S2 of two populations, each selective for one
of the two choices in a random-dot motion task. The populations excite
themselves through slow NMDA synapses and inhibit each other through a shared
pool of interneurons, which the reduction folds into a negative coupling:

    dS_i/dt = -S_i / tau_S + (1 - S_i) gamma H(x_i)
    x_1 = J_same S_1 - J_cross S_2 + I_0 + I_1 + I_noise,1   (and 1 <-> 2)
    H(x) = (a x - b) / (1 - exp(-d (a x - b)))
    tau_AMPA dI_noise,i/dt = -I_noise,i + eta_i(t) sqrt(tau_AMPA) sigma
    I_1,2 = J_ext mu0 (1 ± c'/100)

H is the population rate in Hz for a total input current x in nA, eta is unit
white noise, and c' (%) is the motion coherence. Without a stimulus the model
has a low-rate spontaneous state and two decision states; the stimulus
destabilises the spontaneous state, leaving a saddle between two attractors,
and noise plus the coherence bias decide which one the trajectory falls into.

Time is in ms throughout, so gamma H (H in Hz) enters as ``gamma * H * 1e-3``.
Trials are vectorised: one model unit is one independent trial. The rates used
for the decision read-out exclude the fast AMPA noise (see
:class:`WongWang2006`).

The end of the module holds the spiking network the reduction started from
(:class:`Wang2002Network`): 2000 conductance-based LIF neurons with AMPA, NMDA
and GABA_A synapses and Poisson background input.

References
----------
Wong, K.-F. & Wang, X.-J. (2006). A recurrent network mechanism of time
    integration in perceptual decisions. J. Neurosci. 26, 1314–1328.
Wang, X.-J. (2002). Probabilistic decision making by slow reverberation in
    cortical circuits. Neuron 36, 955–968.
Roitman, J. D. & Shadlen, M. N. (2002). Response of neurons in the lateral
    intraparietal area during a combined visual discrimination reaction time
    task. J. Neurosci. 22, 9475–9489.
"""

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import brainpy as bp
import brainpy.math as bm
import jax
import jax.numpy as jnp
import numpy as np

from neuromodels.utils import run, step_current


@dataclass(frozen=True)
class WongWangParams:
    """Parameters of the reduced model (Wong & Wang 2006); units in comments."""

    a: float = 270.0         # Hz/nA, gain of H
    b: float = 108.0         # Hz, threshold of H
    d: float = 0.154         # s, curvature of H
    gamma: float = 0.641     # NMDA saturation factor
    tau_s: float = 100.0     # ms, NMDA gating decay
    tau_ampa: float = 2.0    # ms, correlation time of the noise
    J_same: float = 0.2609   # nA, J_N11 = J_N22
    J_cross: float = 0.0497  # nA, J_N12 = J_N21 (enters with a minus sign)
    I_0: float = 0.3255      # nA, background current
    J_ext: float = 0.00052   # nA/Hz, J_A,ext
    sigma: float = 0.02      # nA, noise amplitude
    threshold: float = 15.0  # Hz, decision threshold on a population rate


PARAMS = WongWangParams()


class FixedPoint(NamedTuple):
    S1: float
    S2: float
    eigenvalues: np.ndarray  # of the Jacobian, in 1/ms
    kind: str                # "stable", "saddle" or "unstable"


def transfer(x, params: WongWangParams = PARAMS):
    """Population rate H(x) in Hz for input current ``x`` in nA (NumPy or JAX).

    At a x = b the formula reads 0/0; the limit 1/d (6.49 Hz) is used there.
    """
    xp = jnp if isinstance(x, (jax.Array, bm.Array)) else np
    y = params.a * xp.asarray(x) - params.b
    tiny = xp.abs(y) < 1e-9
    y_safe = xp.where(tiny, 1.0, y)
    return xp.where(tiny, 1.0 / params.d + 0.5 * y, y_safe / -xp.expm1(-params.d * y_safe))


def inverse_transfer(rate, params: WongWangParams = PARAMS) -> np.ndarray:
    """Input current in nA at which H reaches ``rate`` (Hz > 0), by bisection (H is increasing)."""
    rate = np.asarray(rate, dtype=float)
    lo, hi = np.full(rate.shape, -1e3), np.full(rate.shape, 1e3)  # bounds on a x - b in Hz
    for _ in range(80):  # 80 halvings of 2000 Hz reach double precision
        mid = 0.5 * (lo + hi)
        above = transfer((mid + params.b) / params.a, params) > rate
        hi, lo = np.where(above, mid, hi), np.where(above, lo, mid)
    return (0.5 * (lo + hi) + params.b) / params.a


def stimulus_currents(coherence, mu0: float, params: WongWangParams = PARAMS) -> Tuple:
    """Stimulus currents (nA) to populations 1 and 2 for coherence ``c'`` in %."""
    return params.J_ext * mu0 * (1 + coherence / 100.0), params.J_ext * mu0 * (1 - coherence / 100.0)


class WongWang2006(bp.dyn.NeuDyn):
    """Reduced decision model; unit ``k`` is an independent trial with coherence ``coherence[k]``.

    ``update(mu)`` takes the stimulus strength mu0 in Hz (0 when the stimulus is
    off). The state is integrated with Euler–Maruyama (``bp.sdeint``, 'euler'),
    which suits the additive noise; the OU noise needs dt << tau_AMPA, since its
    stationary variance comes out as sigma^2 / (2 - dt / tau_AMPA) instead of
    sigma^2 / 2 (2.6 % high at dt = 0.1 ms). The Wiener increments come from
    BrainPy's global generator, so seed it with ``bm.random.seed``.

    ``r1``, ``r2`` hold H(x_i) without the noise term: the rate the slow NMDA
    gating supports. The first time either exceeds ``params.threshold`` while
    the stimulus is on, ``decision_time`` (end-of-step time, ms) and ``choice``
    (1 or 2; 0 while undecided) are set. Thresholding the noisy rate instead
    lets 2-ms noise blips near the saddle trigger "decisions" that the network
    later reverses.
    """

    def __init__(self, coherence, params: WongWangParams = PARAMS, S_init: float = 0.1,
                 method: str = "euler", name: Optional[str] = None):
        coherence = np.atleast_1d(np.asarray(coherence, dtype=float))
        super().__init__(size=coherence.size, name=name)
        self.coherence = jnp.asarray(coherence)
        self.params = params
        self.S_init = S_init
        self.integral = bp.sdeint(f=self.drift, g=self.diffusion, method=method)
        self.reset_state()

    def reset_state(self, batch_size=None, **kwargs):
        full = lambda value, dtype=None: bm.Variable(jnp.full(self.num, value, dtype=dtype))
        self.S1, self.S2 = full(self.S_init), full(self.S_init)
        self.noise1, self.noise2 = full(0.0), full(0.0)
        self.r1, self.r2 = full(0.0), full(0.0)
        self.decision_time = full(jnp.nan)
        self.choice = full(0, dtype=jnp.int32)

    def mean_inputs(self, S1, S2, mu):
        """Total input currents (nA) without the noise."""
        p = self.params
        I1, I2 = stimulus_currents(self.coherence, mu, p)
        return p.J_same * S1 - p.J_cross * S2 + p.I_0 + I1, p.J_same * S2 - p.J_cross * S1 + p.I_0 + I2

    def drift(self, S1, S2, noise1, noise2, t, mu):
        p = self.params
        x1, x2 = self.mean_inputs(S1, S2, mu)
        dS1 = -S1 / p.tau_s + (1 - S1) * p.gamma * transfer(x1 + noise1, p) * 1e-3
        dS2 = -S2 / p.tau_s + (1 - S2) * p.gamma * transfer(x2 + noise2, p) * 1e-3
        return dS1, dS2, -noise1 / p.tau_ampa, -noise2 / p.tau_ampa

    def diffusion(self, S1, S2, noise1, noise2, t, mu):
        amplitude = self.params.sigma / np.sqrt(self.params.tau_ampa)  # nA / sqrt(ms)
        return None, None, amplitude, amplitude

    def update(self, mu=0.0):
        t = bp.share.load("t")
        dt = bp.share.load("dt")
        state = self.integral(self.S1.value, self.S2.value, self.noise1.value, self.noise2.value, t, mu, dt=dt)
        self.S1.value, self.S2.value, self.noise1.value, self.noise2.value = state
        x1, x2 = self.mean_inputs(state[0], state[1], mu)
        r1, r2 = transfer(x1, self.params), transfer(x2, self.params)
        self.r1.value, self.r2.value = r1, r2
        new = jnp.isnan(self.decision_time.value) & (jnp.maximum(r1, r2) >= self.params.threshold) & (mu > 0)
        self.decision_time.value = jnp.where(new, t + dt, self.decision_time.value)
        self.choice.value = jnp.where(new, jnp.where(r1 >= r2, 1, 2), self.choice.value)


def simulate_trials(coherences: Sequence[float], n_trials: int, mu0: float = 30.0, duration: float = 2000.0,
                    onset: float = 100.0, dt: float = 0.1, seed: int = 0, params: WongWangParams = PARAMS,
                    monitors: Sequence[str] = ()) -> Dict[str, np.ndarray]:
    """Run ``n_trials`` trials per coherence (%) with the stimulus on from ``onset`` to the end.

    Returns per-trial ``coherence``, ``choice`` (0 = no decision), ``rt``
    (decision time after stimulus onset, ms; NaN without a decision) and the
    final gating ``S1_final``, ``S2_final``, plus any ``monitors`` (all trials,
    end-of-step ``ts``).
    """
    coherence = np.repeat(np.asarray(coherences, dtype=float), n_trials)
    model = WongWang2006(coherence, params)
    bm.random.seed(seed)
    out = run(model, step_current(duration, dt, mu0, onset=onset), dt, monitors=monitors)
    out.update(coherence=coherence, choice=np.asarray(model.choice.value),
               rt=np.asarray(model.decision_time.value) - onset,
               S1_final=np.asarray(model.S1.value), S2_final=np.asarray(model.S2.value))
    return out


def summarize(result: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Choice statistics per coherence level.

    Population 1 receives the stronger input, so a choice of 1 is correct (at
    c' = 0, ``accuracy`` is simply P(choice = 1)). Decision times are means and
    standard errors (``*_sem``) over decided trials: all of them (``rt``), correct
    ones (``rt_correct``) and errors (``rt_error``); NaN where a group is too small.
    """
    coherence, choice, rt = result["coherence"], result["choice"], result["rt"]
    levels = np.unique(coherence)
    groups = {"rt": lambda decided, choice: decided,
              "rt_correct": lambda decided, choice: decided & (choice == 1),
              "rt_error": lambda decided, choice: decided & (choice == 2)}
    stats = {key: [] for key in ("n_decided", "n_correct", "undecided", *groups, *(g + "_sem" for g in groups))}
    for c in levels:
        trials = coherence == c
        decided = trials & (choice > 0)
        stats["n_decided"].append(decided.sum())
        stats["n_correct"].append((decided & (choice == 1)).sum())
        stats["undecided"].append(1 - decided.sum() / trials.sum())
        for name, select in groups.items():
            times = rt[select(decided, choice)]
            stats[name].append(times.mean() if times.size else np.nan)
            stats[name + "_sem"].append(times.std(ddof=1) / np.sqrt(times.size) if times.size > 1 else np.nan)
    stats = {key: np.asarray(value) for key, value in stats.items()}
    stats["coherence"] = levels
    stats["accuracy"] = stats["n_correct"] / stats["n_decided"]
    return stats


def fit_weibull(coherence, n_correct, n_total) -> Tuple[float, float]:
    """Maximum-likelihood fit of P(c) = 1 - 0.5 exp(-(c / alpha)^beta) by grid search.

    ``alpha`` is the coherence (%) at which accuracy reaches 82 %, ``beta`` the slope.
    """
    c, k, n = (np.asarray(v, dtype=float) for v in (coherence, n_correct, n_total))
    alpha = np.geomspace(0.5, 100.0, 500)[:, None, None]
    beta = np.linspace(0.5, 4.0, 141)[None, :, None]
    p = np.clip(1 - 0.5 * np.exp(-(c / alpha) ** beta), 1e-12, 1 - 1e-12)
    loglik = (k * np.log(p) + (n - k) * np.log(1 - p)).sum(axis=-1)
    i, j = np.unravel_index(np.argmax(loglik), loglik.shape)
    return float(alpha[i, 0, 0]), float(beta[0, j, 0])


def weibull(c, alpha: float, beta: float) -> np.ndarray:
    """Weibull psychometric function for a two-choice task (chance = 0.5)."""
    return 1 - 0.5 * np.exp(-(np.asarray(c, dtype=float) / alpha) ** beta)


# ---- Phase-plane analysis (noise-free, NumPy) -------------------------------------------

def drift(S1, S2, coherence: float = 0.0, mu0: float = 30.0, params: WongWangParams = PARAMS):
    """Noise-free dS1/dt, dS2/dt in 1/ms."""
    p = params
    I1, I2 = stimulus_currents(coherence, mu0, p)
    x1 = p.J_same * S1 - p.J_cross * S2 + p.I_0 + I1
    x2 = p.J_same * S2 - p.J_cross * S1 + p.I_0 + I2
    rate = p.gamma * 1e-3
    return (-S1 / p.tau_s + (1 - S1) * rate * transfer(x1, p),
            -S2 / p.tau_s + (1 - S2) * rate * transfer(x2, p))


def _nullcline_partner(S, own_current, params):
    """On the nullcline of population i, the other population's S as a function of S_i.

    dS_i/dt = 0 fixes the rate H(x_i) = S_i / (gamma tau_S (1 - S_i)); inverting H
    gives x_i, and x_i is linear in the other S.
    """
    p = params
    rate = S / (p.gamma * p.tau_s * 1e-3 * (1 - S))
    return (p.J_same * S + p.I_0 + own_current - inverse_transfer(rate, p)) / p.J_cross


def nullclines(coherence: float = 0.0, mu0: float = 30.0, params: WongWangParams = PARAMS,
               n_points: int = 2000) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """The S1- and S2-nullclines as (S1, S2) point arrays inside the unit square."""
    I1, I2 = stimulus_currents(coherence, mu0, params)
    S = np.linspace(1e-4, 1 - 1e-4, n_points)
    S2_on_1 = _nullcline_partner(S, I1, params)
    S1_on_2 = _nullcline_partner(S, I2, params)
    keep1 = (S2_on_1 >= 0) & (S2_on_1 <= 1)
    keep2 = (S1_on_2 >= 0) & (S1_on_2 <= 1)
    return {"S1": (S[keep1], S2_on_1[keep1]), "S2": (S1_on_2[keep2], S[keep2])}


def jacobian(S1: float, S2: float, coherence: float = 0.0, mu0: float = 30.0,
             params: WongWangParams = PARAMS, eps: float = 1e-6) -> np.ndarray:
    """Jacobian of the noise-free drift (1/ms) by central differences (error ~ eps^2)."""
    J = np.empty((2, 2))
    for k, (dS1, dS2) in enumerate(((eps, 0.0), (0.0, eps))):
        plus = drift(S1 + dS1, S2 + dS2, coherence, mu0, params)
        minus = drift(S1 - dS1, S2 - dS2, coherence, mu0, params)
        J[:, k] = (np.asarray(plus) - np.asarray(minus)) / (2 * eps)
    return J


def fixed_points(coherence: float = 0.0, mu0: float = 30.0, params: WongWangParams = PARAMS,
                 n_grid: int = 4000) -> List[FixedPoint]:
    """All fixed points in the unit square, with Jacobian eigenvalues and stability.

    Walks along the S1-nullcline, S2 = f(S1), and looks for sign changes of
    g(S1) = f2(f(S1)) - S1, where S1 = f2(S2) is the S2-nullcline; each is
    refined by bisection to machine precision.
    """
    I1, I2 = stimulus_currents(coherence, mu0, params)

    def mismatch(S1):
        S2 = _nullcline_partner(S1, I1, params)
        valid = (S2 > 0) & (S2 < 1)
        S1_back = _nullcline_partner(np.clip(S2, 1e-12, 1 - 1e-12), I2, params)
        return np.where(valid, S1_back - S1, np.nan)

    grid = np.linspace(1e-6, 1 - 1e-6, n_grid)
    g = mismatch(grid)
    brackets = np.flatnonzero(np.sign(g[:-1]) * np.sign(g[1:]) < 0)
    lo, hi = grid[brackets], grid[brackets + 1]
    g_lo = g[brackets]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        g_mid = mismatch(mid)
        same = np.sign(g_mid) == np.sign(g_lo)
        lo, hi, g_lo = np.where(same, mid, lo), np.where(same, hi, mid), np.where(same, g_mid, g_lo)
    points = []
    for S1 in 0.5 * (lo + hi):
        S2 = float(_nullcline_partner(S1, I1, params))
        eig = np.linalg.eigvals(jacobian(S1, S2, coherence, mu0, params))
        if np.all(eig.real < 0):
            kind = "stable"
        elif np.any(eig.real < 0):
            kind = "saddle"
        else:
            kind = "unstable"
        points.append(FixedPoint(float(S1), S2, eig, kind))
    return points


# ---- Spiking network of Wang (2002) -------------------------------------------------------

@dataclass(frozen=True)
class Wang2002Params:
    """Parameters of the spiking decision network (Wang 2002). Pairs are (onto E, onto I)."""

    n_exc: int = 1600
    n_inh: int = 400
    f: float = 0.15                                # fraction of E cells in each selective population
    w_plus: float = 1.7                            # weight within a selective population
    C: Tuple[float, float] = (0.5, 0.2)            # nF
    g_leak: Tuple[float, float] = (25.0, 20.0)     # nS
    t_ref: Tuple[float, float] = (2.0, 1.0)        # ms
    V_leak: float = -70.0                          # mV
    V_th: float = -50.0                            # mV
    V_reset: float = -55.0                         # mV
    E_exc: float = 0.0                             # mV, AMPA and NMDA reversal
    E_inh: float = -70.0                           # mV, GABA_A reversal
    tau_ampa: float = 2.0                          # ms
    tau_nmda_rise: float = 2.0                     # ms
    tau_nmda_decay: float = 100.0                  # ms
    alpha_nmda: float = 0.5                        # 1/ms
    tau_gaba: float = 5.0                          # ms
    Mg: float = 1.0                                # mM
    g_ext: Tuple[float, float] = (2.1, 1.62)       # nS, background and stimulus AMPA
    g_ampa: Tuple[float, float] = (0.05, 0.04)     # nS, recurrent AMPA
    g_nmda: Tuple[float, float] = (0.165, 0.13)    # nS
    g_gaba: Tuple[float, float] = (1.3, 1.0)       # nS
    rate_ext: float = 2400.0                       # Hz, 800 background inputs at 3 Hz
    delay: float = 0.5                             # ms


WANG2002 = Wang2002Params()


class PopulationComm(bp.DynamicalSystem):
    """All-to-all connections whose weight depends only on the pre- and postsynaptic populations.

    ``weights[i, j]`` (nS) links presynaptic population j to postsynaptic
    population i; summing each presynaptic population first makes a step cost
    O(N) instead of O(N^2).
    """

    def __init__(self, pre_sizes: Sequence[int], post_sizes: Sequence[int], weights):
        super().__init__()
        self.bounds = [int(b) for b in np.cumsum([0, *pre_sizes])]
        self.post_sizes = np.asarray(post_sizes)
        self.weights = jnp.asarray(weights, dtype=bm.float_)

    def update(self, x):
        x = jnp.asarray(bm.as_jax(x), dtype=self.weights.dtype)
        sums = jnp.stack([x[a:b].sum() for a, b in zip(self.bounds[:-1], self.bounds[1:])])
        return jnp.repeat(self.weights @ sums, self.post_sizes, total_repeat_length=int(self.post_sizes.sum()))


def poisson_counts(mean, k_max: int = 8):
    """Poisson samples with small means (< 1) by inverse transform, truncated at ``k_max``.

    Much faster than ``bm.random.poisson`` inside a JIT-compiled loop on CPU; for
    means near 0.25, a count above 8 has probability ~1e-11.
    """
    u = bm.as_jax(bm.random.uniform(size=jnp.shape(mean)))
    pmf = jnp.exp(-mean)
    cdf, count = pmf, (u > pmf).astype(int)
    for k in range(1, k_max + 1):
        pmf = pmf * mean / k
        cdf = cdf + pmf
        count = count + (u > cdf)
    return count


class Wang2002Network(bp.DynSysGroup):
    """Spiking decision network of Wang (2002): 1600 E and 400 I LIF neurons, all-to-all.

    E cells are ordered as population A (``f`` of them), population B, then the
    non-selective rest. Weights are ``w_plus`` within A and within B, ``w_minus``
    from B to A, A to B and non-selective to A and B, 1 elsewhere; ``w_minus``
    keeps each cell's total recurrent excitation equal to that of the unstructured
    network. Every neuron gets 2.4 kHz of Poisson AMPA input; ``update(mu)`` adds
    stimulus inputs at mu (1 ± c'/100) Hz to A and B. AMPA and GABA_A use the
    align-post projection, NMDA (with rise time, saturation and Mg block) the
    align-pre one so its gating is computed once per presynaptic cell.
    """

    def __init__(self, coherence: float = 0.0, params: Wang2002Params = WANG2002, seed: int = 0):
        super().__init__()
        p = params
        self.params = p
        n_sel = int(round(p.f * p.n_exc))
        self.sizes = (n_sel, n_sel, p.n_exc - 2 * n_sel)
        w_minus = 1 - p.f * (p.w_plus - 1) / (1 - p.f)
        W = np.array([[p.w_plus, w_minus, w_minus], [w_minus, p.w_plus, w_minus], [1.0, 1.0, 1.0]])
        to_inh = np.ones((1, 3))
        rng = np.random.default_rng(seed)
        common = dict(V_rest=p.V_leak, V_reset=p.V_reset, V_th=p.V_th, method="exp_auto")
        self.E = bp.dyn.LifRef(p.n_exc, tau=p.C[0] / p.g_leak[0] * 1e3, R=1 / p.g_leak[0], tau_ref=p.t_ref[0],
                               V_initializer=bm.asarray(rng.uniform(p.V_leak, p.V_th, p.n_exc)), **common)
        self.I = bp.dyn.LifRef(p.n_inh, tau=p.C[1] / p.g_leak[1] * 1e3, R=1 / p.g_leak[1], tau_ref=p.t_ref[1],
                               V_initializer=bm.asarray(rng.uniform(p.V_leak, p.V_th, p.n_inh)), **common)
        nmda = bp.dyn.NMDA.desc(p.n_exc, a=p.alpha_nmda, tau_decay=p.tau_nmda_decay, tau_rise=p.tau_nmda_rise)
        projections = []
        for k, post in enumerate((self.E, self.I)):
            block = W if k == 0 else to_inh
            post_sizes = self.sizes if k == 0 else (p.n_inh,)
            projections += [
                bp.dyn.FullProjAlignPostMg(
                    pre=self.E, delay=p.delay, comm=PopulationComm(self.sizes, post_sizes, p.g_ampa[k] * block),
                    syn=bp.dyn.Expon.desc(post.num, tau=p.tau_ampa), out=bp.dyn.COBA.desc(E=p.E_exc), post=post),
                bp.dyn.FullProjAlignPreSDMg(
                    pre=self.E, syn=nmda, delay=p.delay, comm=PopulationComm(self.sizes, post_sizes, p.g_nmda[k] * block),
                    out=bp.dyn.MgBlock(E=p.E_exc, cc_Mg=p.Mg), post=post),
                bp.dyn.FullProjAlignPostMg(
                    pre=self.I, delay=p.delay, comm=bp.dnn.AllToAll(p.n_inh, post.num, p.g_gaba[k]),
                    syn=bp.dyn.Expon.desc(post.num, tau=p.tau_gaba), out=bp.dyn.COBA.desc(E=p.E_inh), post=post),
            ]
        self.projections = bm.NodeList(projections)  # registered, so bp.reset_state reaches them
        # Background and stimulus spikes share kinetics and reversal with recurrent AMPA, so
        # they add to the same (merged) AMPA conductance of each population.
        self.ampa_E, self.ampa_I = projections[0].syn, projections[3].syn
        self.stim_gain = jnp.asarray(np.concatenate([np.full(n_sel, 1 + coherence / 100),
                                                     np.full(n_sel, 1 - coherence / 100), np.zeros(self.sizes[2])]))

    def update(self, mu=0.0):
        p = self.params
        dt = bp.share.load("dt")
        self.ampa_E.g.value += p.g_ext[0] * poisson_counts((p.rate_ext + mu * self.stim_gain) * dt * 1e-3)
        self.ampa_I.g.value += p.g_ext[1] * poisson_counts(jnp.full(p.n_inh, p.rate_ext * dt * 1e-3))
        for proj in self.projections:
            proj()
        self.E()
        self.I()


def simulate_spiking_trial(coherence: float, mu0: float = 40.0, onset: float = 500.0, offset: float = 1500.0,
                           duration: float = 2500.0, dt: float = 0.1, seed: int = 0,
                           params: Wang2002Params = WANG2002, window: float = 50.0) -> Dict[str, np.ndarray]:
    """One trial of the spiking network with the stimulus on during [onset, offset).

    Returns end-of-step ``ts``, the spikes of all E and I cells, the sizes of
    the E populations (A, B, non-selective; in that order along the E axis) and
    population rates (Hz, ``window`` ms box average) of A, B, the non-selective
    cells and I.
    """
    net = Wang2002Network(coherence, params, seed)
    bm.random.seed(seed)
    out = run(net, step_current(duration, dt, mu0, onset=onset, offset=offset), dt, monitors=("E.spike", "I.spike"))
    out["sizes"] = np.asarray(net.sizes)
    bounds = np.cumsum([0, *net.sizes])
    groups = {"A": out["E.spike"][:, bounds[0]:bounds[1]], "B": out["E.spike"][:, bounds[1]:bounds[2]],
              "nonselective": out["E.spike"][:, bounds[2]:], "I": out["I.spike"]}
    box = np.ones(max(1, int(round(window / dt))))
    counted = np.convolve(np.ones(len(out["ts"])), box, mode="same")  # fewer samples near the edges
    for name, spikes in groups.items():
        out[f"rate_{name}"] = np.convolve(spikes.mean(axis=1) / dt * 1e3, box, mode="same") / counted
    return out
