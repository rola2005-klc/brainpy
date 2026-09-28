"""Synaptic plasticity: short-term dynamics, STDP, and rate-based Hebbian rules.

**Short-term plasticity** (:class:`TsodyksMarkram`). A synapse holds a fraction
``x`` of available resources and a utilization ``u``. Between spikes
``du/dt = -u/tau_f`` and ``dx/dt = (1 - x)/tau_d``; at a spike ``u`` jumps by
``U (1 - u)``, the synapse releases ``r = u+ x-`` and ``x`` drops by ``r``
(Tsodyks, Pawelzik & Markram 1998). A large ``U`` with slow recovery gives
depression; a small ``U`` with slow ``u`` decay gives facilitation (Markram,
Wang & Tsodyks 1998). Under a regular train with interval ``T`` the release
settles at ``r* = u* x*`` with ``u* = U / (1 - (1 - U) e^{-T/tau_f})`` and
``x* = (1 - e^{-T/tau_d}) / (1 - (1 - u*) e^{-T/tau_d})`` (:func:`stp_steady_state`).

**Pair-based STDP** (:class:`PairSTDP`). Each synapse keeps a presynaptic
trace (``+1`` per pre spike, decay ``tau_plus``) and a postsynaptic trace
(``+1`` per post spike, decay ``tau_minus``). A post spike potentiates by
``A_plus`` times the pre trace, a pre spike depresses by ``A_minus`` times the
post trace, so one pair with ``dt = t_post - t_pre`` changes the weight by
``A_plus e^{-dt/tau_plus}`` (``dt > 0``) or ``-A_minus e^{dt/tau_minus}``
(``dt < 0``): the window measured by Bi & Poo (1998). Spikes in the same step
do not interact. With independent Poisson trains the mean drift is
``r_pre r_post (A_plus tau_plus - A_minus tau_minus)``; Song, Miller & Abbott
(2000) made it slightly negative so that only inputs that help cause
postsynaptic spikes grow (:class:`STDPNeuron`).

**Rate-based rules.** Oja's rule ``dw/dt = eta y (x - y w)`` with ``y = w.x``
drives ``w`` to the unit-norm principal eigenvector of ``<x x^T>`` (the input
covariance when inputs have zero mean; Oja 1982; :class:`OjaNeuron`). The BCM rule ``dw/dt = eta x y (y - theta)``
with the sliding threshold ``tau_theta dtheta/dt = y^2 - theta`` makes a neuron
respond to one input pattern and ignore the others (Bienenstock, Cooper &
Munro 1982; quadratic threshold from Intrator & Cooper 1992; :class:`BCMNeuron`).
For patterns shown with probability ``p`` each, the selective state has
``y = 1/p`` for the preferred pattern and ``y = 0`` for the rest.

Differences from the BrainPy 2.8.2 built-ins: ``bp.dyn.STP`` follows the same
equations but starts ``u`` at ``U`` rather than 0, so its first spike already
releases ``U + U (1 - U) e^{-t_1/tau_f}``, and it returns ``u+ x+`` (after the
release) rather than the release ``u+ x-``. ``bp.dyn.STDP_Song2000`` is a
projection with the same trace rule (``tau_s``, ``tau_t``, ``A1``, ``A2`` play
the roles of ``tau_plus``, ``tau_minus``, ``A_plus``, ``A_minus``). It gives the
same pairing window, except that a pre and a post spike in the same step
trigger both updates there (net ``A1 - A2``) and neither here.

References
----------
Oja E (1982) J Math Biol 15:267-273.
Bienenstock EL, Cooper LN, Munro PW (1982) J Neurosci 2:32-48.
Intrator N, Cooper LN (1992) Neural Netw 5:3-17.
Tsodyks MV, Markram H (1997) PNAS 94:719-723.
Markram H, Wang Y, Tsodyks M (1998) PNAS 95:5323-5328.
Tsodyks M, Pawelzik K, Markram H (1998) Neural Comput 10:821-835.
Bi G-Q, Poo M-M (1998) J Neurosci 18:10464-10472.
Song S, Miller KD, Abbott LF (2000) Nat Neurosci 3:919-926.
Maass W, Natschlaeger T, Markram H (2002) Neural Comput 14:2531-2560.
"""

from typing import Sequence, Tuple

import brainpy as bp
import brainpy.math as bm
import numpy as np

from neuromodels.synapses import Exponential, spike_input
from neuromodels.utils import run

# Mean U, tau_d, tau_f for excitatory->excitatory (depressing) and excitatory->inhibitory
# (facilitating) cortical synapses, as compiled by Maass et al. (2002) from Markram et al. (1998).
DEPRESSING = {"U": 0.5, "tau_d": 1100.0, "tau_f": 50.0}
FACILITATING = {"U": 0.05, "tau_d": 125.0, "tau_f": 1200.0}

# Exponential time constants (ms) commonly fitted to the Bi & Poo (1998) window.
BI_POO_TAUS = {"tau_plus": 16.8, "tau_minus": 33.7}


# ----------------------------------------------------------------------------
# Short-term plasticity
# ----------------------------------------------------------------------------

class TsodyksMarkram(bp.dyn.SynDyn):
    """Tsodyks-Markram short-term plasticity; ``update`` returns the release ``u+ x-``.

    ``u`` starts at 0 and ``x`` at 1, so a rested synapse releases ``U`` on its
    first spike. Both equations are linear and decoupled, so exponential Euler
    integrates them exactly. Monitor ``u``, ``x`` and ``release``.
    """

    def __init__(self, size=1, U: float = 0.5, tau_d: float = 1100.0, tau_f: float = 50.0,
                 method: str = "exp_auto", name: str = None):
        super().__init__(size=size, name=name)
        self.U = self.init_param(U)
        self.tau_d = self.init_param(tau_d)
        self.tau_f = self.init_param(tau_f)
        self.integral = bp.odeint(bp.JointEq(self.du, self.dx), method=method)
        self.reset_state()

    def du(self, u, t):
        return -u / self.tau_f

    def dx(self, x, t):
        return (1 - x) / self.tau_d

    def reset_state(self, batch_size=None, **kwargs):
        self.u = self.init_variable(bm.zeros, batch_size)
        self.x = self.init_variable(bm.ones, batch_size)
        self.release = self.init_variable(bm.zeros, batch_size)

    def update(self, spike):
        t, dt = bp.share.load("t"), bp.share.load("dt")
        u, x = self.integral(self.u.value, self.x.value, t, dt)
        u = u + spike * self.U * (1 - u)  # facilitation acts before release (u+)
        release = spike * u * x
        self.u.value = u
        self.x.value = x - release
        self.release.value = release
        return release


class STPSynapse(bp.DynamicalSystem):
    """Tsodyks-Markram release feeding an exponential conductance (``g`` jumps by ``u+ x-``).

    Monitor ``syn.g`` for the postsynaptic conductance and ``stp.release``,
    ``stp.u``, ``stp.x`` for the presynaptic state.
    """

    def __init__(self, size=1, U: float = 0.5, tau_d: float = 1100.0, tau_f: float = 50.0,
                 tau_syn: float = 5.0, name: str = None):
        super().__init__(name=name)
        self.stp = TsodyksMarkram(size, U, tau_d, tau_f)
        self.syn = Exponential(size, tau_syn)

    def reset_state(self, batch_size=None, **kwargs):
        pass  # children reset themselves

    def update(self, spike):
        return self.syn(self.stp(spike))


def stp_steady_state(U: float, tau_d: float, tau_f: float, rate: float) -> Tuple[float, float, float]:
    """Steady ``(u+, x-, release)`` at each spike of a regular train at ``rate`` Hz."""
    interval = 1000.0 / rate
    decay_f, decay_d = np.exp(-interval / tau_f), np.exp(-interval / tau_d)
    u = U / (1 - (1 - U) * decay_f)
    x = (1 - decay_d) / (1 - (1 - u) * decay_d)
    return u, x, u * x


def stp_release_sequence(times: Sequence[float], U: float, tau_d: float, tau_f: float) -> np.ndarray:
    """Release ``u+ x-`` at each spike time (ms), from the exact spike-to-spike recursion."""
    u, x, previous, releases = 0.0, 1.0, None, []
    for t in times:
        if previous is not None:
            gap = t - previous
            u *= np.exp(-gap / tau_f)
            x = 1 - (1 - x) * np.exp(-gap / tau_d)
        u += U * (1 - u)
        releases.append(u * x)
        x -= u * x
        previous = t
    return np.array(releases)


# ----------------------------------------------------------------------------
# Spike-timing-dependent plasticity
# ----------------------------------------------------------------------------

class PairSTDP(bp.dyn.SynDyn):
    """Additive pair-based STDP for ``size`` synapses, with all-to-all trace interactions.

    ``update(pre_spike, post_spike)`` takes arrays broadcastable to ``size``: one
    postsynaptic neuron per synapse, or a single shared one (pass shape ``(1,)``).
    Each step the traces decay, pre spikes depress by ``A_minus`` times the post
    trace, post spikes potentiate by ``A_plus`` times the pre trace, the weights
    are clipped to ``[w_min, w_max]``, and only then are this step's spikes
    added to the traces. Exponential Euler solves the trace decay exactly.
    """

    def __init__(self, size=1, A_plus: float = 0.01, A_minus: float = 0.0105, tau_plus: float = 20.0,
                 tau_minus: float = 20.0, w_min: float = 0.0, w_max: float = 1.0, w_init=0.5,
                 method: str = "exp_auto", name: str = None):
        super().__init__(size=size, name=name)
        self.A_plus, self.A_minus = A_plus, A_minus
        self.tau_plus, self.tau_minus = tau_plus, tau_minus
        self.w_min, self.w_max = w_min, w_max
        self.w_init = np.asarray(w_init, dtype=float)
        self.integral = bp.odeint(bp.JointEq(self.dpre, self.dpost), method=method)
        self.reset_state()

    def dpre(self, trace_pre, t):
        return -trace_pre / self.tau_plus

    def dpost(self, trace_post, t):
        return -trace_post / self.tau_minus

    def reset_state(self, batch_size=None, **kwargs):
        self.w = self.init_variable(bm.ones, batch_size)
        self.w.value = self.w.value * self.w_init
        self.trace_pre = self.init_variable(bm.zeros, batch_size)
        self.trace_post = self.init_variable(bm.zeros, batch_size)

    def update(self, pre_spike, post_spike):
        t, dt = bp.share.load("t"), bp.share.load("dt")
        trace_pre, trace_post = self.integral(self.trace_pre.value, self.trace_post.value, t, dt)
        w = self.w.value - self.A_minus * trace_post * pre_spike + self.A_plus * trace_pre * post_spike
        self.w.value = bm.clip(w, self.w_min, self.w_max)
        self.trace_pre.value = trace_pre + pre_spike
        self.trace_post.value = trace_post + post_spike
        return self.w.value


def stdp_window(delta_t, A_plus: float, A_minus: float, tau_plus: float, tau_minus: float) -> np.ndarray:
    """Weight change for one spike pair, ``delta_t = t_post - t_pre`` (ms); zero at ``delta_t = 0``."""
    delta_t = np.asarray(delta_t, dtype=float)
    ltp = A_plus * np.exp(-np.abs(delta_t) / tau_plus)
    ltd = -A_minus * np.exp(-np.abs(delta_t) / tau_minus)
    return np.where(delta_t > 0, ltp, np.where(delta_t < 0, ltd, 0.0))


def stdp_poisson_drift(rate_pre: float, rate_post: float, A_plus: float, A_minus: float,
                       tau_plus: float, tau_minus: float) -> float:
    """Mean weight change per second for independent Poisson trains (rates in Hz, taus in ms)."""
    return rate_pre * rate_post * (A_plus * tau_plus - A_minus * tau_minus) * 1e-3


def stdp_pairing(delta_ts: Sequence[float], dt: float = 0.1, **stdp_params) -> np.ndarray:
    """Simulate one pre/post pair per ``delta_t = t_post - t_pre`` and return each weight change.

    Weights are unbounded here so the change is not clipped; ``stdp_params`` go to :class:`PairSTDP`.
    """
    delta_ts = np.asarray(delta_ts, dtype=float)
    t_pre = np.abs(delta_ts).max() + 10.0
    duration = 2 * t_pre + 10.0
    pre = spike_input([[t_pre]] * len(delta_ts), duration, dt)
    post = spike_input([[t_pre + d] for d in delta_ts], duration, dt)
    model = PairSTDP(len(delta_ts), w_min=-np.inf, w_max=np.inf, w_init=0.0, **stdp_params)
    return run(model, (pre, post), dt, monitors=("w",))["w"][-1]


class STDPNeuron(bp.DynamicalSystem):
    """Conductance-based LIF neuron whose excitatory inputs learn by additive STDP.

    Following Song, Miller & Abbott (2000): ``n_exc`` excitatory and ``n_inh``
    inhibitory Poisson inputs drive exponential conductances (decay ``tau_syn``,
    measured in units of the leak conductance) onto
    ``tau_m dV/dt = -(V - V_rest) + g_e (E_exc - V) + g_i (E_inh - V) + I``.
    Excitatory peak conductances start uniform in ``[0, g_max]`` and change by
    ``A_plus g_max`` / ``A_ratio A_plus g_max`` per pair (:class:`PairSTDP`,
    ``tau_plus = tau_minus = tau_stdp``) within ``[0, g_max]``. Inhibitory ones
    stay at ``g_inh``. The Poisson inputs are generated inside the step from a
    seeded generator, so long runs need no input arrays; pass zeros as ``I``.
    Monitor ``V``, ``spike``, ``exc.g``; the weights are ``stdp.w``.
    """

    def __init__(self, n_exc: int = 1000, n_inh: int = 200, rate_exc: float = 20.0, rate_inh: float = 10.0,
                 g_max: float = 0.015, g_inh: float = 0.05, A_plus: float = 0.005, A_ratio: float = 1.05,
                 tau_stdp: float = 20.0, tau_syn: float = 5.0, tau_m: float = 20.0, V_rest: float = -70.0,
                 V_reset: float = -60.0, V_th: float = -54.0, E_exc: float = 0.0, E_inh: float = -70.0,
                 seed: int = 0, method: str = "exp_auto", name: str = None):
        super().__init__(name=name)
        self.n_exc, self.n_inh = n_exc, n_inh
        self.rate_exc, self.rate_inh = rate_exc, rate_inh
        self.g_inh = g_inh
        self.tau_m, self.V_rest, self.V_reset, self.V_th = tau_m, V_rest, V_reset, V_th
        self.E_exc, self.E_inh = E_exc, E_inh
        self.seed = seed
        w_init = np.random.default_rng(seed).uniform(0.0, g_max, n_exc)
        self.stdp = PairSTDP(n_exc, A_plus=A_plus * g_max, A_minus=A_ratio * A_plus * g_max, tau_plus=tau_stdp,
                             tau_minus=tau_stdp, w_min=0.0, w_max=g_max, w_init=w_init)
        self.exc = Exponential(1, tau_syn)
        self.inh = Exponential(1, tau_syn)
        self.rng = bm.random.RandomState(seed)
        self.integral = bp.odeint(self.dV, method=method)
        self.reset_state()

    def dV(self, V, t, g_e, g_i, I):
        return (-(V - self.V_rest) + g_e * (self.E_exc - V) + g_i * (self.E_inh - V) + I) / self.tau_m

    def reset_state(self, batch_size=None, **kwargs):
        self.V = bm.Variable(bm.ones(1) * self.V_rest)
        self.spike = bm.Variable(bm.zeros(1, dtype=bool))
        self.rng.seed(self.seed)

    def update(self, I=0.0):
        t, dt = bp.share.load("t"), bp.share.load("dt")
        pre_exc = self.rng.random(self.n_exc) < self.rate_exc * dt * 1e-3
        pre_inh = self.rng.random(self.n_inh) < self.rate_inh * dt * 1e-3
        V = self.integral(self.V.value, t, self.exc.g.value, self.inh.g.value, I, dt)
        spike = V >= self.V_th
        self.V.value = bm.where(spike, self.V_reset, V)
        self.spike.value = spike
        # Spikes arriving at the end of this step reach the conductances and the STDP traces together.
        self.exc(bm.sum(self.stdp.w.value * pre_exc))
        self.inh(self.g_inh * bm.sum(pre_inh))
        self.stdp(pre_exc, spike)
        return spike


# ----------------------------------------------------------------------------
# Rate-based Hebbian rules
# ----------------------------------------------------------------------------

class OjaNeuron(bp.DynamicalSystem):
    """Linear neuron ``y = w.x`` learning by Oja's rule ``dw/dt = eta y (x - y w)``.

    Each step presents one input vector ``x``. Euler integration makes one step
    the classical discrete update ``w += eta dt y (x - y w)``. Monitor ``w``, ``y``.
    """

    def __init__(self, num_inputs: int, eta: float = 0.005, w_init=None, seed: int = 0, method: str = "euler",
                 name: str = None):
        super().__init__(name=name)
        self.eta = eta
        rng = np.random.default_rng(seed)
        self.w_init = rng.normal(0.0, 0.1, num_inputs) if w_init is None else np.asarray(w_init, dtype=float)
        self.integral = bp.odeint(self.derivative, method=method)
        self.reset_state()

    def derivative(self, w, t, x, y):
        return self.eta * y * (x - y * w)

    def reset_state(self, batch_size=None, **kwargs):
        self.w = bm.Variable(bm.asarray(self.w_init))
        self.y = bm.Variable(bm.zeros(1))

    def update(self, x):
        t, dt = bp.share.load("t"), bp.share.load("dt")
        y = bm.dot(self.w.value, x)
        self.w.value = self.integral(self.w.value, t, x, y, dt)
        self.y.value = bm.reshape(y, (1,))
        return y


class BCMNeuron(bp.DynamicalSystem):
    """Rectified linear neuron ``y = [w.x]+`` learning by the BCM rule with a sliding threshold.

    ``dw/dt = eta x y (y - theta)`` and ``tau_theta dtheta/dt = y^2 - theta``, both
    integrated with Euler (one input pattern per step). ``theta`` must track
    ``<y^2>`` faster than ``w`` changes, i.e. ``tau_theta`` well below the
    learning time ``~1/eta``, or the weights oscillate. Monitor ``w``, ``y``, ``theta``.
    """

    def __init__(self, num_inputs: int, eta: float = 0.002, tau_theta: float = 10.0, w_init=None,
                 seed: int = 0, method: str = "euler", name: str = None):
        super().__init__(name=name)
        self.eta, self.tau_theta = eta, tau_theta
        rng = np.random.default_rng(seed)
        self.w_init = rng.uniform(0.1, 0.5, num_inputs) if w_init is None else np.asarray(w_init, dtype=float)
        self.int_w = bp.odeint(self.dw, method=method)
        self.int_theta = bp.odeint(self.dtheta, method=method)
        self.reset_state()

    def dw(self, w, t, x, y, theta):
        return self.eta * x * y * (y - theta)

    def dtheta(self, theta, t, y):
        return (y ** 2 - theta) / self.tau_theta

    def reset_state(self, batch_size=None, **kwargs):
        self.w = bm.Variable(bm.asarray(self.w_init))
        self.y = bm.Variable(bm.zeros(1))
        self.theta = bm.Variable(bm.zeros(1))

    def update(self, x):
        t, dt = bp.share.load("t"), bp.share.load("dt")
        y = bm.maximum(bm.dot(self.w.value, x), 0.0)
        self.w.value = self.int_w(self.w.value, t, x, y, self.theta.value[0], dt)
        self.theta.value = self.int_theta(self.theta.value, t, y, dt)
        self.y.value = bm.reshape(y, (1,))
        return y


def pattern_sequence(patterns: np.ndarray, n: int, seed: int = 0, probabilities=None) -> Tuple[np.ndarray, np.ndarray]:
    """Draw ``n`` patterns at random (rows of ``patterns``); returns the inputs and the chosen indices."""
    rng = np.random.default_rng(seed)
    index = rng.choice(len(patterns), size=n, p=probabilities)
    return np.asarray(patterns, dtype=float)[index], index
