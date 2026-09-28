"""Synapse models driven by presynaptic spike trains.

A chemical synapse turns a presynaptic spike into a transient postsynaptic
conductance. This module builds that chain in three layers, each a small
BrainPy dynamical system (state in ``bm.Variable``, equations integrated with
``bp.odeint``):

1. **Gating kinetics** -- a dimensionless variable ``g`` driven by spikes.

   * :class:`Exponential`: ``dg/dt = -g/tau``, ``g -> g + 1`` per spike.
   * :class:`DualExponential`: rise and decay time constants,
     ``g(t) = A (exp(-t/tau_d) - exp(-t/tau_r))``, peaking at
     ``t_peak = tau_r tau_d / (tau_d - tau_r) ln(tau_d / tau_r)``
     (Roth & van Rossum 2009).
   * :class:`Alpha`: ``g(t) = (t/tau) exp(1 - t/tau)``, peaking at ``t = tau``
     (Rall 1967).
   * :class:`MarkovReceptor`: two-state receptor binding driven by a square
     pulse of transmitter, ``dg/dt = alpha [T] (1 - g) - beta g``
     (Destexhe, Mainen & Sejnowski 1994), with :class:`AMPA` and
     :class:`GABAA` rate constants from Destexhe, Mainen & Sejnowski (1998).
   * :class:`NMDA`: saturating second-order kinetics with a slow decay
     (Wang 2002).

   The three linear kernels are normalized so that one isolated spike gives a
   peak of exactly 1; the synaptic weight lives in the output stage.

2. **Output** -- how ``g`` becomes a current (positive = depolarizing):
   current-based ``I = J g`` (:func:`cuba_current`), conductance-based
   ``I = gbar g (E - V)`` (:func:`coba_current`), and NMDA with the
   instantaneous Mg2+ block of Jahr & Stevens (1990),
   ``B(V) = 1 / (1 + [Mg2+]/3.57 exp(-0.062 V))`` (:func:`mg_block`).

3. **Postsynaptic neuron** -- :class:`SynapticLIF`, a leaky integrate-and-fire
   neuron (``R = 1``, so currents are in mV and ``gbar`` is relative to the
   leak conductance) receiving one synapse through one of those outputs.

Timing convention: every model adds a spike *after* integrating its step, so
a spike given at step ``k`` acts at the end of that step, ``t_{k+1}``, which is
exactly the time :func:`neuromodels.utils.run` stamps on sample ``k``.
:func:`spike_input` places a spike at time ``t`` in the step that ends at
``t``. :class:`SynapticLIF` advances its membrane before its synapse, so a
spike arriving at ``t`` starts to move ``V`` in the following step.

Differences from the BrainPy 2.8.2 built-ins (checked in
``tests/test_synapses.py``):

* ``bp.dyn.Expon``, ``bp.dyn.DualExpon`` and ``bp.dyn.NMDA`` obey the same
  equations and normalization, but default to ``exp_auto``. For the coupled
  rise/decay pairs, exponential Euler holds the rise variable fixed over a
  step, which overshoots the dual-exponential peak by about 5 % at
  ``dt = 0.1 ms, tau_r = 1 ms``. The coupled models here default to ``rk4``.
* ``bp.dyn.Alpha`` adds 1 to its rise variable per spike, so its peak is
  ``1/e``; :class:`Alpha` here peaks at 1 (a factor of ``e``).
* ``bp.dyn.AMPA`` / ``bp.dyn.GABAa`` start the transmitter pulse in the step
  that receives the spike (one step earlier than here) and time the pulse by
  comparing floating-point times; :class:`MarkovReceptor` counts it down.
* ``bp.dyn.COBA``, ``bp.dyn.CUBA`` and ``bp.dyn.MgBlock`` compute the same
  currents as the output functions here.
* In a ``bp.dyn.ProjAlignPostMg2`` projection the postsynaptic neuron decays
  ``Expon.g`` once before using it, so a CUBA EPSP built from ``Expon``,
  ``CUBA`` and ``LifRef`` is exactly ``exp(-dt/tau)`` times the one from
  :class:`SynapticLIF`; the exact EPSP lies between the two.

References
----------
Rall W (1967) J Neurophysiol 30:1138-1168.
Jahr CE, Stevens CF (1990) J Neurosci 10:3178-3182.
Destexhe A, Mainen ZF, Sejnowski TJ (1994) Neural Comput 6:14-18.
Destexhe A, Mainen ZF, Sejnowski TJ (1998) Kinetic models of synaptic
transmission. In: Koch C, Segev I (eds) Methods in Neuronal Modeling,
2nd edn. MIT Press.
Wang X-J (2002) Neuron 36:955-968.
Roth A, van Rossum MCW (2009) Modeling synapses. In: De Schutter E (ed)
Computational Modeling Methods for Neuroscientists. MIT Press.
"""

from typing import Union

import brainpy as bp
import brainpy.math as bm
import numpy as np

from neuromodels.utils import n_steps

MG_CONC = 1.2  # mM, typical extracellular [Mg2+]

ArrayLike = Union[float, np.ndarray]


# ----------------------------------------------------------------------------
# Spike input and closed-form kernels
# ----------------------------------------------------------------------------

def spike_input(times, duration: float, dt: float) -> np.ndarray:
    """Spike array for :func:`neuromodels.utils.run`.

    ``times`` is one train (1-D, giving shape ``(n_steps,)``) or a list of
    trains, one per synapse (giving ``(n_steps, n_trains)``). A spike at ``t``
    goes into the step that ends at ``t``, so the synaptic jump shows up at
    ``t`` in ``run``'s end-of-step ``ts``. Times are rounded to the grid and
    must lie in ``[dt, duration]``.
    """
    total = n_steps(duration, dt)
    several = len(times) > 0 and np.ndim(times[0]) > 0
    trains = list(times) if several else [times]
    out = np.zeros((total, len(trains)))
    for column, train in enumerate(trains):
        steps = np.rint(np.asarray(train, dtype=float) / dt).astype(int) - 1
        if np.any(steps < 0) or np.any(steps >= total):
            raise ValueError(f"spike times must lie in [dt, duration] = [{dt}, {duration}] ms")
        np.add.at(out[:, column], steps, 1.0)
    return out if several else out[:, 0]


def regular_times(rate: float, n: int, start: float) -> np.ndarray:
    """``n`` spike times (ms) at ``rate`` Hz beginning at ``start``."""
    return start + np.arange(n) * 1000.0 / rate


def exponential_kernel(t: ArrayLike, tau: float) -> np.ndarray:
    """Single-exponential response to a spike at ``t = 0`` (peak 1)."""
    t = np.asarray(t, dtype=float)
    return np.where(t >= 0, np.exp(-np.clip(t, 0, None) / tau), 0.0)


def dual_exp_peak_time(tau_rise: float, tau_decay: float) -> float:
    """Time of the dual-exponential peak, ``tau_r tau_d/(tau_d - tau_r) ln(tau_d/tau_r)``."""
    return tau_rise * tau_decay / (tau_decay - tau_rise) * np.log(tau_decay / tau_rise)


def dual_exponential_kernel(t: ArrayLike, tau_rise: float, tau_decay: float) -> np.ndarray:
    """Difference of exponentials scaled to a peak of 1, for a spike at ``t = 0``."""
    t = np.clip(np.asarray(t, dtype=float), 0, None)
    tp = dual_exp_peak_time(tau_rise, tau_decay)
    norm = np.exp(-tp / tau_decay) - np.exp(-tp / tau_rise)
    return (np.exp(-t / tau_decay) - np.exp(-t / tau_rise)) / norm


def alpha_kernel(t: ArrayLike, tau: float) -> np.ndarray:
    """Alpha function ``(t/tau) exp(1 - t/tau)``: peak 1 at ``t = tau``."""
    t = np.clip(np.asarray(t, dtype=float), 0, None)
    return t / tau * np.exp(1 - t / tau)


def markov_pulse_response(t: ArrayLike, alpha: float, beta: float, T_max: float, T_dur: float) -> np.ndarray:
    """Exact open fraction of a two-state receptor after one transmitter pulse at ``t = 0``.

    During the pulse ``g`` relaxes toward ``g_inf = alpha T/(alpha T + beta)`` with
    time constant ``1/(alpha T + beta)``; afterwards it decays with ``1/beta``.
    """
    t = np.clip(np.asarray(t, dtype=float), 0, None)
    rate_on = alpha * T_max + beta
    g_inf = alpha * T_max / rate_on
    during = g_inf * (1 - np.exp(-rate_on * np.minimum(t, T_dur)))
    return np.where(t <= T_dur, during, during * np.exp(-beta * (t - T_dur)))


# ----------------------------------------------------------------------------
# Gating kinetics
# ----------------------------------------------------------------------------

class Exponential(bp.dyn.SynDyn):
    """Single-exponential synapse: ``dg/dt = -g/tau``, ``g -> g + s`` per spike.

    ``s`` is the spike input (0/1, or a graded amplitude such as a release
    fraction). Exponential Euler (the default) solves this linear decay exactly.
    """

    def __init__(self, size=1, tau: float = 5.0, method: str = "exp_auto", name: str = None):
        super().__init__(size=size, name=name)
        self.tau = self.init_param(tau)
        self.integral = bp.odeint(self.derivative, method=method)
        self.reset_state()

    def derivative(self, g, t):
        return -g / self.tau

    def reset_state(self, batch_size=None, **kwargs):
        self.g = self.init_variable(bm.zeros, batch_size)

    def update(self, spike):
        t, dt = bp.share.load("t"), bp.share.load("dt")
        self.g.value = self.integral(self.g.value, t, dt) + spike
        return self.g.value


class DualExponential(bp.dyn.SynDyn):
    """Dual-exponential synapse with separate rise and decay time constants.

    ``dg/dt = -g/tau_d + h`` and ``dh/dt = -h/tau_r``; each spike adds
    ``A (1/tau_r - 1/tau_d)`` to ``h`` so that ``g(t) = A (e^{-t/tau_d} - e^{-t/tau_r})``
    peaks at exactly 1. The default ``rk4`` integrates the coupled pair to ~1e-6
    at ``dt = 0.1 ms``; exponential Euler would treat ``h`` as constant within a step.
    """

    def __init__(self, size=1, tau_rise: float = 1.0, tau_decay: float = 5.0, method: str = "rk4",
                 name: str = None):
        if not np.all(np.asarray(tau_rise) < np.asarray(tau_decay)):
            raise ValueError("DualExponential needs tau_rise < tau_decay (use Alpha for equal time constants)")
        super().__init__(size=size, name=name)
        self.tau_rise = self.init_param(tau_rise)
        self.tau_decay = self.init_param(tau_decay)
        tp = dual_exp_peak_time(np.asarray(tau_rise), np.asarray(tau_decay))
        peak_norm = 1.0 / (np.exp(-tp / np.asarray(tau_decay)) - np.exp(-tp / np.asarray(tau_rise)))
        self.jump = peak_norm * (1.0 / np.asarray(tau_rise) - 1.0 / np.asarray(tau_decay))
        self.integral = bp.odeint(bp.JointEq(self.dg, self.dh), method=method)
        self.reset_state()

    def dg(self, g, t, h):
        return -g / self.tau_decay + h

    def dh(self, h, t):
        return -h / self.tau_rise

    def reset_state(self, batch_size=None, **kwargs):
        self.g = self.init_variable(bm.zeros, batch_size)
        self.h = self.init_variable(bm.zeros, batch_size)

    def update(self, spike):
        t, dt = bp.share.load("t"), bp.share.load("dt")
        g, h = self.integral(self.g.value, self.h.value, t, dt)
        self.g.value = g
        self.h.value = h + self.jump * spike
        return self.g.value


class Alpha(bp.dyn.SynDyn):
    """Alpha-function synapse: ``g(t) = (t/tau) exp(1 - t/tau)`` after each spike.

    Written as ``dg/dt = (h - g)/tau``, ``dh/dt = -h/tau`` with ``h -> h + e``
    per spike, so the peak is 1 at ``t = tau`` (``bp.dyn.Alpha`` adds 1 and peaks
    at ``1/e``). The coupled pair is integrated with ``rk4`` by default.
    """

    def __init__(self, size=1, tau: float = 5.0, method: str = "rk4", name: str = None):
        super().__init__(size=size, name=name)
        self.tau = self.init_param(tau)
        self.integral = bp.odeint(bp.JointEq(self.dg, self.dh), method=method)
        self.reset_state()

    def dg(self, g, t, h):
        return (h - g) / self.tau

    def dh(self, h, t):
        return -h / self.tau

    def reset_state(self, batch_size=None, **kwargs):
        self.g = self.init_variable(bm.zeros, batch_size)
        self.h = self.init_variable(bm.zeros, batch_size)

    def update(self, spike):
        t, dt = bp.share.load("t"), bp.share.load("dt")
        g, h = self.integral(self.g.value, self.h.value, t, dt)
        self.g.value = g
        self.h.value = h + np.e * spike
        return self.g.value


class MarkovReceptor(bp.dyn.SynDyn):
    """Two-state receptor gated by a square pulse of transmitter.

    ``dg/dt = alpha [T] (1 - g) - beta g`` where ``[T] = T_max`` (mM) for ``T_dur``
    ms after each spike and 0 otherwise; ``g`` is the open fraction. The pulse is
    counted down in time rather than compared with an absolute spike time, so
    it lasts ``round(T_dur/dt)`` steps and survives restarting the clock. With
    ``[T]`` constant inside a step, exponential Euler is exact.
    """

    E = 0.0  # reversal potential (mV) used by default in SynapticLIF

    def __init__(self, size=1, alpha: float = 1.1, beta: float = 0.19, T_max: float = 1.0, T_dur: float = 1.0,
                 method: str = "exp_auto", name: str = None):
        super().__init__(size=size, name=name)
        self.alpha = self.init_param(alpha)
        self.beta = self.init_param(beta)
        self.T_max = T_max
        self.T_dur = T_dur
        self.integral = bp.odeint(self.derivative, method=method)
        self.reset_state()

    def derivative(self, g, t, T):
        return self.alpha * T * (1 - g) - self.beta * g

    def reset_state(self, batch_size=None, **kwargs):
        self.g = self.init_variable(bm.zeros, batch_size)
        self.pulse_left = self.init_variable(bm.zeros, batch_size)  # ms of transmitter still to come

    def update(self, spike):
        t, dt = bp.share.load("t"), bp.share.load("dt")
        # Half a step of slack makes the pulse length immune to float rounding.
        T = bm.where(self.pulse_left.value > 0.5 * dt, self.T_max, 0.0)
        self.g.value = self.integral(self.g.value, t, T, dt)
        left = bm.maximum(self.pulse_left.value - dt, 0.0)
        self.pulse_left.value = bm.where(spike > 0, self.T_dur, left)
        return self.g.value


class AMPA(MarkovReceptor):
    """AMPA receptor: fast excitatory kinetics (Destexhe et al. 1998), ``E = 0 mV``."""

    E = 0.0

    def __init__(self, size=1, alpha: float = 1.1, beta: float = 0.19, T_max: float = 1.0, T_dur: float = 1.0,
                 method: str = "exp_auto", name: str = None):
        super().__init__(size, alpha, beta, T_max, T_dur, method, name)


class GABAA(MarkovReceptor):
    """GABA_A receptor: fast inhibitory kinetics (Destexhe et al. 1998), ``E = -80 mV``."""

    E = -80.0

    def __init__(self, size=1, alpha: float = 5.0, beta: float = 0.18, T_max: float = 1.0, T_dur: float = 1.0,
                 method: str = "exp_auto", name: str = None):
        super().__init__(size, alpha, beta, T_max, T_dur, method, name)


class NMDA(bp.dyn.SynDyn):
    """NMDA gating with saturating second-order kinetics (Wang 2002).

    ``dg/dt = -g/tau_d + a x (1 - g)`` and ``dx/dt = -x/tau_r`` with
    ``x -> x + 1`` per spike. ``g`` never exceeds 1, so trains summate
    sublinearly. The Mg2+ block is applied at the output (:func:`nmda_current`).
    """

    E = 0.0

    def __init__(self, size=1, tau_rise: float = 2.0, tau_decay: float = 100.0, a: float = 0.5,
                 method: str = "rk4", name: str = None):
        super().__init__(size=size, name=name)
        self.tau_rise = self.init_param(tau_rise)
        self.tau_decay = self.init_param(tau_decay)
        self.a = self.init_param(a)
        self.integral = bp.odeint(bp.JointEq(self.dg, self.dx), method=method)
        self.reset_state()

    def dg(self, g, t, x):
        return -g / self.tau_decay + self.a * x * (1 - g)

    def dx(self, x, t):
        return -x / self.tau_rise

    def reset_state(self, batch_size=None, **kwargs):
        self.g = self.init_variable(bm.zeros, batch_size)
        self.x = self.init_variable(bm.zeros, batch_size)

    def update(self, spike):
        t, dt = bp.share.load("t"), bp.share.load("dt")
        g, x = self.integral(self.g.value, self.x.value, t, dt)
        self.g.value = g
        self.x.value = x + spike
        return self.g.value


# ----------------------------------------------------------------------------
# Synaptic output
# ----------------------------------------------------------------------------

def mg_block(V, mg: float = MG_CONC):
    """Unblocked fraction of NMDA channels, ``1 / (1 + [Mg]/3.57 exp(-0.062 V))`` (Jahr & Stevens 1990)."""
    exp = np.exp if isinstance(V, (int, float, np.ndarray)) else bm.exp
    return 1.0 / (1.0 + mg / 3.57 * exp(-0.062 * V))


def cuba_current(g, J: float):
    """Current-based output ``I = J g``: independent of the membrane potential."""
    return J * g


def coba_current(g, V, E: float, gbar: float = 1.0):
    """Conductance-based output ``I = gbar g (E - V)``: scaled by the driving force."""
    return gbar * g * (E - V)


def nmda_current(g, V, E: float = 0.0, gbar: float = 1.0, mg: float = MG_CONC):
    """NMDA output ``I = gbar g B(V) (E - V)`` with the instantaneous Mg2+ block ``B``."""
    return gbar * g * mg_block(V, mg) * (E - V)


# ----------------------------------------------------------------------------
# Postsynaptic neuron
# ----------------------------------------------------------------------------

class SynapticLIF(bp.dyn.NeuDyn):
    """Leaky integrate-and-fire neurons, each driven by one synapse of ``synapse``.

    ``tau dV/dt = -(V - V_rest) + I_syn + I_ext`` with ``R = 1``, so currents are
    in mV. ``output`` picks ``I_syn``: ``"cuba"`` gives ``weight * g`` (weight in
    mV), ``"coba"`` gives ``weight * g * (E - V)`` and ``"nmda"`` adds the Mg2+
    block (weight relative to the leak conductance). ``I_ext`` may differ per
    neuron to hold neurons at different baselines; ``V`` starts at the holding
    potential ``V_rest + I_ext``. After a spike ``V`` is reset and clamped for
    ``tau_ref``.

    The membrane is advanced with the synaptic state from the start of the step
    and the synapse afterwards, so a spike arriving at ``t`` acts from ``t`` on.
    Monitors: ``V``, ``spike``, ``I_syn`` and ``syn.g``.
    """

    def __init__(self, synapse: bp.dyn.SynDyn, output: str = "coba", weight: float = 0.05, E: float = None,
                 mg: float = MG_CONC, V_rest: float = -65.0, V_reset: float = -65.0, V_th: float = -50.0,
                 tau: float = 20.0, tau_ref: float = 2.0, I_ext: ArrayLike = 0.0, method: str = "exp_auto",
                 name: str = None):
        if output not in ("cuba", "coba", "nmda"):
            raise ValueError(f"output must be 'cuba', 'coba' or 'nmda', got {output!r}")
        super().__init__(size=synapse.varshape, name=name)
        self.syn = synapse
        self.output = output
        self.weight = weight
        self.E = getattr(synapse, "E", 0.0) if E is None else E
        self.mg = mg
        self.V_rest, self.V_reset, self.V_th = V_rest, V_reset, V_th
        self.tau, self.tau_ref = tau, tau_ref
        self.I_ext = bm.asarray(np.broadcast_to(np.asarray(I_ext, dtype=float), self.varshape))
        self.integral = bp.odeint(self.dV, method=method)
        self.reset_state()

    def synaptic_current(self, V, g):
        if self.output == "cuba":
            return cuba_current(g, self.weight)
        if self.output == "coba":
            return coba_current(g, V, self.E, self.weight)
        return nmda_current(g, V, self.E, self.weight, self.mg)

    def dV(self, V, t, g):
        return (-(V - self.V_rest) + self.synaptic_current(V, g) + self.I_ext) / self.tau

    def reset_state(self, batch_size=None, **kwargs):
        self.V = self.init_variable(bm.ones, batch_size)
        self.V.value = self.V.value * (self.V_rest + self.I_ext)
        self.spike = self.init_variable(lambda s: bm.zeros(s, dtype=bool), batch_size)
        self.I_syn = self.init_variable(bm.zeros, batch_size)
        self.ref_left = self.init_variable(bm.zeros, batch_size)  # ms of refractoriness left

    def update(self, pre_spike):
        t, dt = bp.share.load("t"), bp.share.load("dt")
        V = self.integral(self.V.value, t, self.syn.g.value, dt)
        V = bm.where(self.ref_left.value > 0.5 * dt, self.V_reset, V)
        spike = V >= self.V_th
        self.V.value = bm.where(spike, self.V_reset, V)
        self.spike.value = spike
        self.ref_left.value = bm.where(spike, self.tau_ref, bm.maximum(self.ref_left.value - dt, 0.0))
        g = self.syn(pre_spike)
        self.I_syn.value = self.synaptic_current(self.V.value, g)
        return spike


def psp_amplitude(V: np.ndarray, baseline) -> np.ndarray:
    """Largest deviation of ``V`` (time on axis 0) from ``baseline``, keeping its sign."""
    deviation = np.asarray(V) - np.asarray(baseline)
    index = np.argmax(np.abs(deviation), axis=0)
    return np.take_along_axis(deviation, index[None], axis=0)[0]
