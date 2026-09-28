"""Neuron models written out from their equations with BrainPy (course Days 2–3).

Each class subclasses ``bp.dyn.NeuDyn``, keeps its state in ``bm.Variable``s
and hands its ODEs to ``bp.odeint`` — the pattern the course teaches before
it moves on to BrainPy's built-in models. ``tests/test_neurons.py`` checks
every class against the matching built-in model and against closed-form
results from ``neuromodels.analysis``.

All models run in ms and mV. Currents follow each original paper: µA/cm² for
HH, pA for AdEx, and a dimensionless input for the others.

Models
------
HH          Hodgkin & Huxley (1952), in the modern −65 mV resting convention
LIF         leaky integrate-and-fire (Lapicque 1907; see Abbott 1999)
QIF         quadratic integrate-and-fire (Ermentrout & Kopell 1986; Latham et al. 2000)
ExpIF       exponential integrate-and-fire (Fourcaud-Trocmé et al. 2003)
AdEx        adaptive exponential integrate-and-fire (Brette & Gerstner 2005)
Izhikevich  Izhikevich (2003), with the firing-pattern presets of its Fig. 2

Unlike BrainPy's built-in integrate-and-fire models, which start at 0 mV,
every class here starts at rest unless told otherwise.
"""

import brainpy as bp
import brainpy.math as bm

# Izhikevich (2003), Fig. 2: (a, b, c, d) for common cortical and thalamic cell classes.
IZHIKEVICH_PRESETS = {
    "RS": dict(a=0.02, b=0.2, c=-65.0, d=8.0),     # regular spiking (excitatory, adapting)
    "IB": dict(a=0.02, b=0.2, c=-55.0, d=4.0),     # intrinsically bursting
    "CH": dict(a=0.02, b=0.2, c=-50.0, d=2.0),     # chattering (fast rhythmic bursts)
    "FS": dict(a=0.1, b=0.2, c=-65.0, d=2.0),      # fast spiking (inhibitory interneuron)
    "LTS": dict(a=0.02, b=0.25, c=-65.0, d=2.0),   # low-threshold spiking
    "TC": dict(a=0.02, b=0.25, c=-65.0, d=0.05),   # thalamo-cortical
    "RZ": dict(a=0.1, b=0.26, c=-65.0, d=2.0),     # resonator
}


def _state(shape, value, dtype=None) -> bm.Variable:
    return bm.Variable(bm.ones(shape, dtype=dtype) * value)


class HH(bp.dyn.NeuDyn):
    r"""Hodgkin–Huxley model of the squid giant axon.

    .. math::
        C \frac{dV}{dt} = -\bar g_{Na} m^3 h (V - E_{Na}) - \bar g_K n^4 (V - E_K) - g_L (V - E_L) + I

        \frac{dx}{dt} = \alpha_x(V) (1 - x) - \beta_x(V)\, x, \qquad x \in \{m, h, n\}

    Units: mV, ms, mS/cm², µF/cm², µA/cm². With Hodgkin & Huxley's leak
    ``gL = 0.3`` and ``E_L = -54.387`` mV, rest sits at −65 mV. BrainPy 2.8.2's
    ``bp.dyn.HH`` defaults to ``gL = 0.03``, which moves rest to −70.7 mV and
    removes anode-break excitation (see ``tests/test_hodgkin_huxley.py``). A
    spike is an upward crossing of ``V_th``: a detection level, not a
    biophysical threshold. Gates start at their steady state for ``V_init``.
    """

    def __init__(self, size, ENa=50.0, gNa=120.0, EK=-77.0, gK=36.0, EL=-54.387, gL=0.3, C=1.0,
                 V_th=20.0, V_init=-65.0, method="exp_auto", **kwargs):
        super().__init__(size=size, method=method, **kwargs)
        self.ENa, self.gNa, self.EK, self.gK, self.EL, self.gL, self.C = ENa, gNa, EK, gK, EL, gL, C
        self.V_th, self.V_init = V_th, V_init
        self.integral = bp.odeint(bp.JointEq(self.dV, self.dm, self.dh, self.dn), method=method)
        self.reset_state()

    # Opening (alpha) and closing (beta) rates in 1/ms. m_alpha and n_alpha have a
    # removable 0/0 at V = -40 and -55 mV; exprel(x) = (e^x - 1)/x sidesteps it.
    def m_alpha(self, V):
        return 1.0 / bm.exprel(-(V + 40.0) / 10.0)

    def m_beta(self, V):
        return 4.0 * bm.exp(-(V + 65.0) / 18.0)

    def h_alpha(self, V):
        return 0.07 * bm.exp(-(V + 65.0) / 20.0)

    def h_beta(self, V):
        return 1.0 / (1.0 + bm.exp(-(V + 35.0) / 10.0))

    def n_alpha(self, V):
        return 0.1 / bm.exprel(-(V + 55.0) / 10.0)

    def n_beta(self, V):
        return 0.125 * bm.exp(-(V + 65.0) / 80.0)

    def gate_kinetics(self, V):
        """Steady states ``(m_inf, h_inf, n_inf)`` and time constants ``(tau_m, tau_h, tau_n)`` at ``V``."""
        rates = [(self.m_alpha(V), self.m_beta(V)), (self.h_alpha(V), self.h_beta(V)),
                 (self.n_alpha(V), self.n_beta(V))]
        steady = tuple(alpha / (alpha + beta) for alpha, beta in rates)
        taus = tuple(1.0 / (alpha + beta) for alpha, beta in rates)
        return steady, taus

    def currents(self, V, m, h, n):
        """Outward ionic currents ``(I_Na, I_K, I_L)`` in µA/cm²."""
        I_Na = self.gNa * m ** 3 * h * (V - self.ENa)
        I_K = self.gK * n ** 4 * (V - self.EK)
        I_L = self.gL * (V - self.EL)
        return I_Na, I_K, I_L

    def dm(self, m, t, V):
        return self.m_alpha(V) * (1.0 - m) - self.m_beta(V) * m

    def dh(self, h, t, V):
        return self.h_alpha(V) * (1.0 - h) - self.h_beta(V) * h

    def dn(self, n, t, V):
        return self.n_alpha(V) * (1.0 - n) - self.n_beta(V) * n

    def dV(self, V, t, m, h, n, I):
        I_Na, I_K, I_L = self.currents(V, m, h, n)
        return (I - I_Na - I_K - I_L) / self.C

    def reset_state(self, batch_size=None, **kwargs):
        self.V = _state(self.varshape, self.V_init)
        (m, h, n), _ = self.gate_kinetics(self.V.value)
        self.m, self.h, self.n = bm.Variable(m), bm.Variable(h), bm.Variable(n)
        self.spike = _state(self.varshape, False, dtype=bool)

    def update(self, I=0.0):
        t, dt = bp.share.load("t"), bp.share.load("dt")
        V, m, h, n = self.integral(self.V.value, self.m.value, self.h.value, self.n.value, t, I, dt)
        self.spike.value = bm.logical_and(self.V.value < self.V_th, V >= self.V_th)
        self.V.value, self.m.value, self.h.value, self.n.value = V, m, h, n
        return self.spike.value


class LIF(bp.dyn.NeuDyn):
    r"""Leaky integrate-and-fire neuron with an absolute refractory period.

    .. math:: \tau \frac{dV}{dt} = -(V - V_{rest}) + R I

    When ``V`` reaches ``V_th`` the neuron spikes; ``V`` resets to ``V_reset``
    and stays there for ``t_ref`` ms. Defaults match ``bp.dyn.Lif``.
    """

    def __init__(self, size, V_rest=0.0, V_reset=-5.0, V_th=20.0, tau=10.0, R=1.0, t_ref=0.0,
                 V_init=None, method="exp_auto", **kwargs):
        super().__init__(size=size, method=method, **kwargs)
        self.V_rest, self.V_reset, self.V_th, self.tau, self.R, self.t_ref = V_rest, V_reset, V_th, tau, R, t_ref
        self.V_init = V_rest if V_init is None else V_init
        self.integral = bp.odeint(self.dV, method=method)
        self.reset_state()

    def dV(self, V, t, I):
        return (-(V - self.V_rest) + self.R * I) / self.tau

    def reset_state(self, batch_size=None, **kwargs):
        self.V = _state(self.varshape, self.V_init)
        self.t_spike = _state(self.varshape, -1e7)  # time of the last spike (end of its step)
        self.spike = _state(self.varshape, False, dtype=bool)

    def update(self, I=0.0):
        t, dt = bp.share.load("t"), bp.share.load("dt")
        V = self.integral(self.V.value, t, I, dt)
        # Hold V during the refractory period. Comparing at mid-step keeps float
        # rounding in t from adding or dropping a step: exactly t_ref/dt steps are held.
        refractory = t + 0.5 * dt < self.t_spike.value + self.t_ref
        V = bm.where(refractory, self.V.value, V)
        spike = V >= self.V_th
        self.t_spike.value = bm.where(spike, t + dt, self.t_spike.value)
        self.V.value = bm.where(spike, self.V_reset, V)
        self.spike.value = spike
        return spike


class QIF(bp.dyn.NeuDyn):
    r"""Quadratic integrate-and-fire neuron, the normal form of a saddle-node on invariant circle.

    .. math:: \tau \frac{dV}{dt} = c (V - V_{rest})(V - V_c) + R I

    ``V_c`` is the critical voltage: above it, ``V`` runs away to the spike
    cutoff ``V_th``. Defaults match ``bp.dyn.QuaIF``.
    """

    def __init__(self, size, V_rest=-65.0, V_reset=-68.0, V_th=-30.0, V_c=-50.0, c=0.07, R=1.0, tau=10.0,
                 V_init=None, method="exp_auto", **kwargs):
        super().__init__(size=size, method=method, **kwargs)
        self.V_rest, self.V_reset, self.V_th, self.V_c, self.c, self.R, self.tau = V_rest, V_reset, V_th, V_c, c, R, tau
        self.V_init = V_rest if V_init is None else V_init
        self.integral = bp.odeint(self.dV, method=method)
        self.reset_state()

    def dV(self, V, t, I):
        return (self.c * (V - self.V_rest) * (V - self.V_c) + self.R * I) / self.tau

    def reset_state(self, batch_size=None, **kwargs):
        self.V = _state(self.varshape, self.V_init)
        self.spike = _state(self.varshape, False, dtype=bool)

    def update(self, I=0.0):
        t, dt = bp.share.load("t"), bp.share.load("dt")
        V = self.integral(self.V.value, t, I, dt)
        spike = V >= self.V_th
        self.V.value = bm.where(spike, self.V_reset, V)
        self.spike.value = spike
        return spike


class ExpIF(bp.dyn.NeuDyn):
    r"""Exponential integrate-and-fire neuron.

    .. math:: \tau \frac{dV}{dt} = -(V - V_{rest}) + \Delta_T e^{(V - V_T)/\Delta_T} + R I

    ``V_T`` is the soft threshold and ``Delta_T`` its sharpness; ``V_th`` is only
    the numerical cutoff where the run-away is declared a spike. The cutoff
    must sit well above ``V_T``: BrainPy 2.8.2's ``bp.dyn.ExpIF`` defaults to
    −55 mV (1.4 ``Delta_T`` above ``V_T``, although its docstring says −30 mV),
    which clips the upswing. This class uses −30 mV.
    """

    def __init__(self, size, V_rest=-65.0, V_reset=-68.0, V_th=-30.0, V_T=-59.9, delta_T=3.48, R=1.0, tau=10.0,
                 V_init=None, method="exp_auto", **kwargs):
        super().__init__(size=size, method=method, **kwargs)
        self.V_rest, self.V_reset, self.V_th, self.V_T, self.delta_T = V_rest, V_reset, V_th, V_T, delta_T
        self.R, self.tau = R, tau
        self.V_init = V_rest if V_init is None else V_init
        self.integral = bp.odeint(self.dV, method=method)
        self.reset_state()

    def dV(self, V, t, I):
        spike_current = self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
        return (-(V - self.V_rest) + spike_current + self.R * I) / self.tau

    def reset_state(self, batch_size=None, **kwargs):
        self.V = _state(self.varshape, self.V_init)
        self.spike = _state(self.varshape, False, dtype=bool)

    def update(self, I=0.0):
        t, dt = bp.share.load("t"), bp.share.load("dt")
        V = self.integral(self.V.value, t, I, dt)
        spike = V >= self.V_th
        self.V.value = bm.where(spike, self.V_reset, V)
        self.spike.value = spike
        return spike


class AdEx(bp.dyn.NeuDyn):
    r"""Adaptive exponential integrate-and-fire neuron, in physical units.

    .. math::
        C \frac{dV}{dt} = -g_L (V - E_L) + g_L \Delta_T e^{(V - V_T)/\Delta_T} - w + I

        \tau_w \frac{dw}{dt} = a (V - E_L) - w

    When ``V`` reaches ``V_cut``: ``V ← V_reset`` and ``w ← w + b``. Units: pF,
    nS, mV, ms, pA. Defaults are the regular-spiking fit of Brette & Gerstner
    (2005), with the cutoff at ``V_T + 5 Delta_T``. Dividing the V equation by
    ``g_L`` gives BrainPy's ``bp.dyn.AdExIF`` form: ``tau = C / g_L`` and ``R = 1 / g_L``.
    """

    def __init__(self, size, C=281.0, g_L=30.0, E_L=-70.6, V_T=-50.4, delta_T=2.0, a=4.0, tau_w=144.0, b=80.5,
                 V_reset=-70.6, V_cut=-40.4, V_init=None, method="exp_auto", **kwargs):
        super().__init__(size=size, method=method, **kwargs)
        self.C, self.g_L, self.E_L, self.V_T, self.delta_T = C, g_L, E_L, V_T, delta_T
        self.a, self.tau_w, self.b, self.V_reset, self.V_cut = a, tau_w, b, V_reset, V_cut
        self.V_init = E_L if V_init is None else V_init
        self.integral = bp.odeint(bp.JointEq(self.dV, self.dw), method=method)
        self.reset_state()

    def dV(self, V, t, w, I):
        spike_current = self.g_L * self.delta_T * bm.exp((V - self.V_T) / self.delta_T)
        return (-self.g_L * (V - self.E_L) + spike_current - w + I) / self.C

    def dw(self, w, t, V):
        return (self.a * (V - self.E_L) - w) / self.tau_w

    def reset_state(self, batch_size=None, **kwargs):
        self.V = _state(self.varshape, self.V_init)
        self.w = _state(self.varshape, self.a * (self.V_init - self.E_L))  # adaptation at its steady state
        self.spike = _state(self.varshape, False, dtype=bool)

    def update(self, I=0.0):
        t, dt = bp.share.load("t"), bp.share.load("dt")
        V, w = self.integral(self.V.value, self.w.value, t, I, dt)
        spike = V >= self.V_cut
        self.V.value = bm.where(spike, self.V_reset, V)
        self.w.value = bm.where(spike, w + self.b, w)
        self.spike.value = spike
        return spike


class Izhikevich(bp.dyn.NeuDyn):
    r"""Izhikevich (2003) simple model.

    .. math::
        \frac{dV}{dt} = 0.04 V^2 + 5 V + 140 - u + I, \qquad \frac{du}{dt} = a (b V - u)

    When ``V`` reaches ``V_peak``: ``V ← c`` and ``u ← u + d``. With
    ``V_init=None`` the neuron starts at its resting fixed point for ``I = 0``
    (or at ``c`` if none exists), so presets begin without a transient.
    """

    def __init__(self, size, a=0.02, b=0.2, c=-65.0, d=8.0, V_peak=30.0, V_init=None, u_init=None,
                 method="exp_auto", **kwargs):
        super().__init__(size=size, method=method, **kwargs)
        self.a, self.b, self.c, self.d, self.V_peak = a, b, c, d, V_peak
        if V_init is None:
            # Lower root of 0.04 V^2 + (5 - b) V + 140 = 0 (V' = u' = 0 with I = 0).
            disc = (5.0 - b) ** 2 - 4 * 0.04 * 140.0
            V_init = (-(5.0 - b) - disc ** 0.5) / 0.08 if disc >= 0 else c
        self.V_init = V_init
        self.u_init = b * V_init if u_init is None else u_init
        self.integral = bp.odeint(bp.JointEq(self.dV, self.du), method=method)
        self.reset_state()

    @classmethod
    def from_preset(cls, name: str, size=1, **kwargs) -> "Izhikevich":
        """Build a neuron from one of :data:`IZHIKEVICH_PRESETS` (e.g. ``"RS"``, ``"CH"``)."""
        return cls(size, **IZHIKEVICH_PRESETS[name], **kwargs)

    def dV(self, V, t, u, I):
        return 0.04 * V * V + 5.0 * V + 140.0 - u + I

    def du(self, u, t, V):
        return self.a * (self.b * V - u)

    def reset_state(self, batch_size=None, **kwargs):
        self.V = _state(self.varshape, self.V_init)
        self.u = _state(self.varshape, self.u_init)
        self.spike = _state(self.varshape, False, dtype=bool)

    def update(self, I=0.0):
        t, dt = bp.share.load("t"), bp.share.load("dt")
        V, u = self.integral(self.V.value, self.u.value, t, I, dt)
        spike = V >= self.V_peak
        self.V.value = bm.where(spike, self.c, V)
        self.u.value = bm.where(spike, u + self.d, u)
        self.spike.value = spike
        return spike
