"""Closed-form results for the reduced neuron models (course Day 3).

A simulation is only as trustworthy as the checks behind it. These functions
give exact answers that ``tests/`` compares the BrainPy models against and
that ``scripts/04_dynamics_analysis.py`` draws:

* :func:`lif_rate` — firing rate of the LIF neuron under constant input.
* :func:`qif_rheobase`, :func:`expif_rheobase` — the current at which the
  resting state and the threshold (a saddle) merge: a saddle-node bifurcation.
* :func:`izhikevich_rest_loss`, :func:`adex_rest_loss` — in 2-D models with a
  linear recovery variable, rest loses stability through a saddle-node when
  the recovery coupling is weak and through an Andronov–Hopf bifurcation when
  it is strong. Both "regular spiking" fits sit on the Hopf side: the
  Izhikevich RS preset (b = 0.2 > a = 0.02) starts firing at I ≈ 3.80, just
  before its saddle-node at I = 4.
* :func:`hh_rest_state`, :func:`hh_jacobian` — the Hodgkin–Huxley rest state stays
  linearly stable up to I ≈ 9.78 µA/cm² (a subcritical Hopf), although a step from
  rest already fires from 6.3 µA/cm²: in between, rest and firing coexist.
* :func:`fi_curve`, :func:`adaptation_index` — measurements from simulations.
"""

from typing import Callable, List, NamedTuple, Sequence, Tuple

import numpy as np

from neuromodels.neurons import HH
from neuromodels.utils import firing_rate, run, step_current


class RestLoss(NamedTuple):
    """How and where the resting state loses stability as constant input grows."""

    kind: str       # "saddle-node" or "hopf"
    current: float  # input at which rest becomes unstable (or vanishes)
    voltage: float  # resting voltage at that input


def lif_rate(I, V_rest=0.0, V_reset=-5.0, V_th=20.0, tau=10.0, R=1.0, t_ref=0.0) -> np.ndarray:
    """Firing rate in Hz of the LIF neuron under constant input ``I`` (time in ms).

    Between spikes V relaxes toward ``V_inf = V_rest + R I``; the neuron fires
    only if ``V_inf > V_th``, with period ``t_ref + tau ln((V_inf - V_reset) / (V_inf - V_th))``.
    """
    v_inf = V_rest + R * np.asarray(I, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        period = t_ref + tau * np.log((v_inf - V_reset) / (v_inf - V_th))
        return np.where(v_inf > V_th, 1000.0 / period, 0.0)


def qif_rheobase(V_rest=-65.0, V_c=-50.0, c=0.07, R=1.0) -> float:
    """Smallest constant input that makes the QIF neuron fire.

    Fixed points solve ``c (V - V_rest)(V - V_c) + R I = 0``; the two roots (rest
    and threshold) merge when the discriminant vanishes, at ``I = c (V_c - V_rest)^2 / (4 R)``.
    """
    return c * (V_c - V_rest) ** 2 / (4.0 * R)


def expif_rheobase(V_rest=-65.0, V_T=-59.9, delta_T=3.48, R=1.0) -> float:
    """Smallest constant input that makes the ExpIF neuron fire.

    The steady-state I–V curve ``(V - V_rest - Delta_T e^{(V - V_T)/Delta_T}) / R`` peaks
    at ``V = V_T``; its maximum ``(V_T - V_rest - Delta_T) / R`` is the rheobase.
    """
    return (V_T - V_rest - delta_T) / R


def izhikevich_fixed_points(I: float, b: float) -> List[Tuple[float, float]]:
    """Fixed points ``(V, u)`` of the Izhikevich model at constant input ``I``, lowest V first.

    ``u = b V`` and ``0.04 V^2 + (5 - b) V + 140 + I = 0``: two points (rest and
    saddle), one at the saddle-node, none above it.
    """
    disc = (5.0 - b) ** 2 - 0.16 * (140.0 + I)
    if disc < 0:
        return []
    roots = sorted({(-(5.0 - b) - np.sqrt(disc)) / 0.08, (-(5.0 - b) + np.sqrt(disc)) / 0.08})
    return [(float(V), float(b * V)) for V in roots]


def izhikevich_jacobian(V: float, a: float, b: float) -> np.ndarray:
    """Jacobian of (dV/dt, du/dt) with respect to (V, u)."""
    return np.array([[0.08 * V + 5.0, -1.0], [a * b, -a]])


def izhikevich_rest_loss(a: float, b: float) -> RestLoss:
    """Bifurcation that ends the resting state of the Izhikevich model.

    With ``f(V) = 0.04 V^2 + 5 V + 140`` the Jacobian has trace ``f'(V) - a`` and
    determinant ``a (b - f'(V))``. Rising input pushes the resting V up, and so
    f'(V) up. If ``b < a`` the determinant reaches zero first (saddle-node at
    f'(V) = b); if ``b > a`` the trace reaches zero first while the determinant
    is still positive (Andronov–Hopf at f'(V) = a).
    """
    V = ((b if b < a else a) - 5.0) / 0.08
    current = -(0.04 * V ** 2 + (5.0 - b) * V + 140.0)
    return RestLoss("saddle-node" if b < a else "hopf", float(current), float(V))


def adex_rest_loss(C=281.0, g_L=30.0, E_L=-70.6, V_T=-50.4, delta_T=2.0, a=4.0, tau_w=144.0) -> RestLoss:
    """Bifurcation that ends the resting state of the AdEx model (Touboul & Brette 2008).

    Same argument as :func:`izhikevich_rest_loss` with ``F(V) = -g_L (V - E_L) + g_L
    Delta_T e^{(V - V_T)/Delta_T}``: saddle-node where ``F'(V) = a``, Hopf where
    ``F'(V) = C / tau_w``, whichever comes first — Hopf exactly when ``a > C / tau_w``.
    """
    hopf = a > C / tau_w
    slope = C / tau_w if hopf else a
    V = V_T + delta_T * np.log(1.0 + slope / g_L)
    current = (g_L + a) * (V - E_L) - g_L * delta_T * np.exp((V - V_T) / delta_T)
    return RestLoss("hopf" if hopf else "saddle-node", float(current), float(V))


def hh_rest_state(current: float, **params) -> np.ndarray:
    """Resting state ``(V, m, h, n)`` of the HH model under constant input ``current``.

    Solves ``I = I_Na + I_K + I_L`` with every gate at its steady state, by bisection:
    the steady-state I–V curve rises monotonically, so the rest state is unique.
    """
    hh = HH(1, **params)

    def steady_current(V):
        (m, h, n), _ = hh.gate_kinetics(V)
        return float(sum(np.asarray(x) for x in hh.currents(V, m, h, n)))

    low, high = -100.0, 0.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        low, high = (middle, high) if steady_current(middle) < current else (low, middle)
    V = 0.5 * (low + high)
    (m, h, n), _ = hh.gate_kinetics(V)
    return np.array([V, float(m), float(h), float(n)])


def hh_jacobian(state: np.ndarray, current: float, eps: float = 1e-6, **params) -> np.ndarray:
    """Jacobian of the HH equations with respect to ``(V, m, h, n)``, by central differences."""
    hh = HH(1, **params)

    def rhs(x):
        V, m, h, n = x
        return np.array([float(hh.dV(V, 0.0, m, h, n, current)), float(hh.dm(m, 0.0, V)),
                         float(hh.dh(h, 0.0, V)), float(hh.dn(n, 0.0, V))])

    columns = [(rhs(state + step) - rhs(state - step)) / (2 * eps) for step in np.eye(4) * eps]
    return np.stack(columns, axis=1)


def fi_curve(make_model: Callable[[int], object], currents: Sequence[float], duration: float = 1000.0,
             dt: float = 0.05, transient: float = 200.0) -> np.ndarray:
    """Steady firing rate (Hz) for each constant current, one neuron per value in a single run.

    ``make_model(n)`` must return a population of ``n`` neurons with a ``spike``
    variable. Spikes during the first ``transient`` ms are ignored.
    """
    currents = np.asarray(currents, dtype=float)
    out = run(make_model(len(currents)), step_current(duration, dt, currents), dt, monitors=("spike",))
    return firing_rate(out["spike"], dt, start=transient)


def adaptation_index(times: np.ndarray) -> float:
    """Mean normalized change between consecutive inter-spike intervals.

    ``A = mean((ISI_{k+1} - ISI_k) / (ISI_{k+1} + ISI_k))``: 0 for a regular train,
    positive when firing slows down (adaptation), negative when it speeds up.
    """
    intervals = np.diff(np.asarray(times))
    if len(intervals) < 2:
        return float("nan")
    return float(np.mean(np.diff(intervals) / (intervals[1:] + intervals[:-1])))
