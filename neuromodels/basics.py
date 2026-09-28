"""Day 1: what a numerical integrator does to a model.

Every model in this repo is a system of ODEs handed to ``bp.odeint``. Before
trusting a spike train it pays to know how the integrator's error scales with
the step size and when the integrator stops being stable:

* :func:`logistic_error` integrates the logistic equation, whose exact solution
  is known, so each method's global error can be measured directly.
* :func:`convergence_order` fits the slope of log(error) against log(dt):
  about 1 for forward Euler, 2 for Heun's method, 4 for classical RK4.
* :func:`decay_trajectory` integrates a fast linear decay. Forward Euler
  oscillates and then blows up once ``dt > 2 tau``; exponential Euler is exact
  for linear equations at any ``dt``, which is why BrainPy's neuron models
  default to ``exp_auto``.

``bp.IntegratorRunner`` stamps its samples with end-of-step times (the first
sample sits at ``t = dt``), unlike ``bp.DSRunner``; see ``neuromodels.utils``.
"""

from typing import Sequence, Tuple

import brainpy as bp
import numpy as np


def integrate(f, x0: float, dt: float, duration: float, method: str) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the scalar ODE ``dx/dt = f(x, t)`` and return ``(ts, xs)`` at end-of-step times."""
    integral = bp.odeint(f, method=method)
    runner = bp.IntegratorRunner(integral, monitors=["x"], inits={"x": x0}, dt=dt, progress_bar=False)
    runner.run(duration)
    return np.asarray(runner.mon.ts), np.asarray(runner.mon.x).ravel()


def logistic_exact(t: np.ndarray, x0: float = 0.1) -> np.ndarray:
    """Exact solution of dx/dt = x (1 - x)."""
    return 1.0 / (1.0 + (1.0 / x0 - 1.0) * np.exp(-t))


def logistic_error(method: str, dt: float, duration: float = 10.0, x0: float = 0.1) -> float:
    """Largest absolute error along the trajectory of dx/dt = x (1 - x)."""
    ts, xs = integrate(lambda x, t: x * (1.0 - x), x0, dt, duration, method)
    return float(np.max(np.abs(xs - logistic_exact(ts, x0))))


def convergence_order(method: str, dts: Sequence[float] = (0.4, 0.2, 0.1, 0.05)) -> float:
    """Slope of log(error) vs log(dt) on the logistic equation (the method's order)."""
    errors = [logistic_error(method, dt) for dt in dts]
    slope, _ = np.polyfit(np.log(dts), np.log(errors), 1)
    return float(slope)


def decay_trajectory(method: str, dt: float, tau: float = 1.0, duration: float = 20.0) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate dx/dt = -x / tau from x = 1; the exact solution is exp(-t / tau)."""
    return integrate(lambda x, t: -x / tau, 1.0, dt, duration, method)
