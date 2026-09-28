import numpy as np
import pytest

from neuromodels.basics import convergence_order, decay_trajectory, integrate


@pytest.mark.parametrize("method, order", [("euler", 1), ("heun2", 2), ("rk4", 4)])
def test_convergence_order_matches_theory(method, order):
    assert convergence_order(method) == pytest.approx(order, abs=0.15)


def test_forward_euler_is_unstable_beyond_two_tau():
    # Euler multiplies x by (1 - dt/tau) each step: it decays only while |1 - dt/tau| < 1.
    _, stable = decay_trajectory("euler", dt=1.5)
    _, unstable = decay_trajectory("euler", dt=2.5)
    assert abs(stable[-1]) < 1e-3
    assert abs(unstable[-1]) > 1.0


def test_exponential_euler_is_exact_for_linear_decay():
    ts, xs = decay_trajectory("exp_euler", dt=2.5)
    np.testing.assert_allclose(xs, np.exp(-ts), rtol=1e-12)


def test_integrator_runner_reports_end_of_step_times():
    ts, xs = integrate(lambda x, t: -x, 1.0, dt=0.5, duration=2.0, method="exp_euler")
    np.testing.assert_allclose(ts, [0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(xs, np.exp(-ts), rtol=1e-12)
