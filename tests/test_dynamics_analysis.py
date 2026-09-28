import contextlib
import io

import brainpy as bp
import numpy as np
import pytest

from neuromodels.analysis import (adex_rest_loss, fi_curve, izhikevich_fixed_points, izhikevich_jacobian,
                                  izhikevich_rest_loss)
from neuromodels.neurons import AdEx, Izhikevich
from neuromodels.utils import run, step_current


def test_regular_spiking_loses_rest_through_hopf():
    loss = izhikevich_rest_loss(a=0.02, b=0.2)
    assert loss.kind == "hopf"
    assert loss.current == pytest.approx(3.7975)
    # At a Hopf point the Jacobian has a purely imaginary pair of eigenvalues.
    eig = np.linalg.eigvals(izhikevich_jacobian(loss.voltage, a=0.02, b=0.2))
    np.testing.assert_allclose(eig.real, 0.0, atol=1e-12)
    assert np.all(np.abs(eig.imag) > 0)


def test_weak_recovery_coupling_gives_saddle_node():
    # With b < a the determinant vanishes first: rest ends at the fold, I = (5 - b)^2 / 0.16 - 140.
    loss = izhikevich_rest_loss(a=0.02, b=-0.1)
    assert loss.kind == "saddle-node"
    assert loss.current == pytest.approx(5.1 ** 2 / 0.16 - 140.0)
    # Rest and saddle exist just below the fold and vanish just above it.
    assert len(izhikevich_fixed_points(loss.current - 1e-3, b=-0.1)) == 2
    assert izhikevich_fixed_points(loss.current + 1e-3, b=-0.1) == []


@pytest.mark.parametrize("offset, fires", [(-0.03, False), (0.03, True)])
def test_simulation_confirms_hopf_point(offset, fires):
    current = izhikevich_rest_loss(a=0.02, b=0.2).current + offset
    V0, u0 = izhikevich_fixed_points(current, b=0.2)[0]
    out = run(Izhikevich(1, V_init=V0 + 0.1, u_init=u0), step_current(3000.0, 0.05, current), 0.05)
    assert bool(out["spike"].any()) == fires


def test_onset_rate_reflects_bifurcation_type():
    # Past a saddle-node the trajectory crawls through the ghost of the fixed points, so the
    # period diverges and rates start from zero. Past a Hopf point it joins an oscillation
    # whose period stays finite, so the rate jumps to a non-zero value.
    dt, offset = 0.05, 0.005
    rates = {}
    for kind, pars in [("saddle-node", dict(a=0.02, b=-0.1, c=-55.0, d=6.0)), ("hopf", dict(a=0.02, b=0.2, c=-65.0, d=8.0))]:
        current = izhikevich_rest_loss(pars["a"], pars["b"]).current + offset
        V0, u0 = izhikevich_fixed_points(current - 2 * offset, pars["b"])[0]  # rest, just before the bifurcation
        out = run(Izhikevich(1, V_init=V0, u_init=u0, **pars), step_current(8000.0, dt, current), dt)
        times = out["ts"][out["spike"][:, 0]]
        rates[kind] = 1000.0 / np.mean(np.diff(times[-3:]))
    assert rates["saddle-node"] < 1.5
    assert rates["hopf"] > 4.0


def test_brainpy_phase_plane_finds_the_same_fixed_points():
    a, b, current = 0.02, 0.2, 3.7
    integral = bp.odeint(bp.JointEq(lambda V, t, u, I: 0.04 * V * V + 5 * V + 140 - u + I,
                                    lambda u, t, V: a * (b * V - u)), method="rk4")
    analyzer = bp.analysis.PhasePlane2D(model=integral, target_vars={"V": [-80.0, -40.0], "u": [-18.0, -8.0]},
                                        pars_update={"I": current}, resolutions={"V": 0.05, "u": 0.05})
    # The analyzer narrates every step; it seeds its fixed-point search with nullcline points.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        analyzer.plot_nullcline(with_plot=False)
        found = np.asarray(analyzer.plot_fixed_point(with_plot=False, with_return=True))
    # The analyzer solves for V on its u grid (0.05 apart), so it is only grid-accurate:
    # its saddle sits at u = -11.45 exactly, 0.005 mV from the true root.
    np.testing.assert_allclose(sorted(found[:, 0]), [V for V, _ in izhikevich_fixed_points(current, b)], atol=0.01)


def test_adex_regular_spiking_rheobase():
    # a = 4 nS exceeds C / tau_w = 1.95 nS, so rest ends in a Hopf bifurcation near 627 pA.
    loss = adex_rest_loss()
    assert loss.kind == "hopf"
    # Skip the first second: a step from rest fires one spike before adaptation catches up.
    rates = fi_curve(lambda n: AdEx(n), [loss.current - 5.0, loss.current + 5.0],
                     duration=4000.0, dt=0.05, transient=1000.0)
    assert rates[0] == 0.0 and rates[1] > 0.0
