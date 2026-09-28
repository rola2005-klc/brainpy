import brainpy as bp
import numpy as np
import pytest

from neuromodels.analysis import fi_curve, hh_jacobian, hh_rest_state
from neuromodels.neurons import HH
from neuromodels.utils import n_steps, run, step_current

DT = 0.01


def test_matches_brainpy_builtin():
    # Same equations, parameters, and integrator, so traces agree to rounding error.
    inputs = step_current(100.0, DT, 10.0, onset=10.0)
    mine = run(HH(1), inputs, DT)
    builtin = run(bp.dyn.HH(1, gL=0.3, V_initializer=bp.init.Constant(-65.0)), inputs, DT)
    np.testing.assert_allclose(mine["V"], builtin["V"], atol=1e-9)
    assert mine["spike"].sum() == builtin["spike"].sum() > 0


def test_rests_at_minus_65_mV():
    # E_L = -54.387 mV is chosen so the ionic currents cancel at -65 mV when gL = 0.3.
    out = run(HH(1), np.zeros(n_steps(200.0, DT)), DT)
    assert out["V"][-1, 0] == pytest.approx(-65.0, abs=0.01)


def test_brainpy_default_leak_moves_rest():
    # bp.dyn.HH defaults to gL = 0.03, ten times Hodgkin & Huxley's value.
    out = run(bp.dyn.HH(1, V_initializer=bp.init.Constant(-65.0)), np.zeros(n_steps(200.0, DT)), DT)
    assert out["V"][-1, 0] == pytest.approx(-70.68, abs=0.05)


def test_type_ii_excitability():
    # HH starts firing at a finite rate (Hodgkin's class 2): the F-I curve jumps from 0 to ~50 Hz.
    currents = np.arange(5.0, 7.51, 0.1)
    rates = fi_curve(lambda n: HH(n), currents, duration=1000.0, dt=DT, transient=200.0)
    onset = int(np.argmax(rates > 0))
    assert currents[onset] == pytest.approx(6.3, abs=0.15)
    assert rates[onset] > 40.0


def test_rest_loses_stability_near_9_78():
    # The largest eigenvalue of the Jacobian at rest crosses zero between 9.75 and 9.80 µA/cm².
    def growth(current):
        return np.linalg.eigvals(hh_jacobian(hh_rest_state(current), current)).real.max()

    assert growth(9.75) < 0 < growth(9.80)


def test_rest_and_firing_coexist_below_the_hopf_point():
    # At I = 8 rest is stable, yet a stable firing cycle already exists (it appears at 6.3):
    # a neuron placed at rest stays silent, and a 1 ms kick switches it to firing for good.
    current = 8.0
    V_rest = hh_rest_state(current)[0]
    quiet = run(HH(1, V_init=V_rest), step_current(400.0, DT, current), DT)
    kick = step_current(400.0, DT, 20.0, onset=100.0, offset=101.0)
    kicked = run(HH(1, V_init=V_rest), step_current(400.0, DT, current) + kick, DT)
    assert quiet["spike"].sum() == 0
    assert kicked["spike"].sum() > 15


def test_anode_break_excitation():
    # A hyperpolarizing pulse de-inactivates Na (h rises) and closes K (n falls);
    # on release, Na activates faster than h and n recover, so the cell fires once.
    out = run(HH(1), step_current(150.0, DT, -10.0, onset=20.0, offset=40.0), DT)
    times = out["ts"][out["spike"][:, 0]]
    assert len(times) == 1 and 40.0 < times[0] < 60.0
