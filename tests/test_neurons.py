import brainpy as bp
import numpy as np
import pytest

from izhikevich_demo import simulate_izhikevich
from neuromodels.analysis import adaptation_index, expif_rheobase, fi_curve, lif_rate, qif_rheobase
from neuromodels.neurons import LIF, QIF, AdEx, ExpIF, Izhikevich
from neuromodels.utils import run, spike_times, step_current

DT = 0.05

# Each custom class next to BrainPy's built-in model with the same parameters and start state.
PAIRS = {
    "LIF": (lambda: LIF(1), lambda: bp.dyn.Lif(1, V_initializer=bp.init.Constant(0.0)), 25.0),
    "QIF": (lambda: QIF(1), lambda: bp.dyn.QuaIF(1, V_initializer=bp.init.Constant(-65.0)), 5.0),
    "ExpIF": (lambda: ExpIF(1), lambda: bp.dyn.ExpIF(1, V_th=-30.0, V_initializer=bp.init.Constant(-65.0)), 5.0),
    "AdEx": (lambda: AdEx(1),
             lambda: bp.dyn.AdExIF(1, V_rest=-70.6, V_reset=-70.6, V_th=-40.4, V_T=-50.4, delta_T=2.0, a=4.0, b=80.5,
                                   tau=281.0 / 30.0, tau_w=144.0, R=1.0 / 30.0,
                                   V_initializer=bp.init.Constant(-70.6), w_initializer=bp.init.Constant(0.0)), 800.0),
    "Izhikevich": (lambda: Izhikevich(1), lambda: bp.dyn.Izhikevich(1, V_initializer=bp.init.Constant(-70.0)), 10.0),
}


@pytest.mark.parametrize("name", PAIRS)
def test_matches_brainpy_builtin(name):
    make_mine, make_builtin, current = PAIRS[name]
    inputs = step_current(300.0, DT, current, onset=20.0)
    mine, builtin = run(make_mine(), inputs, DT), run(make_builtin(), inputs, DT)
    np.testing.assert_allclose(mine["V"], builtin["V"], atol=1e-9)
    assert mine["spike"].sum() == builtin["spike"].sum() > 0


def test_izhikevich_matches_numpy_demo():
    # Both use forward Euler at dt = 0.25 ms; the demo prints V = 30 at spikes, BrainPy the reset value.
    times, demo_v, _, demo_spikes = simulate_izhikevich(250, 0.25)
    current = np.where((times >= 50) & (times <= 200), 10.0, 0.0)
    out = run(Izhikevich(1, V_init=-65.0, method="euler"), current, 0.25)
    spiking = out["spike"][:-1, 0]
    np.testing.assert_allclose(out["ts"][out["spike"][:, 0]], demo_spikes)
    np.testing.assert_allclose(out["V"][:-1, 0][~spiking], demo_v[1:][~spiking], atol=1e-9)


@pytest.mark.parametrize("current", [21.0, 25.0, 40.0])
def test_lif_period_matches_closed_form(current):
    # exp_euler integrates the linear LIF equation exactly, so the only error is that a spike
    # registers at the end of the step in which V crosses threshold: ISIs lie in [T, T + dt).
    dt = 0.01
    out = run(LIF(1, t_ref=2.0, method="exp_euler"), step_current(500.0, dt, current), dt)
    period = 1000.0 / lif_rate(current, t_ref=2.0)
    isi = np.diff(spike_times(out["spike"][:, 0], out["ts"]))
    assert len(isi) > 5
    assert np.all(isi > period - 1e-6) and np.all(isi < period + dt + 1e-6)


@pytest.mark.parametrize("make, rheobase", [(QIF, qif_rheobase()), (ExpIF, expif_rheobase())])
def test_saddle_node_rheobase(make, rheobase):
    # Just above a saddle-node the period diverges like 1/sqrt(I - I_rh): class 1 excitability.
    rates = fi_curve(lambda n: make(n), [0.98 * rheobase, 1.02 * rheobase], duration=3000.0, dt=DT, transient=0.0)
    assert rates[0] == 0.0
    assert 0.0 < rates[1] < 5.0


def _pattern(name, inputs):
    out = run(Izhikevich.from_preset(name), inputs, DT)
    return spike_times(out["spike"][:, 0], out["ts"])


def test_izhikevich_regular_spiking_adapts():
    isi = np.diff(_pattern("RS", step_current(600.0, DT, 10.0, onset=100.0)))
    assert isi[-1] > 2 * isi[0]


def test_izhikevich_fast_spiking_does_not_adapt():
    times = _pattern("FS", step_current(600.0, DT, 10.0, onset=100.0))
    rs_times = _pattern("RS", step_current(600.0, DT, 10.0, onset=100.0))
    assert len(times) > 4 * len(rs_times)
    assert abs(adaptation_index(times[5:])) < 0.01  # after the first few spikes the train is regular


def test_izhikevich_intrinsic_burst_then_tonic():
    isi = np.diff(_pattern("IB", step_current(600.0, DT, 10.0, onset=100.0)))
    assert np.all(isi[:2] < 5.0) and np.all(isi[2:] > 25.0)


def test_izhikevich_chattering_repeats_bursts():
    isi = np.diff(_pattern("CH", step_current(600.0, DT, 10.0, onset=100.0)))
    assert np.sum(isi > 25.0) >= 5  # pauses between bursts
    assert np.mean(isi < 5.0) > 0.7  # most intervals fall inside bursts


def test_izhikevich_thalamocortical_rebound_burst():
    times = _pattern("TC", step_current(600.0, DT, -10.0, onset=100.0, offset=300.0))
    assert np.all((times > 300.0) & (times < 350.0)) and len(times) >= 3


def test_izhikevich_resonator_fires_only_above_its_hopf_point():
    # The RZ preset loses rest through a Hopf bifurcation at I = 0.2625 (neuromodels.analysis).
    assert len(_pattern("RZ", step_current(600.0, DT, 0.2, onset=100.0))) == 0
    assert len(_pattern("RZ", step_current(600.0, DT, 0.3, onset=100.0))) > 5
