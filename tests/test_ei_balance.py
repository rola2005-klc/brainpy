import numpy as np
import pytest

from neuromodels.networks.ei_balance import (TAU_M, TAU_REF, V_REST, V_RESET, V_TH, network_statistics,
                                             scaled_network, simulate)
from neuromodels.utils import n_steps, spike_times

DT = 0.1
DRIVE = 20.0  # mV
TRANSIENT = 200.0  # ms


# The tests use 1000 neurons with the benchmark's in-degree (64 E and 16 I inputs, p = 0.08):
# each neuron sees the same input statistics as in the 4000-neuron network, at a quarter of the cost.
@pytest.fixture(scope="module")
def balanced_run():
    net = scaled_network(1000, seed=1)
    return net, simulate(net, 1200.0, DT, DRIVE, n_record=20)


@pytest.fixture(scope="module")
def balanced(balanced_run):
    return network_statistics(balanced_run[1], DT, TRANSIENT)


@pytest.fixture(scope="module")
def no_inhibition():
    out = simulate(scaled_network(1000, w_inh=0.0, seed=1), 1200.0, DT, DRIVE, n_record=20)
    return network_statistics(out, DT, TRANSIENT)


def test_uncoupled_neurons_fire_regularly_at_the_lif_rate():
    # With all weights zero each neuron only sees the drive and relaxes towards
    # V_inf = V_rest + 20 = -40 mV, firing with period t_ref + tau ln((V_inf - V_reset) / (V_inf - V_th))
    # = 18.86 ms (53 Hz) and CV = 0. Any irregularity in the network is made by its connections.
    out = simulate(scaled_network(100, w_exc=0.0, w_inh=0.0), 500.0, DT, DRIVE)
    v_inf = V_REST + DRIVE
    period = TAU_REF + TAU_M * np.log((v_inf - V_RESET) / (v_inf - V_TH))
    intervals = np.concatenate([np.diff(t) for t in spike_times(out["spike_E"], out["ts"])])
    assert intervals.size > 1000  # 80 neurons x ~25 intervals
    # Spikes are registered at the end of the step in which V crosses threshold, so each
    # interval is the true period moved onto the time grid.
    np.testing.assert_allclose(intervals, period, atol=DT)


@pytest.mark.parametrize("proj_name, pre_key, weight, tau", [("E2E", "spike_E", 0.6, 5.0),
                                                             ("I2E", "spike_I", 6.7, 10.0)])
def test_mean_conductance_follows_campbells_theorem(balanced_run, proj_name, pre_key, weight, tau):
    # Each presynaptic spike adds a decaying exponential of area weight * tau, so the mean
    # conductance of a neuron is weight * tau * (sum of its presynaptic partners' rates),
    # whatever the dynamics. This checks weights, time constants and wiring at once. In
    # discrete time the synapse decays by exp(-h), h = dt / tau, before the conductance is
    # used, which scales the area by h exp(-h) / (1 - exp(-h)) (0.990 for tau = 5 ms).
    net, out = balanced_run
    start = n_steps(TRANSIENT, DT)
    spikes = out[pre_key][start:]
    rates = spikes.sum(axis=0) / (spikes.shape[0] * DT) * 1e3
    proj = getattr(net, proj_name)
    indices, indptr = np.asarray(proj.comm.indices), np.asarray(proj.comm.indptr)
    pre_of_synapse = np.repeat(np.arange(indptr.size - 1), np.diff(indptr))
    n_record = out["g_E"].shape[1]
    input_rate = np.array([rates[pre_of_synapse[indices == i]].sum() for i in range(n_record)])
    h = DT / tau
    expected = weight * tau * 1e-3 * input_rate * h * np.exp(-h) / -np.expm1(-h)
    measured = out["g_E" if proj_name == "E2E" else "g_I"][start:].mean(axis=0)
    np.testing.assert_allclose(measured, expected, rtol=0.02)


def test_balanced_network_is_asynchronous_and_irregular(balanced):
    # Rates are low (far below the 53 Hz of an uncoupled neuron and the 200 Hz refractory
    # limit), spike trains at least as irregular as a Poisson process (CV = 1; bursty
    # trains exceed it), and neurons nearly independent (count correlations in 10 ms bins).
    assert 2.0 < balanced["rate_E"] < 40.0 and 2.0 < balanced["rate_I"] < 40.0
    assert balanced["cv_counted"] > 500
    assert balanced["cv_mean"] > 0.7
    assert balanced["correlation"] < 0.05


def test_inhibition_cancels_most_of_the_excitation(balanced):
    # Excitation (drive + recurrent) and inhibition are each ten times larger than the 10 mV
    # from rest to threshold, yet their sum is a small fraction of either.
    assert balanced["I_exc"] > 10 * (V_TH - V_REST)
    assert -balanced["I_inh"] > 10 * (V_TH - V_REST)
    assert balanced["net_to_exc"] < 0.2


def test_removing_inhibition_breaks_the_asynchronous_irregular_state(balanced, no_inhibition):
    # Recurrent excitation then only adds to the suprathreshold drive, so every neuron fires
    # as soon as its refractory period ends: just under 1 / t_ref = 200 Hz, regularly and in
    # step with the others.
    assert 0.9 * 1e3 / TAU_REF < no_inhibition["rate_E"] < 1e3 / TAU_REF
    assert no_inhibition["cv_mean"] < 0.05
    assert no_inhibition["correlation"] > 10 * balanced["correlation"]
    assert no_inhibition["net_to_exc"] == pytest.approx(1.0)


def test_simulation_is_reproducible():
    net = scaled_network(400, seed=2)
    first, second = simulate(net, 200.0, DT, DRIVE), simulate(net, 200.0, DT, DRIVE)
    np.testing.assert_array_equal(first["spike_E"], second["spike_E"])
    np.testing.assert_array_equal(first["V"], second["V"])
