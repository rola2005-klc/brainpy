import brainpy as bp
import numpy as np

from neuromodels.utils import cv_isi, firing_rate, run, spike_times, step_current


def test_run_returns_end_of_step_times():
    # A LIF neuron below threshold has the exact solution V(t) = RI(1 - exp(-t/tau)),
    # and exponential Euler integrates linear ODEs exactly, so any mismatch here
    # would be a time-labelling error, not an integration error.
    dt, tau, current = 0.1, 10.0, 15.0
    lif = bp.dyn.Lif(1, V_rest=0.0, V_reset=0.0, V_th=20.0, tau=tau, method="exp_euler",
                     V_initializer=bp.init.Constant(0.0))
    out = run(lif, step_current(50.0, dt, current), dt)
    exact = current * (1 - np.exp(-out["ts"] / tau))
    np.testing.assert_allclose(out["V"][:, 0], exact, atol=1e-10)


def test_run_is_repeatable():
    lif = bp.dyn.Lif(1, V_rest=0.0, V_reset=0.0, V_th=20.0, V_initializer=bp.init.Constant(0.0))
    inputs = step_current(100.0, 0.1, 25.0)
    first, second = run(lif, inputs, 0.1), run(lif, inputs, 0.1)
    np.testing.assert_array_equal(first["V"], second["V"])


def test_step_current_edges_use_step_indices():
    current = step_current(1.0, 0.1, 2.0, onset=0.3, offset=0.6)
    np.testing.assert_array_equal(current, [0, 0, 0, 2, 2, 2, 0, 0, 0, 0])
    assert step_current(1.0, 0.1, [1.0, 2.0, 3.0]).shape == (10, 3)


def test_spike_statistics():
    ts = np.arange(1, 1001) * 1.0  # 1 s at dt = 1 ms
    spikes = np.zeros((1000, 2), dtype=bool)
    spikes[99::100, 0] = True  # regular 10 Hz train
    np.testing.assert_allclose(firing_rate(spikes, 1.0), [10.0, 0.0])
    times = spike_times(spikes, ts)[0]
    np.testing.assert_allclose(times, np.arange(100, 1001, 100))
    assert cv_isi(times) == 0.0
    assert np.isnan(cv_isi(times[:2]))
