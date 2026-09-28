import brainpy as bp
import brainpy.math as bm
import numpy as np
import pytest

from neuromodels.plasticity import (
    BI_POO_TAUS, DEPRESSING, FACILITATING, BCMNeuron, OjaNeuron, PairSTDP, STDPNeuron, TsodyksMarkram,
    pattern_sequence, stdp_pairing, stdp_poisson_drift, stdp_window, stp_release_sequence,
    stp_steady_state,
)
from neuromodels.synapses import regular_times, spike_input
from neuromodels.utils import run

DT = 0.1


@pytest.mark.parametrize("params, trend", [(DEPRESSING, -1), (FACILITATING, +1)],
                         ids=["depressing", "facilitating"])
def test_stp_release_follows_the_exact_recursion_to_the_analytic_steady_state(params, trend):
    rate = 20.0
    times = regular_times(rate, 120, 10.0)
    inputs = spike_input(times, times[-1] + 10.0, DT)
    release = run(TsodyksMarkram(1, **params), inputs, DT, monitors=["release"])["release"][inputs > 0, 0]
    # u and x relax linearly between spikes, so exponential Euler reproduces the spike-to-spike map exactly.
    np.testing.assert_allclose(release, stp_release_sequence(times, **params), rtol=1e-10)
    assert release[0] == pytest.approx(params["U"])  # a rested synapse releases U
    # 120 spikes leave the slower (facilitating) recursion within ~1e-5 of its fixed point.
    assert release[-1] == pytest.approx(stp_steady_state(rate=rate, **params)[2], rel=1e-4)
    assert np.sign(release[-1] - release[0]) == trend
    assert abs(np.log(release[-1] / release[0])) > np.log(3)  # at least threefold at 20 Hz


def test_stp_matches_brainpy_once_u_starts_at_rest():
    params = FACILITATING
    inputs = spike_input(regular_times(40.0, 20, 5.0), 600.0, DT)
    mine = run(TsodyksMarkram(1, **params), inputs, DT, monitors=["u", "x"])
    builtin = bp.dyn.STP(1, U=params["U"], tau_f=params["tau_f"], tau_d=params["tau_d"])
    bp.reset_state(builtin)
    builtin.u.value = bm.zeros(1)  # bp.dyn.STP starts u at U rather than at 0
    theirs = run(builtin, inputs, DT, monitors=["u", "x"], reset=False)
    np.testing.assert_allclose(mine["u"], theirs["u"], atol=1e-12)
    np.testing.assert_allclose(mine["x"], theirs["x"], atol=1e-12)
    # With its default start the first spike already sees u = U + U(1 - U) exp(-5/tau_f), nearly 2U.
    default = run(bp.dyn.STP(1, U=params["U"], tau_f=params["tau_f"], tau_d=params["tau_d"]), inputs, DT,
                  monitors=["u"])
    first = np.argmax(inputs > 0)
    assert default["u"][first, 0] > 1.9 * params["U"]


def test_stdp_window_has_the_right_sign_and_exponential_shape():
    delta = np.array([-60.0, -40.0, -20.0, -10.0, -5.0, -1.0, 1.0, 5.0, 10.0, 20.0, 40.0, 60.0])
    params = dict(A_plus=1.0, A_minus=0.5, **BI_POO_TAUS)
    dw = stdp_pairing(delta, dt=DT, **params)
    assert np.all(dw[delta > 0] > 0) and np.all(dw[delta < 0] < 0)  # pre before post potentiates
    # The traces decay exactly, so one pair reproduces the window to rounding error ...
    np.testing.assert_allclose(dw, stdp_window(delta, **params), rtol=1e-10)
    # ... and log|dw| falls linearly with |dt|, with slope -1/tau on each side.
    for side, tau in [(delta > 0, params["tau_plus"]), (delta < 0, params["tau_minus"])]:
        slope = np.polyfit(np.abs(delta[side]), np.log(np.abs(dw[side])), 1)[0]
        assert -1 / slope == pytest.approx(tau, rel=1e-9)
    assert stdp_pairing([0.0], dt=DT, **params)[0] == 0.0  # spikes in the same step do not interact


class BuiltinSTDPNet(bp.DynSysGroup):
    """Given pre and post spike times through bp.dyn.STDP_Song2000 with one-to-one weights."""

    def __init__(self, pre_times, post_times, **params):
        super().__init__()
        n = len(pre_times)
        self.pre = bp.dyn.SpikeTimeGroup(n, indices=np.arange(n), times=pre_times)
        self.post = bp.dyn.SpikeTimeGroup(n, indices=np.arange(n), times=post_times)
        self.stdp = bp.dyn.STDP_Song2000(pre=self.pre, delay=None, comm=bp.dnn.OneToOne(n, bm.Variable(bm.zeros(n))),
                                         syn=bp.dyn.Expon.desc(n, tau=5.0), out=bp.dyn.CUBA.desc(), post=self.post,
                                         **params)

    def update(self, x):
        self.pre()
        self.post()
        self.stdp()


def test_stdp_pairing_matches_brainpy_stdp_song2000():
    delta = np.array([-40.0, -10.0, -1.0, 1.0, 10.0, 40.0, 0.0])
    net = BuiltinSTDPNet(np.full(len(delta), 50.0), 50.0 + delta, tau_s=16.8, tau_t=33.7, A1=1.0, A2=0.5)
    theirs = run(net, np.zeros(1200), DT, monitors=["stdp.comm.weight"])["stdp.comm.weight"][-1]
    mine = stdp_pairing(delta, dt=DT, A_plus=1.0, A_minus=0.5, **BI_POO_TAUS)
    np.testing.assert_allclose(theirs[:-1], mine[:-1], atol=1e-12)
    # Spikes in the same step: BrainPy applies both updates (A1 - A2), PairSTDP neither.
    assert theirs[-1] == pytest.approx(0.5) and mine[-1] == 0.0


def test_stdp_weights_saturate_at_hard_bounds():
    pre_times = regular_times(10.0, 30, 20.0)
    pre = spike_input([pre_times, pre_times], 3100.0, DT)
    post = spike_input([pre_times + 5.0, pre_times - 5.0], 3100.0, DT)
    model = PairSTDP(2, A_plus=0.1, A_minus=0.1, w_min=0.0, w_max=1.0, w_init=0.5)
    w = run(model, (pre, post), DT, monitors=("w",))["w"]
    assert w.min() >= 0.0 and w.max() <= 1.0
    np.testing.assert_array_equal(w[-1], [1.0, 0.0])  # causal pairs saturate up, anti-causal down


def test_stdp_drift_for_independent_poisson_trains_matches_theory():
    dt, duration, n, rate = 1.0, 20000.0, 1000, 20.0
    steps, p = int(duration / dt), rate * dt * 1e-3
    rng = np.random.default_rng(7)
    pre, post = rng.random((steps, n)) < p, rng.random((steps, n)) < p
    params = dict(A_plus=1.0, A_minus=0.6, **BI_POO_TAUS)  # A_minus tau_minus > A_plus tau_plus
    model = PairSTDP(n, w_min=-np.inf, w_max=np.inf, w_init=0.0, **params)
    w = run(model, (pre, post), dt, monitors=("w",))["w"][-1]
    # Each step a post spike (probability p) meets a pre trace of mean p q/(1 - q), q = exp(-dt/tau_plus),
    # and a pre spike meets a post trace likewise; 5 % is about four standard errors of the mean.
    q_plus, q_minus = np.exp(-dt / params["tau_plus"]), np.exp(-dt / params["tau_minus"])
    expected = steps * p * p * (params["A_plus"] * q_plus / (1 - q_plus) - params["A_minus"] * q_minus / (1 - q_minus))
    assert w.mean() == pytest.approx(expected, rel=0.05)
    assert w.mean() < 0  # uncorrelated inputs are depressed, as in Song et al. (2000)
    # The continuous-time drift differs from the grid value by the excluded same-step pairs (~5 % here).
    assert stdp_poisson_drift(rate, rate, **params) * duration / 1e3 == pytest.approx(expected, rel=0.07)


def test_stdp_neuron_is_reproducible_and_keeps_weights_in_bounds():
    dt, g_max = 0.25, 0.015
    neuron = STDPNeuron(g_max=g_max, A_plus=0.01, seed=3)
    w0 = np.asarray(neuron.stdp.w.value).copy()
    first = run(neuron, np.zeros(8000), dt, monitors=["spike"])
    w1 = np.asarray(neuron.stdp.w.value).copy()
    second = run(neuron, np.zeros(8000), dt, monitors=["spike"])
    # Resetting reseeds the internal Poisson generator, so a rerun repeats every spike.
    np.testing.assert_array_equal(first["spike"], second["spike"])
    assert first["spike"].sum() > 0
    assert w1.min() >= 0.0 and w1.max() <= g_max
    assert np.abs(w1 - w0).max() > 0


def test_oja_rule_finds_the_unit_norm_principal_eigenvector():
    rng = np.random.default_rng(0)
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    covariance = rotation @ np.diag([4.0, 1.0, 0.25]) @ rotation.T
    samples = rng.multivariate_normal(np.zeros(3), covariance, size=10000)
    oja = OjaNeuron(3, eta=0.005, seed=1)
    w = run(oja, samples, 1.0, monitors=["w"])["w"][-1]
    top = np.linalg.eigh(covariance)[1][:, -1]
    assert np.linalg.norm(oja.w_init) < 0.2  # it starts far from unit norm
    assert abs(w @ top) / np.linalg.norm(w) > 0.99
    assert np.linalg.norm(w) == pytest.approx(1.0, abs=0.02)


def test_bcm_rule_becomes_selective_to_one_of_two_patterns():
    patterns = np.array([[1.0, 0.3], [0.3, 1.0]])
    winners = []
    for seed in (0, 1):
        inputs, _ = pattern_sequence(patterns, 40000, seed=seed)
        bcm = BCMNeuron(2, seed=seed)
        before = patterns @ bcm.w_init
        out = run(bcm, inputs, 1.0, monitors=["w", "theta"])
        after = np.maximum(out["w"][-10000:] @ patterns.T, 0.0).mean(axis=0)
        winner = int(np.argmax(after))
        winners.append(winner)
        assert before.min() > 0.5 * before.max()  # initially both patterns drive the neuron
        # Each pattern appears with p = 1/2. At the selective fixed point theta = <y^2> = p y^2
        # must equal the winner's response y, so y = 1/p = 2 for one pattern and 0 for the other.
        assert after[winner] == pytest.approx(2.0, rel=0.05)
        assert after[1 - winner] < 0.02
        assert out["theta"][-10000:].mean() == pytest.approx(2.0, rel=0.05)
    assert sorted(winners) == [0, 1]  # the initial weights and input order decide which pattern wins
