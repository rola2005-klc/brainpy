from dataclasses import replace

import brainpy.math as bm
import jax.numpy as jnp
import numpy as np
import pytest

from neuromodels.networks.decision import (PARAMS, WongWang2006, drift, fixed_points, inverse_transfer,
                                           simulate_spiking_trial, simulate_trials, summarize, transfer)
from neuromodels.utils import n_steps, run

DT = 0.1


def test_transfer_function_limits():
    # At a x = b the formula is 0/0 with limit 1/d; far above threshold exp(-d(ax - b))
    # vanishes and H is the straight line a x - b.
    assert transfer(PARAMS.b / PARAMS.a) == pytest.approx(1 / PARAMS.d, rel=1e-12)
    assert transfer(1.0) == pytest.approx(PARAMS.a - PARAMS.b, rel=1e-9)
    x = np.linspace(0.2, 0.6, 101)
    assert np.all(np.diff(transfer(x)) > 0)
    np.testing.assert_allclose(np.asarray(transfer(jnp.asarray(x))), transfer(x), rtol=1e-12)
    np.testing.assert_allclose(inverse_transfer(transfer(x)), x, atol=1e-12)


def test_stimulus_leaves_two_attractors_and_a_saddle():
    # With the stimulus on at c' = 0 the spontaneous state is gone: two mirror-image
    # decision attractors (one population high, the other suppressed) and a saddle on
    # the diagonal between them.
    points = fixed_points(coherence=0.0, mu0=30.0)
    assert sorted(p.kind for p in points) == ["saddle", "stable", "stable"]
    for p in points:
        assert np.max(np.abs(drift(p.S1, p.S2, 0.0, 30.0))) < 1e-14
    saddle = next(p for p in points if p.kind == "saddle")
    assert saddle.S1 == pytest.approx(saddle.S2, abs=1e-10)
    low, high = sorted((p for p in points if p.kind == "stable"), key=lambda p: p.S1)
    assert (low.S1, low.S2) == pytest.approx((high.S2, high.S1), abs=1e-10)
    assert high.S1 > 0.5 and high.S2 < 0.1
    # The saddle fires below the 15 Hz threshold, so a crossing means the state has passed it.
    x_saddle = (PARAMS.J_same - PARAMS.J_cross) * saddle.S1 + PARAMS.I_0 + PARAMS.J_ext * 30.0
    assert transfer(x_saddle) < PARAMS.threshold


def test_without_stimulus_the_spontaneous_state_is_stable():
    # Before the stimulus the network is tristable (spontaneous + two decision states),
    # with a saddle between each decision state and the spontaneous one.
    points = fixed_points(coherence=0.0, mu0=0.0)
    assert sorted(p.kind for p in points) == ["saddle", "saddle", "stable", "stable", "stable"]
    spontaneous = min(points, key=lambda p: p.S1 + p.S2)
    assert spontaneous.kind == "stable"
    assert spontaneous.S1 == pytest.approx(spontaneous.S2, abs=1e-10)
    assert spontaneous.S1 == pytest.approx(0.1, abs=0.01)  # the default initial condition sits here


def test_noise_free_trials_settle_on_the_fixed_points():
    # Without noise the simulation must end where the phase-plane analysis says. At c' = 6.4 %
    # population 1 wins. At c' = 0 the diagonal S1 = S2 is invariant, so the state slides
    # along the saddle's stable direction into the saddle and never reaches threshold.
    quiet = replace(PARAMS, sigma=0.0)
    out = simulate_trials([6.4, 0.0], 1, duration=6000.0, dt=DT, params=quiet)
    winner = next(p for p in fixed_points(6.4, 30.0) if p.kind == "stable" and p.S1 > p.S2)
    saddle = next(p for p in fixed_points(0.0, 30.0) if p.kind == "saddle")
    np.testing.assert_allclose([out["S1_final"][0], out["S2_final"][0]], [winner.S1, winner.S2], atol=1e-6)
    np.testing.assert_allclose([out["S1_final"][1], out["S2_final"][1]], [saddle.S1, saddle.S2], atol=1e-6)
    assert out["choice"].tolist() == [1, 0]


def test_noise_has_the_ou_stationary_variance():
    # Euler–Maruyama turns the OU equation into I_{n+1} = (1 - h) I_n + sigma sqrt(h) xi_n with
    # h = dt / tau_AMPA, whose stationary variance is exactly sigma^2 / (2 - h) (sigma^2 / 2 as
    # dt -> 0). 100 ms is 50 correlation times, so the start from zero is forgotten.
    model = WongWang2006(np.zeros(4000))
    bm.random.seed(5)
    run(model, np.zeros(n_steps(100.0, DT)), DT, monitors=())
    samples = np.concatenate([np.asarray(model.noise1.value), np.asarray(model.noise2.value)])
    expected = PARAMS.sigma / np.sqrt(2 - DT / PARAMS.tau_ampa)
    # The SD of 8000 Gaussian samples has a relative standard error of 1/sqrt(2 * 8000) = 0.8 %.
    assert samples.std() == pytest.approx(expected, rel=0.03)
    assert abs(samples.mean()) < 3 * expected / np.sqrt(samples.size)


def test_zero_coherence_choices_are_unbiased():
    # Both populations get identical input, so each choice is a fair coin flip: the fraction
    # choosing population 1 must lie within 3 binomial standard errors of 0.5.
    stats = summarize(simulate_trials([0.0], 800, duration=2100.0, dt=DT, seed=3))
    n = stats["n_decided"][0]
    assert n == 800
    assert abs(stats["accuracy"][0] - 0.5) < 3 * 0.5 / np.sqrt(n)


@pytest.fixture(scope="module")
def sweep():
    # 3 s of stimulus, so trials that decide within 2 s have a second to settle.
    return simulate_trials([0.0, 6.4, 12.8, 51.2], 300, duration=3100.0, dt=DT, seed=4)


def test_accuracy_rises_with_coherence(sweep):
    stats = summarize(sweep)
    assert np.all(stats["undecided"] == 0)
    assert np.all(np.diff(stats["accuracy"]) >= 0)
    assert stats["accuracy"][1] > 0.75 and stats["accuracy"][-1] > 0.99


def test_decisions_are_faster_at_higher_coherence(sweep):
    # A stronger bias pushes the state off the saddle sooner. Each drop in mean decision
    # time must exceed twice its standard error.
    stats = summarize(sweep)
    drops = -np.diff(stats["rt"])
    assert np.all(drops > 2 * np.hypot(stats["rt_sem"][:-1], stats["rt_sem"][1:]))


def test_winner_takes_all(sweep):
    # The population that crossed threshold first is ahead at the end of every trial (no
    # changes of mind). Trials given a second to settle after deciding sit at a decision
    # attractor: one population high, the other suppressed (S ~ 0.66 vs 0.05 at c' = 0).
    winner = np.where(sweep["S1_final"] > sweep["S2_final"], 1, 2)
    np.testing.assert_array_equal(winner, sweep["choice"])
    settled = sweep["rt"] < 2000.0
    assert settled.mean() > 0.99
    high = np.maximum(sweep["S1_final"], sweep["S2_final"])[settled]
    low = np.minimum(sweep["S1_final"], sweep["S2_final"])[settled]
    assert np.all(high > 0.5) and np.all(low < 0.15)


def test_simulation_is_reproducible():
    first = simulate_trials([6.4], 50, duration=600.0, dt=DT, seed=7)
    second = simulate_trials([6.4], 50, duration=600.0, dt=DT, seed=7)
    np.testing.assert_array_equal(first["S1_final"], second["S1_final"])
    np.testing.assert_array_equal(first["choice"], second["choice"])


def test_spiking_network_smoke():
    # Smoke test of the spiking Wang (2002) network: without a stimulus it must settle into
    # a low-rate spontaneous state (a few Hz; interneurons faster than pyramidal cells).
    out = simulate_spiking_trial(0.0, mu0=0.0, duration=400.0, dt=DT, seed=0)
    assert out["E.spike"].shape == (4000, 1600) and out["I.spike"].shape == (4000, 400)
    late = slice(n_steps(200.0, DT), None)
    rate_E = out["E.spike"][late].mean() / DT * 1e3
    rate_I = out["I.spike"][late].mean() / DT * 1e3
    assert 0.5 < rate_E < 6.0
    assert rate_E < rate_I < 20.0
