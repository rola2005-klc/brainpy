import numpy as np
import pytest

from neuromodels.networks.cann import (CANN1D, critical_inhibition, gaussian_input, max_tracking_speed,
                                       population_vector, ring_distance, stationary_bump, step_midpoints,
                                       tracking_lag)
from neuromodels.utils import run

DT = 0.05
N = 128  # a / Δx ≈ 10: the Riemann sums over Gaussians are already exact to machine precision
K_C = critical_inhibition(N / (2 * np.pi))
U_C = stationary_bump(K_C, N / (2 * np.pi))[0]  # bump height at k = k_c
WRAP = np.exp(-np.pi ** 2 / (4 * 0.5 ** 2))  # Gaussian tail e^{-π²/4a²} at the antipode, a = 0.5


def simulate(model, center, amplitude, duration):
    """Drive ``model`` with a Gaussian stimulus whose centre and height are functions of time."""
    t = step_midpoints(duration, DT)
    inputs = gaussian_input(model.x, center(t), amplitude(t), model.a)
    return run(model, inputs, DT, monitors=("u", "r"))


def flash(z0, height, off=10.0):
    """Stimulus fixed at ``z0`` that switches off at ``off``."""
    return (lambda t: np.full_like(t, z0)), (lambda t: np.where(t < off, height, 0.0))


@pytest.mark.parametrize("ratio", [0.98, 1.02])
def test_bump_persists_only_below_critical_inhibition(ratio):
    # The ring reproduces the continuum k_c up to the wrap-around term e^{-π²/4a²} ≈ 5e-5, so a 2%
    # margin is limited only by time: above k_c the height must creep past the ghost of the
    # saddle-node, which takes about 2πτ/√(k/k_c − 1) ≈ 44τ at 2%. The run waits 140τ.
    model = CANN1D(N, k=ratio * K_C)
    out = simulate(model, *flash(0.0, 2 * U_C), duration=150.0)
    peak = out["u"].max(axis=1)
    at_removal = peak[np.searchsorted(out["ts"], 10.0)]
    if ratio < 1:
        expected = stationary_bump(model.k, model.rho, model.J0, model.a)[0]
        assert peak[-1] == pytest.approx(expected, rel=1e-6)
        assert population_vector(out["r"][-1], model.x) == pytest.approx(0.0, abs=1e-9)
    else:
        assert np.isnan(stationary_bump(model.k, model.rho, model.J0, model.a)[0])
        assert peak[-1] < 1e-6 * at_removal


def test_stationary_bump_matches_closed_form():
    model = CANN1D(N)
    u0, r0 = stationary_bump(model.k, model.rho, model.J0, model.a)
    out = simulate(model, *flash(0.0, u0), duration=60.0)
    # The only deviation from the infinite-line Gaussian is the tail wrapping round the ring: of
    # relative size WRAP in u, and of order WRAP² in r ∝ u².
    np.testing.assert_allclose(out["u"][-1], gaussian_input(model.x, 0.0, u0, model.a), atol=1.1 * WRAP * u0)
    np.testing.assert_allclose(out["r"][-1], r0 * np.exp(-model.x ** 2 / (2 * model.a ** 2)), atol=4 * WRAP ** 2 * r0)
    # A wrapped Gaussian of width a has mean resultant length |<e^{ix}>| = e^{-a²/2}.
    resultant = abs(out["r"][-1] @ np.exp(1j * model.x)) / out["r"][-1].sum()
    assert resultant == pytest.approx(np.exp(-model.a ** 2 / 2), rel=1e-7)


@pytest.mark.parametrize("strength", [0.1, 0.5, 2.0])
def test_driven_bump_height_solves_amplitude_cubic(strength):
    # A centred stimulus with the bump's own shape keeps u Gaussian, so the height obeys an exact scalar ODE.
    model = CANN1D(N)
    A = strength * stationary_bump(model.k, model.rho, model.J0, model.a)[0]
    out = simulate(model, lambda t: 0.0 * t, lambda t: np.full_like(t, A), duration=80.0)
    u_driven, r_driven = stationary_bump(model.k, model.rho, model.J0, model.a, A=A)
    assert out["u"][-1].max() == pytest.approx(u_driven, rel=1e-7)
    assert out["r"][-1].max() == pytest.approx(r_driven, rel=1e-7)


@pytest.mark.parametrize("z0", [0.4321, -3.0, 3.1, np.pi])
def test_population_vector_recovers_off_grid_stimulus(z0):
    # Off-grid positions and the ±π seam: lattice pinning and aliasing of a sampled Gaussian are
    # both exponentially small in (a/Δx)², so the decoded position is exact to rounding error.
    model = CANN1D(N)
    u0 = stationary_bump(model.k, model.rho, model.J0, model.a)[0]
    out = simulate(model, *flash(z0, u0), duration=60.0)
    decoded = population_vector(out["r"], model.x)
    during = np.searchsorted(out["ts"], 10.0) - 1
    assert abs(ring_distance(decoded[during], z0)) < 1e-9
    assert abs(ring_distance(decoded[-1], z0)) < 1e-9  # the bump stays put once the stimulus is gone
    # Between grid points the largest sample undershoots u0, so compare the whole profile.
    np.testing.assert_allclose(out["u"][-1], gaussian_input(model.x, z0, u0, model.a), atol=1.1 * WRAP * u0)


def test_bump_shape_is_translation_invariant():
    # J is circulant, so rotating the stimulus by m grid steps rotates the whole trajectory by m steps.
    model = CANN1D(N)
    u0 = stationary_bump(model.k, model.rho, model.J0, model.a)[0]
    reference = simulate(model, *flash(model.x[0], u0), duration=40.0)["u"]
    for shift in (17, 64, N - 1):  # N − 1 puts the bump on the ±π seam
        shifted = simulate(model, *flash(model.x[shift], u0), duration=40.0)["u"]
        np.testing.assert_allclose(shifted, np.roll(reference, shift, axis=1), rtol=0, atol=1e-12 * u0)


def test_moving_stimulus_is_tracked_with_bounded_lag():
    model = CANN1D(N)
    u0 = stationary_bump(model.k, model.rho, model.J0, model.a)[0]
    A = 0.1 * u0  # weak input: the recurrent dynamics, not the stimulus, shape the bump
    U_A = stationary_bump(model.k, model.rho, model.J0, model.a, A=A)[0]
    t_move, velocity = 20.0, 0.5 * max_tracking_speed(model.tau, A, U_A, model.a)
    duration = t_move + 12 * U_A / A  # the lag relaxes with time constant ≈ τU/A, so ~11 of them

    def lag_for(v):
        out = simulate(model, lambda t: np.where(t < t_move, 0.0, v * (t - t_move)),
                       lambda t: np.where(t < 5.0, u0, A), duration)  # ignite, then weak input
        stimulus = np.where(out["ts"] < t_move, 0.0, v * (out["ts"] - t_move))
        lag = ring_distance(stimulus, population_vector(out["r"], model.x))
        last = out["ts"] > duration - 20.0
        return lag[last], out["u"][last].max(axis=1).mean()

    forward, height = lag_for(velocity)
    backward, _ = lag_for(-velocity)
    assert np.all(forward > 0)                       # the bump trails the stimulus ...
    np.testing.assert_allclose(backward, -forward, atol=1e-9)  # ... whichever way it moves
    assert np.ptp(forward) < 1e-3 * forward.mean()   # steady lag: the bump moves at the stimulus speed
    assert forward.mean() < np.sqrt(3) * model.a      # below the largest lag the first-order theory allows
    # First-order theory in A (adjoint projection); the O(A/U) ≈ 0.1 corrections are about 1% here.
    expected = tracking_lag(velocity, model.tau, A, height, model.a)
    assert forward.mean() == pytest.approx(expected, rel=0.03)


def test_weak_input_lag_follows_adjoint_projection():
    # Near the maximal speed the Gaussian factor of the lag equation matters. Projecting onto the
    # adjoint translation mode u ∂u/∂z gives exp(−s²/6a²); projecting onto ∂u/∂z itself would give
    # exp(−s²/8a²) and a lag about 9% shorter here. With A = 0.02 u0 first-order theory (U = u0) holds.
    model = CANN1D(N)
    u0 = stationary_bump(model.k, model.rho, model.J0, model.a)[0]
    A = 0.02 * u0
    velocity = 0.9 * max_tracking_speed(model.tau, A, u0, model.a)
    duration = 20.0 + 40.0 * u0 / A  # the lag relaxes slowly near the fold (time constant ≈ 120τ)
    out = simulate(model, lambda t: np.where(t < 20.0, 0.0, velocity * (t - 20.0)),
                   lambda t: np.where(t < 5.0, u0, A), duration)
    stimulus = np.where(out["ts"] < 20.0, 0.0, velocity * (out["ts"] - 20.0))
    lag = ring_distance(stimulus, population_vector(out["r"], model.x))[-1]
    assert lag == pytest.approx(tracking_lag(velocity, model.tau, A, u0, model.a), rel=0.02)
