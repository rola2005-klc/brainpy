import brainpy as bp
import brainpy.math as bm
import numpy as np
import pytest

from neuromodels.networks.reservoir import (forecast, generate, harvest_states, mackey_glass, make_reservoir, nrmse,
                                            ridge_regression)

N_TRAIN = 3000


@pytest.fixture(scope="module")
def series():
    """Standardised Mackey–Glass series (training-set mean and std), sampled every time unit."""
    x = mackey_glass(5000, transient=500)
    return (x - x[:N_TRAIN].mean()) / x[:N_TRAIN].std()


@pytest.fixture(scope="module")
def driven(series):
    """The default reservoir (300 units, ρ = 0.9, seed 0) and its states over the whole series."""
    reservoir = make_reservoir(seed=0)
    return reservoir, harvest_states(reservoir, series)


def separation(spectral_radius, series, steps=1500, seed=0):
    """Per-unit RMS distance between runs from the zero state and a random state, same input."""
    reservoir = make_reservoir(spectral_radius=spectral_radius, seed=seed)
    start = np.random.default_rng(seed + 100).uniform(-1, 1, 300)
    from_zero = harvest_states(reservoir, series[:steps])
    from_random = harvest_states(reservoir, series[:steps], initial_state=start)
    return np.linalg.norm(from_zero - from_random, axis=1) / np.sqrt(300), np.linalg.norm(start) / np.sqrt(300)


def test_mackey_glass_rest_state_and_fourth_order_convergence():
    # x* = (β/γ − 1)^{1/n} = 1 is an equilibrium, so a flat history at 1 must stay there exactly.
    np.testing.assert_array_equal(mackey_glass(200, x0=1.0), 1.0)
    coarse, fine, finest = (mackey_glass(300, dt=0.1 / m, sample_every=10 * m) for m in (1, 2, 4))
    error_coarse, error_fine = np.abs(coarse - fine).max(), np.abs(fine - finest).max()
    assert error_coarse < 1e-8
    assert 12 < error_coarse / error_fine < 20  # halving the step divides the error by 2⁴ = 16
    assert 0.2 < coarse.min() and coarse.max() < 1.4  # the τ = 17 attractor lies in about [0.4, 1.35]


def test_reservoir_update_and_spectral_radius():
    reservoir = make_reservoir(num_units=50, spectral_radius=0.8, leak_rate=0.3, seed=3)
    W_in, W, b = (np.asarray(v) for v in (reservoir.Win, reservoir.Wrec, reservoir.bias))
    assert np.abs(np.linalg.eigvals(W)).max() == pytest.approx(0.8, rel=1e-10)
    inputs = np.random.default_rng(1).normal(size=(20, 1))
    x, expected = np.zeros(50), []
    for u in inputs:  # leaky tanh update, row-vector convention as in BrainPy
        x = 0.7 * x + 0.3 * np.tanh(u @ W_in + x @ W + b)
        expected.append(x)
    np.testing.assert_allclose(harvest_states(reservoir, inputs), expected, atol=1e-12)


def test_echo_state_property_below_unit_spectral_radius(series):
    # Linearised at the origin, the update contracts asymptotically by at most (1 − α) + α ρ(W) = 0.97
    # per step (α = 0.3, ρ = 0.9), and saturation (tanh′ ≤ 1) only lowers the gain: 0.97¹⁰⁰⁰ ≈ 6e-14,
    # so a 1e-8 bound after 1000 steps leaves a wide margin.
    distance, initial = separation(0.9, series)
    assert distance[999] < 1e-8 * initial


def test_echo_state_property_fails_for_large_spectral_radius(series):
    # With ρ(W) = 3, W has a real eigenvalue near +3, so the linearization at the origin,
    # (1 − α)I + αW, has spectral radius ≈ 0.7 + 0.3·3 = 1.6 > 1. The unit-variance input is weak
    # next to the recurrent drive, so the reservoir sustains its own irregular activity: the two
    # runs stay a finite distance apart instead of forgetting where they started.
    distance, _ = separation(3.0, series)
    assert distance[-500:].min() > 0.1


@pytest.mark.parametrize("penalize_bias", [False, True])
def test_ridge_matches_lstsq_on_augmented_system(series, driven, penalize_bias):
    # Ridge = ordinary least squares on [Φ; √α D] W = [Y; 0], solved here by SVD (np.linalg.lstsq).
    alpha = 1e-4  # cond(ΦᵀΦ + αD) ≈ 5e8, so the normal equations keep ~8 correct digits
    features, targets = driven[1][100:599], series[101:600]
    weights = ridge_regression(features, targets, alpha, penalize_bias=penalize_bias)
    design = np.hstack([np.ones((len(features), 1)), features])
    penalty = np.sqrt(alpha) * np.eye(design.shape[1])[0 if penalize_bias else 1:]
    augmented = np.vstack([design, penalty])
    reference = np.linalg.lstsq(augmented, np.concatenate([targets, np.zeros(len(penalty))]), rcond=None)[0]
    assert np.linalg.norm(weights - reference) < 1e-6 * np.linalg.norm(reference)


def test_brainpy_ridge_trainer_matches_closed_form_with_penalised_bias(series):
    # bp.RidgeTrainer prepends a ones column and regularises it like every other weight.
    washout, alpha, units = 100, 1e-4, 100

    class ESN(bp.DynamicalSystem):
        def __init__(self):
            super().__init__(mode=bm.batching_mode)
            self.reservoir = make_reservoir(units, seed=5, mode=bm.batching_mode)
            self.readout = bp.dnn.Dense(units, 1, mode=bm.training_mode)

        def update(self, x):
            return self.readout(self.reservoir(x))

    model = ESN()
    trainer = bp.RidgeTrainer(model, alpha=alpha, progress_bar=False)
    inputs, targets = bm.asarray(series[None, :699, None]), bm.asarray(series[None, 1:700, None])
    trainer.predict(inputs[:, :washout])  # washout; the state carries over into fit
    trainer.fit([inputs[:, washout:], targets[:, washout:]])
    trained = np.concatenate([np.ravel(model.readout.b), np.ravel(model.readout.W)])

    states = harvest_states(make_reservoir(units, seed=5), series[:699])
    penalised = ridge_regression(states[washout:], series[1 + washout:700], alpha, penalize_bias=True)
    free = ridge_regression(states[washout:], series[1 + washout:700], alpha, penalize_bias=False)
    assert np.linalg.norm(trained - penalised) < 1e-6 * np.linalg.norm(penalised)
    assert np.linalg.norm(trained - free) > 1e-3 * np.linalg.norm(free)


@pytest.mark.parametrize("horizon, threshold", [(1, 1e-3), (84, 0.1)])
def test_mackey_glass_prediction_on_held_out_data(series, driven, horizon, threshold):
    # Persistence (predicting u(n + h) = u(n)) scores 0.15 at h = 1 and 1.6 at h = 84, so these
    # thresholds demand at least 100× and 16× better than doing nothing.
    _, targets, predictions = forecast(series, driven[1], horizon, N_TRAIN)
    assert len(targets) == len(series) - N_TRAIN - horizon
    assert nrmse(targets, predictions) < threshold
    assert nrmse(targets, predictions) < 0.1 * nrmse(series[N_TRAIN + horizon:], series[N_TRAIN:-horizon])


def test_closed_loop_generation_follows_the_attractor(series, driven):
    # Fed its own one-step predictions, the network must continue the series for 84 steps, about
    # 1.7 periods of its dominant ~50-step oscillation, before chaos amplifies the readout errors.
    reservoir, states = driven
    weights, _, _ = forecast(series[:N_TRAIN], states[:N_TRAIN], 1, N_TRAIN)
    reservoir.state.value = bm.asarray(states[N_TRAIN - 1])  # the state right after series[N_TRAIN - 1]
    generated = generate(reservoir, weights, series[N_TRAIN - 1], 84)[:, 0]
    assert nrmse(series[N_TRAIN:N_TRAIN + 84], generated) < 0.05
