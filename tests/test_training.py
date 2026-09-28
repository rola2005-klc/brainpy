"""Tests for BPTT on a rate RNN and surrogate-gradient training of a spiking network.

Networks are small and trained briefly with fixed seeds; each trained network is
shared by the tests that inspect it through a module-scoped fixture.
"""

import brainpy.math as bm
import jax.numpy as jnp
import numpy as np
import pytest

from neuromodels.training import rnn, snn


# --------------------------------------------------------------------------- rate RNN

@pytest.fixture(scope="module")
def trained_rnn():
    task = rnn.DecisionTask()
    model = rnn.RateRNN(2, 32, 1, seed=0)
    # A short run keeps a larger final step (lr_decay=0.1) than the 1000-iteration script.
    history = rnn.train(model, task, n_iterations=300, batch_size=32, lr=1e-2, lr_decay=0.1, seed=0, eval_every=0)
    result = rnn.evaluate(model, task, n_per_level=150, seed=1)
    return task, history, result


def test_ideal_observer_matches_closed_form():
    # The summed evidence of n steps is N(n c, n noise^2), so the ideal observer is right with
    # probability Phi(|c| sqrt(n) / noise), and at c = 0 its kernel is 2 noise sqrt(2 / (pi n))
    # at every step. Both are the yardsticks for the trained network below.
    task = rnn.DecisionTask()
    rng = np.random.default_rng(0)
    trials = task.sample(rng, coherence=np.full(20_000, 0.1))
    ideal = trials["inputs"][task.stimulus_steps, :, 1].sum(axis=0) > 0
    assert abs(ideal.mean() - task.ideal_accuracy(0.1)) < 0.01  # 3 standard errors
    zero = task.sample(rng, coherence=np.zeros(20_000))
    choice = np.sign(zero["inputs"][task.stimulus_steps, :, 1].sum(axis=0))
    kernel = rnn.psychophysical_kernel(zero["inputs"], zero["coherence"], choice, task)
    np.testing.assert_allclose(kernel.mean(), task.ideal_kernel(), rtol=0.03)


def test_rnn_loss_approaches_bayes_floor(trained_rnn):
    task, history, _ = trained_rnn
    # The random initial readout starts above the loss of a readout stuck at 0 (the fraction of
    # decision steps), and because the evidence is noisy no network can go below the Bayes floor.
    # Closing at least half of the gap between the two requires reporting the sign of the
    # integrated evidence while holding the output at 0 until the go signal. (Minibatch noise in
    # the last-50 mean is ~5% of the gap; the trained network ends near 25%.)
    zero, floor = task.zero_output_loss(), task.bayes_loss()
    start, end = history["loss"][:10].mean(), history["loss"][-50:].mean()
    assert start > zero
    assert end < floor + 0.5 * (zero - floor)


def test_rnn_held_out_accuracy_near_ideal_observer(trained_rnn):
    _, _, result = trained_rnn
    # Chance is 0.5; the ideal observer on the same held-out evidence is the ceiling (0.76 here).
    assert result["accuracy"] > 0.65
    assert result["accuracy"] > result["ideal_accuracy"] - 0.06


def test_rnn_integrates_early_evidence(trained_rnn):
    task, _, result = trained_rnn
    kernel = rnn.psychophysical_kernel(result["inputs"], result["coherence"], result["choice"], task,
                                       max_coherence=0.05)
    half = kernel.size // 2
    # A lone unit with tau = 100 ms keeps at most exp(-4) ~ 2% of evidence from the first half of
    # the stimulus (>= 400 ms before the go signal), so a clearly positive early kernel shows that
    # the recurrent dynamics hold the running sum, as a perfect integrator (flat kernel) would.
    assert kernel[:half].mean() > 0.4 * task.ideal_kernel()
    assert kernel[half:].mean() > 0.4 * task.ideal_kernel()


def test_bptt_gradient_matches_finite_differences():
    # float64 (conftest) and no recurrent noise make the loss a deterministic function of the
    # weights, smooth away from ReLU kinks, so central differences along a random direction agree
    # with the BPTT gradient up to O(eps^2) truncation and O(1e-16 / eps) rounding error.
    task = rnn.DecisionTask(fixation=40.0, stimulus=100.0, decision=60.0)
    model = rnn.RateRNN(2, 6, 1, sigma_rec=0.0, seed=3)
    trials = task.sample(np.random.default_rng(0), 4)
    train_vars = model.train_vars().unique()

    def loss():
        outputs, _ = rnn.simulate(model, trials["inputs"], task.dt)
        return rnn.masked_mse(outputs, trials["targets"], trials["mask"])

    grads = bm.grad(loss, grad_vars=train_vars)()
    loss_at = bm.jit(loss)
    rng, eps = np.random.default_rng(1), 1e-6
    for key, var in train_vars.items():
        direction = rng.standard_normal(var.shape)
        original = np.asarray(var.value)
        var.value = original + eps * direction
        plus = float(loss_at())
        var.value = original - eps * direction
        minus = float(loss_at())
        var.value = original
        analytic = float(np.sum(np.asarray(grads[key]) * direction))
        np.testing.assert_allclose(analytic, (plus - minus) / (2 * eps), rtol=1e-6, err_msg=key)


def test_simulate_leaves_global_dt_unchanged():
    # bp.share.save(dt=...) writes BrainPy's global dt in 2.8.2; simulate() must scope it.
    before = bm.get_dt()
    task = rnn.DecisionTask(dt=10.0, fixation=20.0, stimulus=40.0, decision=20.0)
    rnn.simulate(rnn.RateRNN(2, 4, 1), task.sample(np.random.default_rng(0), 2)["inputs"], task.dt)
    assert bm.get_dt() == before


# --------------------------------------------------------------------------- spiking network

@pytest.fixture(scope="module")
def trained_snn():
    task = snn.LatencyTask(n_classes=3, jitter=3.0, background_rate=5.0)
    model = snn.SpikingClassifier(task.n_inputs, 32, task.n_classes, seed=0)
    # A short run with a constant learning rate (lr_decay=1).
    history = snn.train(model, task, n_iterations=150, batch_size=32, lr=5e-3, lr_decay=1.0, seed=0, eval_every=0)
    return task, model, history


def test_latency_task_hides_the_class_in_spike_timing():
    # Without background spikes every input fires exactly once per trial in every class, so any
    # classifier of spike counts is at chance. Spike times, in contrast, identify the class: the
    # templates differ by 70/3 ~ 23 ms per input on average (uniform latencies over 70 ms),
    # far beyond the 5 ms jitter.
    task = snn.LatencyTask(background_rate=0.0)
    spikes, labels = task.sample(np.random.default_rng(0), 200)
    np.testing.assert_array_equal(spikes.sum(axis=0), 1.0)
    spike_time = spikes.argmax(axis=0) * task.dt  # (trial, input)
    distance = ((spike_time[:, None, :] - task.latencies[None]) ** 2).sum(axis=-1)
    assert np.mean(distance.argmin(axis=1) == labels) > 0.99


def test_surrogate_derivative_is_nonzero_where_heaviside_derivative_is_zero():
    # H(x) is flat on both sides of threshold: dH/dx = 0 for every x != 0 and a Dirac delta at 0,
    # so backpropagating through H carries no learning signal. The SuperSpike surrogate keeps H in
    # the forward pass but reports 1 / (beta |x| + 1)^2 > 0 in the backward pass.
    x = np.array([-2.0, -0.5, -0.1, 0.1, 0.5, 2.0])
    surrogate = bm.surrogate.InvSquareGrad(alpha=10.0)
    np.testing.assert_array_equal(np.asarray(surrogate(jnp.asarray(x))), x >= 0)
    np.testing.assert_array_equal(snn.spike_derivative(snn.heaviside, x), 0.0)
    np.testing.assert_allclose(snn.spike_derivative(surrogate, x), 1 / (10 * np.abs(x) + 1) ** 2)


def test_only_surrogate_gradients_reach_weights_upstream_of_spikes():
    # Same weights, same forward pass. With the true derivative the loss is piecewise constant in
    # the input weights (their gradient is exactly 0), although the readout weights downstream of
    # the spikes still get one. The surrogate also credits neurons that never fired.
    task = snn.LatencyTask(n_classes=3)
    spikes, labels = task.sample(np.random.default_rng(0), 16)
    grads, losses, hidden = {}, {}, {}
    for name, spk_fun in [("surrogate", None), ("heaviside", snn.heaviside)]:
        model = snn.SpikingClassifier(task.n_inputs, 16, task.n_classes, spk_fun=spk_fun, seed=0)

        def loss():
            readout, hidden_spikes = snn.simulate(model, spikes, task.dt)
            return snn.classification_loss(readout, labels)[0], hidden_spikes

        grad_fn = bm.grad(loss, grad_vars={"in": model.i2h.W, "out": model.h2o.W}, return_value=True, has_aux=True)
        g, value, hidden_spikes = grad_fn()
        grads[name] = {key: np.asarray(v) for key, v in g.items()}
        losses[name], hidden[name] = float(value), np.asarray(hidden_spikes)
    np.testing.assert_array_equal(hidden["surrogate"], hidden["heaviside"])
    assert losses["surrogate"] == losses["heaviside"]
    silent = hidden["surrogate"].sum(axis=(0, 1)) == 0
    assert np.all(grads["heaviside"]["in"] == 0.0)
    assert np.any(grads["heaviside"]["out"] != 0.0)
    assert silent.any() and np.all(np.abs(grads["surrogate"]["in"][:, silent]).sum(axis=0) > 0)


def test_snn_loss_decreases(trained_snn):
    task, _, history = trained_snn
    # Untrained class scores are nearly equal, so cross-entropy starts near ln(n_classes).
    chance_loss = np.log(task.n_classes)
    assert history["loss"][:10].mean() > 0.9 * chance_loss
    assert history["loss"][-20:].mean() < 0.4 * chance_loss


def test_snn_held_out_accuracy_above_chance(trained_snn):
    task, model, _ = trained_snn
    spikes, labels = task.sample(np.random.default_rng(123), 300)
    accuracy = np.mean(snn.predict(model, spikes, task.dt) == labels)
    assert accuracy > 0.8  # chance is 1/3, and spike counts alone cannot beat it
