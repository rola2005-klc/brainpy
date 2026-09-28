import numpy as np
import pytest

from neuromodels.networks.decision import PARAMS, fixed_points
from neuromodels.networks.need_choice import (FOOD, MISS, WATER, bout_lengths, goal_states, need_stimulus,
                                              shuffled_stay_probabilities, simulate_sessions, stay_probabilities,
                                              switch_needs)


def stable(thirst, hunger):
    return [point for point in goal_states(thirst, hunger) if point.kind == "stable"]


def test_needs_map_onto_the_decision_circuit_inputs():
    # Drives mu_max T and mu_max H written as mu0 (1 ± c'/100).
    mu0, coherence = need_stimulus(1.0, 0.5, mu_max=40.0)
    assert mu0 == pytest.approx(30.0) and coherence == pytest.approx(100.0 / 3.0)
    assert need_stimulus(0.0, 0.0) == (0.0, 0.0)


def test_goal_states_exist_only_while_a_need_drives_them():
    assert len(stable(0.0, 0.0)) == 1  # sated: only the disengaged state
    both = stable(1.0, 1.0)
    assert len(both) == 2 and {point.S1 > point.S2 for point in both} == {True, False}
    thirsty = stable(1.0, 0.2)
    assert len(thirsty) == 1 and thirsty[0].S1 > thirsty[0].S2  # only the water goal survives


def test_decision_strength_recurrence_would_keep_goals_without_need():
    # Chapter 08's J_same = 0.2609 nA holds a goal with no input at all: a sated mouse would keep seeking.
    assert sum(point.kind == "stable" for point in fixed_points(0.0, 0.0, PARAMS)) == 3


def test_bout_bookkeeping_skips_misses():
    choices = np.array([[WATER, WATER, MISS, WATER, FOOD, FOOD, WATER]]).T
    # Engaged sequence W W W F F W: after water 2 of 3 stay, after food 1 of 2.
    stay = stay_probabilities(choices)
    assert stay["water"] == pytest.approx(2 / 3) and stay["food"] == pytest.approx(1 / 2)
    np.testing.assert_array_equal(bout_lengths(choices, WATER), [3, 1])
    np.testing.assert_array_equal(bout_lengths(choices, FOOD), [2])


@pytest.fixture(scope="module")
def sessions():
    """Eight 80-minute sessions from full thirst and hunger at each noise level."""
    return {sigma: simulate_sessions(8, 320, sigma=sigma, seed=5) for sigma in (0.02, 0.04)}


def test_sessions_are_reproducible(sessions):
    again = simulate_sessions(8, 320, sigma=0.04, seed=5)
    np.testing.assert_array_equal(sessions[0.04]["choice"], again["choice"])


def test_choices_come_in_bouts(sessions):
    # Shuffling a session's choices keeps their proportions but destroys persistence (stay ≈ 0.5).
    for out in sessions.values():
        stay, chance = stay_probabilities(out["choice"]), shuffled_stay_probabilities(out["choice"])
        assert stay["water"] > chance["water"] + 0.3 and stay["food"] > chance["food"] + 0.3


def test_rewards_lower_needs_until_the_mouse_stops(sessions):
    for out in sessions.values():
        choice = out["choice"]
        np.testing.assert_allclose(out["thirst"][-1], 1.0 - 0.01 * (choice == WATER).sum(axis=0))
        np.testing.assert_allclose(out["hunger"][-1], 1.0 - 0.01 * (choice == FOOD).sum(axis=0))
        assert np.mean(choice[:40] != MISS) > 0.9  # engaged while needy
        assert np.mean(choice[-80:] != MISS) < 0.2  # disengaged once sated


def test_the_larger_need_is_served_first():
    out = simulate_sessions(8, 20, thirst=1.0, hunger=0.3, seed=6)
    assert np.mean(out["choice"] == WATER) > 0.95


def test_low_noise_switches_show_hysteresis(sessions):
    # Without much noise a bout ends only once its need has fallen well below the other.
    low = switch_needs(sessions[0.02]["choice"], sessions[0.02]["thirst"], sessions[0.02]["hunger"])
    assert np.median(low["to_food"]) < -0.08 and np.median(low["to_water"]) > 0.08
    # With more noise the switches cluster where the needs are equal.
    high = switch_needs(sessions[0.04]["choice"], sessions[0.04]["thirst"], sessions[0.04]["hunger"])
    assert abs(np.median(high["to_food"])) < 0.05 and abs(np.median(high["to_water"])) < 0.05
