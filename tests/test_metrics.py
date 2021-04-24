from rave.metrics import RelativeGain
import numpy as np


def test_relative_gain_equal_distance_gives_zero_reward():
    metrics = RelativeGain()
    target = np.array([1.0, 1.0], dtype=float)
    dry = np.array([0.5, 0.5], dtype=float)
    wet = np.array([0.5, 0.5], dtype=float)
    reward = metrics.calculate_reward(dry, wet, target)
    assert reward == 0.0


def test_relative_gain_when_dry_is_closer_reward_is_negative():
    metrics = RelativeGain()
    target = np.array([1.0, 1.0], dtype=float)
    dry = np.array([0.5, 0.75], dtype=float)
    wet = np.array([0.5, 0.5], dtype=float)
    reward = metrics.calculate_reward(dry, wet, target)
    assert reward < 0


def test_relative_gain_when_wet_is_closer_reward_is_positive():
    metrics = RelativeGain()
    target = np.array([1.0, 1.0], dtype=float)
    dry = np.array([0.5, 0.5], dtype=float)
    wet = np.array([0.5, 0.75], dtype=float)
    reward = metrics.calculate_reward(dry, wet, target)
    assert reward > 0


def test_relative_gain_when_wet_is_as_close_as_possible_reward_is_the_distance_between_dry_and_target():
    """Assumes range is (0, 1)"""
    metrics = RelativeGain()
    target = np.array([1.0, 1.0], dtype=float)
    dry = np.array([0.0, 0.0], dtype=float)
    wet = np.array([1.0, 1.0], dtype=float)
    reward = metrics.calculate_reward(dry, wet, target)
    assert reward == np.linalg.norm(dry - target)

    dry = np.array([-10.0, -10.0], dtype=float)
    reward = metrics.calculate_reward(dry, wet, target)
    assert reward == np.linalg.norm(dry - target)
