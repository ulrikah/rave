import pytest

from rave.env import CrossAdaptiveEnv, CROSS_ADAPTIVE_DEFAULT_CONFIG
from rave.effect import Effect


def test_linear_mapping():
    action = 0.5
    min_value = 0.0
    max_value = 1.0
    skew_factor = 1.0
    assert (
        CrossAdaptiveEnv.map_action_to_effect_parameter(
            action, min_value, max_value, skew_factor
        )
        == 0.5
    )

    action = 0.5
    min_value = 50
    max_value = 20000
    skew_factor = 1.0
    assert (
        CrossAdaptiveEnv.map_action_to_effect_parameter(
            action, min_value, max_value, skew_factor
        )
        == 10025
    )

    action = 1.0
    min_value = 0.0
    max_value = 1.0
    skew_factor = 1.0
    assert (
        CrossAdaptiveEnv.map_action_to_effect_parameter(
            action, min_value, max_value, skew_factor
        )
        == 1.0
    )


def test_non_linear_mapping():
    action = 0.5
    min_value = 0.0
    max_value = 1.0
    skew_factor = 0.5

    assert (
        CrossAdaptiveEnv.map_action_to_effect_parameter(
            action, min_value, max_value, skew_factor
        )
        == 0.25
    )


def test_lower_skew_factors_yield_lower_value():
    action = 0.5
    min_value = 0.0
    max_value = 1.0
    skew_factors = [0.5, 0.3]

    assert CrossAdaptiveEnv.map_action_to_effect_parameter(
        action, min_value, max_value, skew_factors[0]
    ) > CrossAdaptiveEnv.map_action_to_effect_parameter(
        action, min_value, max_value, skew_factors[1]
    )


def test_non_linear_mapping_still_obtains_the_max_value():
    action = 1.0
    min_value = 0.0
    max_value = 1.0
    skew_factor = 0.1
    assert (
        CrossAdaptiveEnv.map_action_to_effect_parameter(
            action, min_value, max_value, skew_factor
        )
        == 1.0
    )