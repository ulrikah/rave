import numpy as np

from rave.env import CrossAdaptiveEnv, CROSS_ADAPTIVE_DEFAULT_CONFIG
from rave.effect import Effect
from rave.constants import DEBUG_SUFFIX


def test_linear_mapping():
    action = 0.5
    min_value = 0.0
    max_value = 1.0
    skew_factor = 1.0
    assert (
        CrossAdaptiveEnv.map_action_to_effect_parameter(
            action, min_value, max_value, skew_factor
        )
        == 0.75
    )

    action = 0.0
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
        == 0.5625
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


def test_debug_mode_defines_debug_channels():
    config = {
        **CROSS_ADAPTIVE_DEFAULT_CONFIG,
        "debug": True,
    }
    env = CrossAdaptiveEnv(config)
    action = env.action_space.sample()
    env.step(action)
    source = env.render()
    debug_channels = list(
        map(lambda param: f"{param.name}{DEBUG_SUFFIX}", env.effect.parameters)
    )
    for ch in debug_channels:
        assert f'chn_k "{ch}"' in source.csd


def test_non_debug_mode_does_not_define_debug_channels():
    config = {
        **CROSS_ADAPTIVE_DEFAULT_CONFIG,
        "debug": False,
    }
    env = CrossAdaptiveEnv(config)
    debug_channels = list(
        map(lambda param: f"{param.name}{DEBUG_SUFFIX}", env.effect.parameters)
    )
    action = env.action_space.sample()
    env.step(action)
    source = env.render()
    for ch in debug_channels:
        assert f'chn_k "{ch}"' not in source.csd


def test_debug_mode_sets_debug_channels():
    config = {
        **CROSS_ADAPTIVE_DEFAULT_CONFIG,
        "debug": True,
    }
    env = CrossAdaptiveEnv(config)
    debug_channels = list(
        map(lambda param: f"{param.name}{DEBUG_SUFFIX}", env.effect.parameters)
    )

    action = env.action_space.sample()
    env.step(action)
    source = env.render()
    debug_values = source.player.get_channels(debug_channels)
    for v in debug_values:
        assert env.action_space.low[0] < v < env.action_space.high[1]


def test_debug_mode_renders_channels_to_debug_wave_file():
    config = {
        **CROSS_ADAPTIVE_DEFAULT_CONFIG,
        "debug": True,
    }
    env = CrossAdaptiveEnv(config)
    debug_channels = list(
        map(lambda param: f"{param.name}{DEBUG_SUFFIX}", env.effect.parameters)
    )
    action = env.action_space.sample()
    env.step(action)
    source = env.render()
    assert "fout" in source.csd
    for ch in debug_channels:
        assert f"upsamp(k_{ch})" in source.csd


def test_env_inits_and_makes_first_step_correctly():
    env = CrossAdaptiveEnv()
    empty_features = np.zeros(shape=len(env.analysis_features))
    initial_state = env.get_state()
    assert np.array_equal(
        initial_state, np.concatenate((empty_features, empty_features))
    )
    action = env.action_space.sample()
    state, reward, done, _ = env.step(action)
    assert done is True
    assert reward == 0.0
    assert not np.array_equal(state, initial_state)
    assert np.abs(state.mean()) > 0.0


def test_source_wet_is_delayed_by_one_k():
    env = CrossAdaptiveEnv()
    action = env.action_space.sample()
    N = 2
    for _ in range(N):
        env.step(action)
    assert env.source_dry.player.k == N
    assert env.target.player.k == N
    assert env.source_wet.player.k == N - 1


def test_source_wet_is_equal_to_previous_source_dry_when_effect_is_thru():
    config = {**CROSS_ADAPTIVE_DEFAULT_CONFIG, "effect": Effect("thru")}
    env = CrossAdaptiveEnv(config)
    env.step(env.action_space.sample())
    source_dry_features_after_first_step = env.source_dry_features.copy()
    source_wet_features_after_first_step = env.source_wet_features.copy()
    assert np.array_equal(
        source_wet_features_after_first_step, np.zeros(shape=len(env.analysis_features))
    )
    env.step(env.action_space.sample())
    source_dry_features_after_second_step = env.source_dry_features.copy()
    source_wet_features_after_second_step = env.source_wet_features.copy()
    assert np.array_equal(
        source_wet_features_after_second_step, source_dry_features_after_first_step
    )
    assert not np.array_equal(
        source_dry_features_after_second_step, source_dry_features_after_first_step
    )


def test_source_wet_wraps_correctly_at_the_end_of_the_sound():
    config = {
        **CROSS_ADAPTIVE_DEFAULT_CONFIG,
        "eval_interval": None,  # episode is done at the end of the source
    }
    env = CrossAdaptiveEnv(config)
    action = env.action_space.sample()
    assert env.should_delay_source_wet_one_frame is True
    done = False
    while not done:
        _, _, done, _ = env.step(action)
    assert env.source_dry.player.k == 0
    assert env.source_wet.player.k > 0
    assert env.should_delay_source_wet_one_frame is False
    _, _, done, _ = env.step(action)
    assert env.source_wet.player.k == 0
    assert env.source_dry.player.k == 1


def test_target_and_source_dry_has_no_effects():
    env = CrossAdaptiveEnv()
    assert "aOut = aIn" in env.source_dry.csd
    assert "aOut = aIn" in env.target.csd
    assert "aOut = aIn" not in env.source_wet.csd
