import pytest

from rave.effect import Effect


def test_effect_initialisation():
    effect = Effect("bandpass")

    assert effect.parameters[0].name == "cutoff_freq"
    assert effect.parameters[0].mapping.min_value == 50.0
    assert effect.parameters[0].mapping.max_value == 5000.0
    assert effect.parameters[0].mapping.skew_factor == 0.3

    assert effect.parameters[1].name == "bandwidth"
    assert effect.parameters[1].mapping.min_value == 0.01
    assert effect.parameters[1].mapping.max_value == 1.0
    assert effect.parameters[1].mapping.skew_factor == 1.0


def test_gets_csd_channels():
    effect = Effect("bandpass")
    channels = effect.get_csd_channels()
    channel_names = [channel.name for channel in channels]
    assert channel_names == ["cutoff_freq", "bandwidth"]