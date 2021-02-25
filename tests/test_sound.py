import numpy as np

from rave.sound import Sound
from rave.effect import Effect
from rave.analyser import Analyser


def test_dry_and_wet_are_not_the_same():
    amen = "amen_trim.wav"
    feature_extractors = ["rms"]
    analyser = Analyser(feature_extractors)
    analysis_channels = analyser.analysis_features
    effect = Effect("bandpass")
    dry = Sound(amen)
    dry.prepare_to_render(analyser=analyser)
    wet = Sound(amen)
    wet.prepare_to_render(effect=effect, analyser=analyser)

    for i in range(100):
        dry.render()
        wet.render()
        dry_chans = dry.player.get_channels(analysis_channels)
        wet_chans = wet.player.get_channels(analysis_channels)
        assert not np.array_equal(
            dry_chans, wet_chans
        ), "Dry and wet should not be equal"


def test_two_dry_signals_yield_the_same_features():
    amen = "amen_trim.wav"
    feature_extractors = ["rms"]
    analyser = Analyser(feature_extractors)
    analysis_channels = analyser.analysis_features
    dry1 = Sound(amen)
    dry1.prepare_to_render(analyser=analyser)
    dry2 = Sound(amen)
    dry2.prepare_to_render(analyser=analyser)

    for i in range(100):
        dry1.render()
        dry2.render()
        dry1_chans = dry1.player.get_channels(analysis_channels)
        dry2_chans = dry2.player.get_channels(analysis_channels)
        assert np.array_equal(
            dry1_chans, dry2_chans
        ), "Two dry signals should yield the same features"
