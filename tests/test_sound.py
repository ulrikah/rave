from rave.sound import Sound
from rave.effect import Effect
from rave.analyser import Analyser
from rave.constants import KSMPS


def test_dry_and_wet_are_not_the_same():
    amen = "amen_trim.wav"
    feature_extractors = ["rms"]
    analyser = Analyser(feature_extractors)
    analysis_channels = analyser.analysis_features

    for effect_name in [
        "bandpass",
        "formant",
        "dist_lpf",
        "freeverb",
        "distortion",
        "gain",
    ]:
        effect = Effect(effect_name)
        dry = Sound(amen)
        dry.prepare_to_render(analyser=analyser)
        wet = Sound(amen)
        wet.prepare_to_render(effect=effect, analyser=analyser)

        for i in range(10):
            dry.render()
            wet.render()
            dry_chans = dry.player.get_channels(analysis_channels)
            wet_chans = wet.player.get_channels(analysis_channels)
            assert (
                dry_chans != wet_chans
            ), f"Dry and wet should not be equal for {effect_name}"


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
        assert (
            dry1_chans == dry2_chans
        ), "Two dry signals should yield the same features"


def test_fftsize_is_twice_the_size_of_ksmps():
    sound = Sound("amen.wav")
    sound.prepare_to_render()
    assert f"ksmps = {KSMPS}" in sound.csd
    assert f"gifftsize = {KSMPS * 2}" in sound.csd
