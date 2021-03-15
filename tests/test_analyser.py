import pytest
import ctcsound
import numpy as np

from rave.analyser import Analyser


def test_analyser_initialisation():
    features = ["rms", "pitch"]
    analyser = Analyser(features)
    for feature in features:
        assert f"START {feature}" in analyser.analyser_csd
        assert f"END {feature}" in analyser.analyser_csd


def test_analyser_raises_error_on_unknown_extractor():
    features = ["unkwown_extractor"]
    with pytest.raises(ValueError):
        analyser = Analyser(features)


def test_output_channels_exist():
    features = ["rms", "pitch", "spectral"]
    analyser = Analyser(features)
    for extractor in analyser.feature_extractors:
        for channel in extractor["channels"]:
            assert f'"{channel}"' in analyser.analyser_csd


def test_all_extractors_use_same_audio_input():
    features = ["rms", "pitch"]
    analyser = Analyser(features)
    assert analyser.analyser_csd.count("aAnalyserInput") == len(features) + 1


def test_globals_are_included():
    features = ["spectral"]
    analyser = Analyser(features)
    global_variables = ["gifftsize", "giFftTabSize", "gifna", "gifnf"]
    for gv in global_variables:
        assert gv in analyser.analyser_csd


def test_osc_route_works():
    features = ["rms", "spectral"]
    osc_route = "/test/test/test"
    analyser = Analyser(features, osc_route=osc_route)
    assert "OSCsend" in analyser.analyser_csd
    fff = f"\"{'f' * len(analyser.analysis_features)}\""
    assert f'{fff}, {", ".join(analyser.analysis_features)}' in analyser.analyser_csd
    assert osc_route in analyser.analyser_csd


def test_feature_extractors_output_something():
    feature_extractors = ["pitch", "spectral", "mfcc"]
    audio_to_analyse = "aSig"
    for fe in feature_extractors:
        # the other extractors depend on RMS for now
        analyser = Analyser(["rms", fe], audio_to_analyse=audio_to_analyse)
        analysis_features = analyser.analysis_features
        orc = f"""
        sr=44100
        ksmps=64
        nchnls=1
        0dbfs=1

        instr 1
        {audio_to_analyse} poscil 1.0, 220
        out {audio_to_analyse}
        {analyser.analyser_csd}
        endin
        """

        sco = """
        i1 0 3
        """

        cs = ctcsound.Csound()
        cs.setOption("--nosound")

        cs.compileOrc(orc)
        cs.readScore(sco)

        cs.start()
        features = []

        while cs.performBuffer() == 0:
            features.append(
                [cs.controlChannel(feature)[0] for feature in analysis_features]
            )
        features = np.array(features)
        for i in range(len(analysis_features)):
            assert features[:, i].mean() > 0.0
        cs.cleanup()
        cs.reset()
        del cs
