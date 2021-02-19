import pytest

from rave.analyser import Analyser


def test_analyser_initialisation():
    features = ["rms", "pitch"]
    analyser = Analyser(features)
    for feature in features:
        assert f"START {feature}" in analyser.analyser_csd
        assert f"END {feature}" in analyser.analyser_csd


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
    fff = f"\"{'f' * len(analyser.channels)}\""
    assert f'{fff}, {", ".join(analyser.channels)}' in analyser.analyser_csd
    assert f"kwhen += km" in analyser.analyser_csd
    assert osc_route in analyser.analyser_csd