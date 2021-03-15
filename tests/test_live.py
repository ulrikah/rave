import pytest
import numpy as np

import os
import time

from rave.analyser import Analyser
from rave.musician import Musician
from rave.effect import Effect
from rave.mediator import Mediator
from rave.constants import (
    AUDIO_INPUT_DIR,
    KSMPS,
    SAMPLE_RATE,
    NO_SOUND,
    OSC_SOURCE_FEATURES_ROUTE,
)
from rave.tools import k_per_sec, get_duration
from rave.osc_client import OscClient

INPUT_SOURCE = os.path.join(AUDIO_INPUT_DIR, "noise.wav")


def test_mediator_uses_lifo_queue_of_size_1():
    mediator = Mediator(run=False)
    assert mediator.source_q.maxsize == 1
    a = [1, 2, 3]
    b = [4, 5, 6]
    mediator.add_source_features("/test", a)
    mediator.add_source_features("/test", b)
    source_features = mediator.get_source_features(blocking=False)
    features = np.array(source_features)
    assert (features == np.array(b)).all()
    assert (features != np.array(a)).any()
    source_features = mediator.get_source_features(blocking=False)
    assert source_features is None


def test_mediator_receives_values_from_musician():
    dur_s = get_duration(INPUT_SOURCE)
    dur_k = round(k_per_sec(KSMPS, SAMPLE_RATE) * dur_s)
    analyser = Analyser(
        ["rms", "mfcc"], osc_route=OSC_SOURCE_FEATURES_ROUTE, audio_to_analyse="aOut"
    )
    mediator = Mediator(run=False)
    effect = Effect("bandpass")
    musician = Musician(
        effect=effect,
        analyser=analyser,
        input_source=INPUT_SOURCE,
        output_source=NO_SOUND,
        duration=dur_s,
    )
    mediator.run()
    musician.start()
    mediator.terminate()

    source_features = mediator.get_source_features(blocking=False)
    assert np.array(source_features).mean() > 0
    assert np.array(source_features).size == len(analyser.analysis_features)
