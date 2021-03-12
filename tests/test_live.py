import pytest
import numpy as np

import os

from rave.analyser import Analyser
from rave.musician import Musician
from rave.mediator import Mediator
from rave.constants import AUDIO_INPUT_DIR, KSMPS, SAMPLE_RATE, NO_SOUND
from rave.tools import k_per_sec, get_duration

INPUT_SOURCE = os.path.join(AUDIO_INPUT_DIR, "noise.wav")


@pytest.mark.skip
def test_mediator_receives_values_from_musician():
    dur_s = get_duration(INPUT_SOURCE)
    dur_k = round(k_per_sec(KSMPS, SAMPLE_RATE) * dur_s)
    analyser = Analyser(
        ["rms", "mfcc"], osc_route="/rave/source/features", audio_to_analyse="aOut"
    )
    mediator = Mediator(run=False)
    musician = Musician(
        input_source=INPUT_SOURCE,
        output_source=NO_SOUND,
        analyser=analyser,
        duration=dur_s,
    )
    mediator.run()
    musician.start()
    mediator.terminate()

    assert mediator.source_q.qsize() == dur_k
    assert np.array(mediator.get_source_features()).mean() > 0
    assert np.array(mediator.get_source_features()).size == len(
        analyser.analysis_features
    )
