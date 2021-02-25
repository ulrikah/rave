import os
import wave

import numpy as np

from rave.effect import Effect
from rave.analyser import Analyser
from rave.template_handler import TemplateHandler
from rave.tools import timestamp, get_duration, k_to_sec
from rave.player import Player

# TODO: define these contants somewhere more globally accessible

LIVE = "adc"
NO_SOUND = "--nosound"
DAC = "dac"
AUDIO_INPUT_FOLDER = "/Users/ulrikah/fag/thesis/rave/rave/input_audio/"
AUDIO_OUTPUT_FOLDER = "/Users/ulrikah/fag/thesis/rave/rave/bounces/"
CSD_FOLDER = "/Users/ulrikah/fag/thesis/rave/rave/csd"
EFFECTS_TEMPLATE_DIR = "/Users/ulrikah/fag/thesis/rave/rave/effects"
SAMPLE_RATE = 44100
KSMPS = 64


class Sound:
    """
    Representation of a sound object onto which an effect can be applied

    Args:
        input_source: path to a WAVE file or 'adc' for live input
        output_file_path: where to save the rendered audio
        loop: if the audio files should be wrapped at the end of the file
        duration: sets the render duration for LIVE input. Duration is dictacted by the length of the audio file for static files
    """

    def __init__(self, input_source, output_file_path=None, loop=True, duration=10):
        if input_source == LIVE:
            self.save_to = LIVE
            self.input = LIVE
            self.duration = duration  # NOTE: unsure how to better specify this
        else:
            if os.path.isfile(input_source):
                abs_path = os.path.abspath(input_source)
            elif os.path.isfile(os.path.join(AUDIO_INPUT_FOLDER, input_source)):
                abs_path = os.path.abspath(
                    os.path.join(AUDIO_INPUT_FOLDER, input_source)
                )
            else:
                raise IOError(f"Couldn't find file {input_source}")
            self.save_to = os.path.splitext(os.path.basename(abs_path))[0]
            self.input = abs_path
            _, _, self.duration = self.get_properties(abs_path)

        if output_file_path is None:
            self.output = NO_SOUND
            self.flags = ""
        else:
            self.output = os.path.join(AUDIO_OUTPUT_FOLDER, output_file_path)
            self.flags = "-W"  # write as WAVE file

        self.player = None
        self.loop = loop
        self.csd = None

    @staticmethod
    def get_properties(wav_path):
        with wave.open(wav_path, "rb") as wav:
            frame_rate = wav.getframerate()
            n_frames = wav.getnframes()
            duration = n_frames / frame_rate
            return frame_rate, n_frames, duration

    def prepare_to_render(self, effect: Effect = None, analyser: Analyser = None):
        """
        Prepares the Sound to be rendered by compiling the CSD templates.

        Args:
            effect: which Effect to apply, potentially None if no effect is desired
            analyser: an Analyser object, potentially None if the Sound doesn't need to be analysed
        """

        effect_csd = effect.to_csd() if effect is not None else None

        base = TemplateHandler("base.csd.jinja2", template_dir=EFFECTS_TEMPLATE_DIR)
        channels = effect.get_csd_channels() if effect is not None else []

        self.csd = base.compile(
            input=f"-i{self.input}",
            output=f"-o{self.output}" if self.output != NO_SOUND else self.output,
            channels=channels,
            sample_rate=SAMPLE_RATE,
            ksmps=KSMPS,
            flags=self.flags,
            effect=effect_csd,
            analyser=analyser.analyser_csd if analyser is not None else "",
            duration=self.duration,
        )
        save_to_path = os.path.join(
            CSD_FOLDER,
            f"{self.save_to}_{timestamp()}.csd",
        )
        base.save_to_file(save_to_path)
        return self.csd

    def render(self, mapping=None):
        """
        Applies the mapping to the sound object

        Args:
            mapping: a dict of parameter values corresponding to the parameters in the Effect

        Returns:
            a boolean indicating if the rendered frame was the last one (True) or not (False)
        """
        if self.csd is None:
            raise Exception("render is called prior to prepare_to_render")

        if self.player is None:
            self.player = Player(debug=True)
            self.player.start_halting(self.csd)

        if mapping is not None:
            self.player.set_channels(mapping)

        done = self.player.render_one_frame(loop=self.loop)
        return done

    def bounce(self):
        """
        Renders a sound until it ends. Creates a new Player instance to start fresh

        Returns:
            the path to the bounce
        """
        assert not self.output.startswith(
            DAC
        ), "Can't bounce a sound that is sent to DAC"
        if self.player:
            self.player.cleanup()
            del self.player
        self.player = Player()
        self.player.start_halting(self.csd)
        self.player.render_until_end()
        return self.output
