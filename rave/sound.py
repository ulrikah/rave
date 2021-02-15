import os
import wave

import numpy as np

from effect import Effect
from template_handler import TemplateHandler
from tools import timestamp, get_duration, k_to_sec
from player import Player

# TODO: define these contants somewhere more globally accessible
AMEN = "amen_trim.wav"
NOISE = "noise.wav"

LIVE = "adc"
NO_SOUND = "--nosound"
DAC = "dac"
AUDIO_INPUT_FOLDER = "/Users/ulrikah/fag/thesis/rave/rave/input_audio/"
AUDIO_OUTPUT_FOLDER = "/Users/ulrikah/fag/thesis/rave/rave/bounces/"
AUDIO_INPUT_FILE = AMEN
FLAGS = "-W"  # write WAVE instead of AIFF
EFFECTS_TEMPLATE_DIR = "/Users/ulrikah/fag/thesis/rave/rave/effects"
SAMPLE_RATE = 44100
KSMPS = 64


class Sound:
    """
    Representation of a sound object onto which an effect can be applied

    TODO: add customization options for I/O, flags etc.
    """

    def __init__(self, filename, output_file_path=None, loop=True):
        rel_path = os.path.join(AUDIO_INPUT_FOLDER, filename)
        assert os.path.isfile(
            rel_path), f"Couldn't find {rel_path} from {os.getcwd()}"
        self.filename = filename
        self.input_file_path = rel_path
        if output_file_path is None:
            # TODO: don't use -o dac at all when --nosound flag is set
            self.output = f"{DAC} {NO_SOUND}"
        else:
            self.output = os.path.join(AUDIO_OUTPUT_FOLDER, output_file_path)
        self.get_properties()
        self.player = None
        self.loop = loop
        self.csd = None

    def get_properties(self):
        with wave.open(self.input_file_path, "rb") as wav:
            self.frame_rate = wav.getframerate()
            self.n_frames = wav.getnframes()
            self.n_sec = self.n_frames / self.frame_rate

    @staticmethod
    def compile_effect(effect: Effect):
        return TemplateHandler(f"{effect.name}.csd.jinja2", template_dir=EFFECTS_TEMPLATE_DIR).compile()

    @staticmethod
    def compile_analyzer(osc_route: str = "/rave/target/features"):
        return TemplateHandler("analyzer.csd.jinja2", template_dir=EFFECTS_TEMPLATE_DIR).compile(osc_route=osc_route)

    def apply_effect(self, effect: Effect = None, analyze=True, osc_route=None):
        """
        Applies an effect to the sound object

        Args:
            effect: which Effect to apply, potentially blank for target sound
            analyze: whether or not the sound should be analyzed
        """
        effect_csd = self.compile_effect(
            effect) if effect is not None else None
        analyzer_csd = self.compile_analyzer(
            osc_route=osc_route) if analyze else ""

        base = TemplateHandler(
            "base.csd.jinja2", template_dir=EFFECTS_TEMPLATE_DIR)
        channels = effect.get_csd_channels() if effect is not None else []
        self.csd = base.compile(
            input=self.input_file_path,
            output=self.output,
            channels=channels,
            sample_rate=SAMPLE_RATE,
            ksmps=KSMPS,
            flags="-W",
            effect=effect_csd,
            analyzer=analyzer_csd,
            duration=get_duration(os.path.join(
                AUDIO_INPUT_FOLDER, AUDIO_INPUT_FILE))
        )
        save_to_path = os.path.join(
            "/Users/ulrikah/fag/thesis/rave/rave/csd", f"{os.path.splitext(self.filename)[0]}_{timestamp()}.csd")
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
            raise Exception("render is called prior to apply_effect")

        if self.player is None:
            self.player = Player()
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
            DAC), "Can't bounce a sound that is sent to DAC"
        if self.player:
            self.player.cleanup()
            del self.player
        self.player = Player()
        self.player.start_halting(self.csd)
        self.player.render_until_end()
        return self.output


if __name__ == "__main__":
    ANALYSIS_CHANNELS = ["rms", "pitch_n", "centroid", "flux"]
    fx = Effect("bandpass")
    dry = Sound(AMEN)
    dry.apply_effect()
    wet = Sound(AMEN)
    wet.apply_effect(fx)

    for i in range(100):
        dry.render()
        wet.render()
        dry_chans = dry.player.get_channels(ANALYSIS_CHANNELS)
        wet_chans = wet.player.get_channels(ANALYSIS_CHANNELS)
        assert not np.array_equal(
            dry_chans, wet_chans), "Dry and wet should not be equal"
