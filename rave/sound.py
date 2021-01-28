import os
from effect import Effect
from template_handler import TemplateHandler
from tools import timestamp, get_duration, k_to_sec
from player import Player

# TODO: define these contants somewhere more globally accessible
LIVE = "adc"
AUDIO_INPUT_FOLDER = "rave/input_audio/"
AUDIO_OUTPUT_FOLDER = "rave/bounces/"
AUDIO_INPUT_FILE = "amen.wav"
FLAGS = "-W"  # write WAVE instead of AIFF
EFFECTS_TEMPLATE_DIR = "rave/effects"
SAMPLE_RATE = 44100
KSMPS = 64


class Sound:
    """
    Representation of a sound object onto which an effect can be applied

    TODO: add customization options for I/O, flags etc.
    """

    def __init__(self, filename):
        rel_path = os.path.join(AUDIO_INPUT_FOLDER, filename)
        assert os.path.isfile(rel_path), f"Couldn't find {rel_path}"
        self.filename = filename
        self.input_file_path = rel_path
        self.player = Player()

    def compile_effect(self, effect: Effect, mapping):
        return TemplateHandler(f"{effect.name}.csd.jinja2", template_dir=EFFECTS_TEMPLATE_DIR).compile(mapping)

    def compile_analyzer(self, osc_route: str):
        return TemplateHandler("analyzer.csd.jinja2", template_dir=EFFECTS_TEMPLATE_DIR).compile(osc_route=osc_route)

    def apply_effect(self, effect: Effect, mapping, analyzer_osc_route=False):
        """
        Applies an effect to the sound object

        Args:
            effect: which Effect to apply
            mapping: a dict of parameter values corresponding to the parameters in the Effect
            analyze: whether or not the bounce should be analyzed
        """
        effect_csd = self.compile_effect(effect, mapping)
        analyzer_csd = self.compile_analyzer(
            osc_route=analyzer_osc_route) if analyzer_osc_route else ""

        AUDIO_OUTPUT_FILE = f"{os.path.splitext(self.filename)[0]}_{effect.name}_{timestamp()}.wav"
        output_file_path = os.path.join(
            AUDIO_OUTPUT_FOLDER, AUDIO_OUTPUT_FILE)

        base_csd = TemplateHandler(
            "base.csd.jinja2", template_dir=EFFECTS_TEMPLATE_DIR)

        csd = base_csd.compile(
            input=self.input_file_path,
            output=output_file_path,
            sample_rate=SAMPLE_RATE,
            ksmps=KSMPS,
            flags="-W",
            effect=effect_csd,
            analyzer=analyzer_csd,
            length=get_duration(os.path.join(
                AUDIO_INPUT_FOLDER, AUDIO_INPUT_FILE))
        )

        self.player.render_csd(csd)
