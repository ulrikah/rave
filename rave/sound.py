import os
from effect import Effect
from template_handler import TemplateHandler
from tools import timestamp
from player import Player

# TODO: define these contants somewhere more globally accessible
LIVE = "adc"
AUDIO_INPUT_FOLDER = "rave/input_audio/"
AUDIO_OUTPUT_FOLDER = "rave/bounces/"
AUDIO_INPUT_FILE = "amen.wav"
FLAGS = "-W"  # write WAVE instead of AIFF


class Sound:
    """
    Representation of a sound object onto which an effect can be applied
    """

    def __init__(self):
        # self.source = source
        self.player = Player()

    def compile_effect(self, effect: Effect, mapping):
        AUDIO_OUTPUT_FILE = f"{effect.name}_{timestamp()}.wav"
        base_csd = TemplateHandler("base_effect.csd.jinja2")
        effect_csd = TemplateHandler(
            f"{effect.name}.csd.jinja2").compile(mapping)
        csd = base_csd.compile(
            input=os.path.join(AUDIO_INPUT_FOLDER, AUDIO_INPUT_FILE),
            output=os.path.join(AUDIO_OUTPUT_FOLDER, AUDIO_OUTPUT_FILE),
            flags="-W",
            effect=effect_csd)
        return csd

    def apply_effect(self, effect: Effect, mapping):
        csd = self.compile_effect(effect, mapping)
        self.player.render_csd(csd)
