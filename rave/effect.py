import json
import sys
from types import SimpleNamespace
from random import uniform

import ctcsound

from .template_handler import TemplateHandler
from .player import Player
from .utils import play_wav, now


class Effect:
    def __init__(self, effect_name):
        """
        Effect object model

        Args:
            effect_name: string corresponding to an effect in the rave/effects/ folder
        """

        effect = self.parse_effect_from_json(
            f"rave/effects/{effect_name}.json")

        self.parameters = effect.parameters
        self.name = effect_name

    def random_mapping(self):
        """
        Generate a random mapping of all parameter values
        """
        effect_params = {}
        for param in self.parameters:
            effect_params[param.name] = uniform(param.mapping.min_value,
                                                param.mapping.max_value)
        return effect_params

    def parse_effect_from_json(self, effect_json_path: str):
        try:
            with open(effect_json_path, 'r') as file:
                data = file.read()
                effect = json.loads(
                    data, object_hook=lambda d: SimpleNamespace(**d))
                return effect
        except json.decoder.JSONDecodeError as error:
            print("Unable to parse effect", effect_json_path)
            raise error


def apply_effect(wav_file, effect):
    """
    Applies an effect to a sound source by generating a CSound orchestra

    Args:
        sound: path to a .wav file
        effect: an object representation of an effect
    Return:
        path to the generated .wav file
    """

    return


def main():
    effect = Effect("bandpass")
    INPUT_AUDIO = "rave/input_audio"
    sound_source = "amen.wav"

    player = Player()
    for _ in range(3):
        output_file_path = f"rave/bounces/{effect.name}_{now()}.wav"
        effect_params = effect.random_mapping()
        fx = TemplateHandler(
            f"{effect.name}.csd.jinja2").compile(effect_params)
        base = TemplateHandler("base_effect.csd.jinja2")
        csd = base.compile(
            input=f"{INPUT_AUDIO}/{sound_source}", output=output_file_path, flags="-W", effect=fx)
        player.render_csd(csd)

    player.cleanup(exit=True)


if __name__ == "__main__":
    main()
