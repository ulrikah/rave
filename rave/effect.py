import json
import sys
from types import SimpleNamespace
from random import uniform

import ctcsound

from .template_handler import TemplateHandler
from .player import Player


def parse_json_effect(effect_json_path: str):
    try:
        with open(effect_json_path, 'r') as file:
            data = file.read()
            fx = json.loads(data, object_hook=lambda d: SimpleNamespace(**d))
            return fx
    except json.decoder.JSONDecodeError as error:
        print("Unable to parse effect", effect_json_path)
        raise error


def apply_effect(sound, effect):
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
    effect_name = "distortion"
    effect_json = parse_json_effect(f"rave/effects/{effect_name}.json")

    player = Player()
    for _ in range(3):
        effect_params = {}
        for param in effect_json.parameters:
            effect_params[param.name] = uniform(param.mapping.min_value,
                                                param.mapping.max_value)

        bandpass = TemplateHandler(
            f"{effect_name}.csd.jinja2").compile(effect_params)
        base = TemplateHandler("base_effect.csd.jinja2")
        csd = base.compile(
            wav_file="rave/test_audio/amen.wav", effect=bandpass)

        player.play_csd_string(csd)

    player.cleanup(exit=True)


if __name__ == "__main__":
    main()
