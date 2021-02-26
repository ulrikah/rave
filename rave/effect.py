import json
import sys
import os
from types import SimpleNamespace
from random import uniform
from collections import namedtuple

import numpy as np

from rave.template_handler import TemplateHandler
from rave.player import Player
from rave.tools import timestamp, get_duration

from rave.constants import EFFECT_TEMPLATE_DIR, EFFECT_BASE, CSD_JINJA_SUFFIX

Channel = namedtuple("Channel", ["name", "value"])


class Effect:
    def __init__(self, effect_name):
        """
        Effect object model

        Args:
            effect_name: string corresponding to an effect in the rave/effects/ folder
        """

        effect = self.parse_effect_from_json(
            os.path.join(EFFECT_TEMPLATE_DIR, f"{effect_name}.json")
        )

        self.parameters = effect.parameters
        self.name = effect_name

    def mapping_from_numerical_array(self, array: np.ndarray):
        """
        Outputs effect parameters in JSON format based on a numerical array.
        Assumes that parameters have been generated from top to bottom.
        """
        assert len(array) == len(
            self.parameters
        ), "Number of params doesn't match length of action"
        mapping = {}
        for i, param in enumerate(self.parameters):
            mapping[param.name] = array[i]
        return mapping

    def get_csd_channels(self):
        """
        Returns an array of Channel namedtuples with a random mapping
        """
        mapping = self.mapping_from_numerical_array(self.random_numerical_mapping())
        return [Channel(name=name, value=value) for (name, value) in mapping.items()]

    def random_numerical_mapping(self):
        """
        Generate a random mapping of all parameter values as an array
        """
        numerical_mapping = [
            uniform(param.mapping.min_value, param.mapping.max_value)
            for param in self.parameters
        ]
        return numerical_mapping

    def parse_effect_from_json(self, effect_json_path: str):
        try:
            with open(effect_json_path, "r") as file:
                data = file.read()
                effect = json.loads(data, object_hook=lambda d: SimpleNamespace(**d))
                return effect
        except json.decoder.JSONDecodeError as error:
            print("Unable to parse effect", effect_json_path)
            raise error

    def to_csd(self):
        return TemplateHandler(
            f"{self.name}{CSD_JINJA_SUFFIX}", template_dir=EFFECT_TEMPLATE_DIR
        ).compile()


def main():
    effect = Effect("bandpass")
    INPUT_AUDIO = "rave/input_audio"
    sound_source = "amen.wav"

    input_file_path = os.path.join(INPUT_AUDIO, sound_source)

    player = Player(debug=False)
    live_mode = False
    for _ in range(4):
        output_file_path = f"rave/bounces/{effect.name}_{timestamp()}.wav"
        mapping = effect.mapping_from_numerical_array(effect.random_numerical_mapping())
        channels = [
            Channel(name=name, value=value) for (name, value) in mapping.items()
        ]
        effect_csd = effect.to_csd()
        base = TemplateHandler(EFFECT_BASE, template_dir=EFFECT_TEMPLATE_DIR)
        if live_mode:
            # Add OSC receivers and stuff
            raise NotImplementedError
        # Initialize channels
        csd = base.compile(
            sample_rate=44100,
            ksmps=64,
            input=f"-i{input_file_path}",
            output="-odac",
            flags="-W",
            effect=effect_csd,
            channels=channels,
            duration=get_duration(input_file_path),
        )
        player.render_csd(csd)

    player.cleanup(exit=True)


if __name__ == "__main__":
    main()