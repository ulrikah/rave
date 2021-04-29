from rave.analyser import Analyser
from rave.sound import Sound
from rave.constants import (
    LIVE,
    DAC,
    ADC,
    OSC_TARGET_FEATURES_ROUTE,
    OSC_SOURCE_FEATURES_ROUTE,
)
from rave.config import parse_config_file
from rave.effect import Effect

import subprocess
import argparse


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config_file",
        action="store",
        required=False,
        default="default.toml",
        help="Path to a config file",
    )
    parser.add_argument(
        "--target",
        dest="is_target",
        action="store_true",
        default=False,
        help="Whether or not this musician represents the target",
    )
    parser.add_argument(
        "--live",
        dest="live_mode",
        action="store_true",
        default=False,
        help="Enable live input from adc",
    )
    return parser.parse_args()


class Musician:
    """
    This class represents a virtual musician as a continous stream of OSC messages simulating live input
    """

    def __init__(
        self,
        analyser: Analyser,
        effect: Effect = None,
        input_source: str = None,
        output_source: str = None,
        duration=10,
        is_target=False,
    ):
        if input_source is None:
            input_source_index = self.choose_audio_device(
                "🎤 Choose the index of the INPUT device you want to use: "
            )
            input_source = f"{ADC}{input_source_index}"
        if output_source is None:
            output_source_index = self.choose_audio_device(
                "🎤 Choose the index of the OUTPUT device you want to use: "
            )
            output_source = f"{DAC}{output_source_index}"
        self.analyser = analyser
        self.effect = effect
        self.sound = Sound(input_source, output_source, duration=duration)
        self.is_target = is_target

    @staticmethod
    def list_audio_devices():
        subprocess.call(["csound", "--devices", "-m", "128"])
        return

    def choose_audio_device(self, message):
        self.list_audio_devices()
        print("\n")
        device = input(message)
        print("\n")
        return device

    def start(self):
        print(
            """
        ╦  ╦╦  ╦╔═╗
        ║  ║╚╗╔╝║╣
        ╩═╝╩ ╚╝ ╚═╝
        """
        )
        _ = self.sound.prepare_to_render(
            effect=self.effect,
            analyser=self.analyser,
            receive_mapping_over_osc=not self.is_target,
        )
        return self.sound.stream()


def main():
    args = arguments()
    config = parse_config_file(args.config_file)

    # NOTE: at the current moment, this was the index of Blackhole,
    # a virtual soundcard. this functions as a sink for all audio
    BLACKHOLE = "dac2"

    if args.is_target:
        osc_route = OSC_TARGET_FEATURES_ROUTE
        # NOTE: temporary hack to loop the target sound
        input_source = "amen_loop.wav"
        effect = None
        output_source = BLACKHOLE
    else:
        osc_route = OSC_SOURCE_FEATURES_ROUTE
        # NOTE: temporary hack to loop the source sound
        input_source = "noise_loop.wav"
        effect = Effect(config["env"]["effect"])
        output_source = DAC

    if args.live_mode:
        input_source = LIVE

    analyser = Analyser(config["env"]["feature_extractors"], osc_route=osc_route)
    musician = Musician(
        analyser,
        effect=effect,
        input_source=input_source,
        output_source=output_source,
        duration=1000,
        is_target=args.is_target,
    )
    musician.start()


if __name__ == "__main__":
    main()
