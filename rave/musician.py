from rave.analyser import Analyser
from rave.sound import Sound
from rave.constants import LIVE, DAC, ADC, NO_SOUND
from rave.config import parse_config_file

import subprocess
import argparse


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config_file",
        action="store",
        required=True,
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
        self, analyser: Analyser, input_source=None, output_source=None, duration=10
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
        self.sound = Sound(input_source, output_source, duration=duration)

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
        csd_file = self.sound.prepare_to_render(analyser=self.analyser)
        self.sound.stream()
        return


def main():
    args = arguments()
    config = parse_config_file(args.config_file)

    if args.is_target:
        osc_route = "/rave/target/features"
        # NOTE: temporary hack to loop the target sound
        # input_source = config["env"]["target"]
        input_source = "amen_loop.wav"
    else:
        osc_route = "/rave/source/features"
        input_source = config["env"]["source"]
    if args.live_mode:
        input_source = LIVE

    analyser = Analyser(config["env"]["feature_extractors"], osc_route=osc_route)
    BLACKHOLE = "dac2"
    musician = Musician(
        analyser, input_source=input_source, output_source=BLACKHOLE, duration=100
    )
    musician.start()


if __name__ == "__main__":
    main()
