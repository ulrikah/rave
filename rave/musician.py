from rave.analyser import Analyser
from rave.sound import Sound
from rave.constants import LIVE, DAC, NO_SOUND

import subprocess


class Musician:
    """
    This class represents a virtual musician as a continous stream of OSC messages simulating live input
    """

    def __init__(self, analyser: Analyser, input_source=None, duration=10):
        if input_source is None:
            input_source_index = self.choose_audio_device()
            input_source = f"adc{input_source_index}"
        self.analyser = analyser
        self.sound = Sound(input_source, NO_SOUND, duration=duration)

    @staticmethod
    def list_audio_devices():
        subprocess.call(["csound", "--devices=in", "-m", "128"])
        return

    def choose_audio_device(self):
        self.list_audio_devices()
        print("\n")
        device = input("🎤 Choose the index of the input device you want to use: ")
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
    a = Analyser(["rms"], osc_route="/rave/source/features", audio_to_analyse="aOut")
    m = Musician(input_device=LIVE, analyser=a)
    m.start()
    print("Performance done")


if __name__ == "__main__":
    main()
