import ctcsound
import sys


class Player:
    def __init__(self):
        self.cs = ctcsound.Csound()

    def play_csd_string(self, csd):
        # play the csd from text
        result = self.cs.compileCsdText(csd)
        result = self.cs.start()
        while True:
            result = self.cs.performKsmps()
            if result != 0:
                break
        self.cleanup()

    def cleanup(self, exit=False):
        result = self.cs.cleanup()
        self.cs.reset()
        if exit:
            del self.cs
            sys.exit(result)
