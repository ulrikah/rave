import ctcsound
import sys


class Player:
    def __init__(self):
        self.cs = ctcsound.Csound()

    def render_csd(self, csd: str, exit=False):
        """
        Render CSD string. Options for I/O type should be set in the CSD string
        """
        result = self.cs.compileCsdText(csd)
        result = self.cs.start()
        while True:
            result = self.cs.performKsmps()
            if result != 0:
                break
        self.cleanup(exit)

    def cleanup(self, exit=False):
        result = self.cs.cleanup()
        self.cs.reset()
        if exit:
            del self.cs
            sys.exit(result)
