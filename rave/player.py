import ctcsound

import sys
import logging
import numpy as np


class Player:
    def __init__(self, debug=False):
        self.k = 0
        self.cs = ctcsound.Csound()
        self.csd = None
        self.debug = debug
        if not self.debug:
            self.cs.setOption("--nodisplays")

    def render_csd(self, csd: str, exit=False):
        """
        Renders a CSD string and optionally exits afterwards
        """
        if not self.debug:
            self._open_message_buffer()
        result = self.cs.compileCsdText(csd)
        result = self.cs.start()
        while self.cs.performKsmps() == 0:
            pass
        self.cleanup(exit)

    def _open_message_buffer(self):
        self.cs.createMessageBuffer(False)
        self._has_message_buffer = True

    def start_halting(self, csd: str):
        """Compiles the CSD and starts the engine, but wait for render function to actually render a k"""
        if not self.debug:
            self._open_message_buffer()
        self.csd = csd
        self.cs.compileCsdText(csd)
        self.cs.start()

    def set_channels(self, mappings):
        for name, value in mappings.items():
            self.cs.setControlChannel(name, value)

    def get_channels(self, channels):
        return np.array([self.cs.controlChannel(channel)[0] for channel in channels])

    def render_until_end(self):
        while self.cs.performKsmps() == 0:
            pass
        self.cleanup()

    def render_one_frame(self, loop=True):
        """Performs one k-rate update of the compiled csd"""
        result = self.cs.performKsmps()
        self.k += 1
        if result != 0:
            self.cleanup()
            # end of score
            if result == 2:
                if loop == True:
                    assert (
                        self.csd is not None
                    ), "Tried to restart without a reference to any CSD"
                    self.k = 0
                    self.start_halting(self.csd)
            else:
                logging.critical(f"URGENT. Result: {result} | k: {self.k}")
                sys.exit(result)
        return result

    def cleanup(self, exit=False):
        if not self.debug:
            if self._has_message_buffer:
                self.cs.destroyMessageBuffer()
                self._has_message_buffer = False
        exit_code = self.cs.cleanup()
        self.cs.reset()
        if exit:
            del self.cs
