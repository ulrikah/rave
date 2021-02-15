import ctcsound

import sys
import logging
import numpy as np


class Player:
    def __init__(self):
        self.k = 0
        self.cs = ctcsound.Csound()
        self.csd = None
        self.debug = False
        if not self.debug:
            self.cs.setOption("--nodisplays")

    def render_csd(self, csd: str, exit=False):
        """
        Renders a CSD string and optionally exits afterwards
        """
        result = self.cs.compileCsdText(csd)
        result = self.cs.start()
        while True:
            result = self.cs.performKsmps()
            if result != 0:
                break
        self.cleanup(exit)

    def start_halting(self, csd: str):
        """Compiles the CSD and starts the engine, but wait for render function to actually render a k"""
        if not self.debug:
            self.cs.createMessageBuffer(False)
            self._has_message_buffer = True
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
                    assert self.csd is not None, "Tried to restart without a reference to any CSD"
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


if __name__ == "__main__":

    test = """
    <CsoundSynthesizer>
    <CsOptions>
    ; Select audio/midi flags here according to platform
    ;-o test.wav -W    ;;;realtime audio out
    --nosound
    ;-iadc    ;;;uncomment -iadc if realtime audio input is needed too
    ; For Non-realtime ouput leave only the line below:
    ; -o oscils.wav -W ;;; for file output any platform
    </CsOptions>
    <CsInstruments>

    sr = 44100
    ksmps = 32
    nchnls = 2
    0dbfs  = 1

    gi1         ftgen   1, 0, 32, -2, 0  ; analysis signal display

    giSine	    ftgen	0, 0, 65536, 10, 1			; sine wave
    gifftsize 	= 1024
                chnset gifftsize, "fftsize"
    giFftTabSize	= (gifftsize / 2)+1
    gifna     	ftgen   1 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs analysis
    gifnf     	ftgen   2 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs analysis

    giSinEnv    ftgen   0, 0, 8192, 19, 1, 0.5, 270, 0.5        ; sinoid transient envelope shape for autocorr

    instr 1
        kCount    init      0; set kcount to 0 first
        kCount    =         kCount + 1; increase at each k-pass
        printk    0, kCount; print the value

        asig oscils .7, 220, 0
        outs asig, asig
    endin
    </CsInstruments>
    <CsScore>
    ; Play Instrument #1 for 5 seconds.
    i 1 0 5
    e
    </CsScore>
    </CsoundSynthesizer>
    """

    player = Player()
    player.start_halting(test)
    for i in range(3):
        player.render_one_frame()
        """Performs one k-rate update of the compiled csd"""
    player.cleanup()
