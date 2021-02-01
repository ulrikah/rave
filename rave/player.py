import ctcsound

import sys
import logging


class Player:
    def __init__(self):
        self.k = 0
        self.cs = ctcsound.Csound()

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
        self.cs.compileCsdText(csd)
        self.cs.start()

    def set_channels(self, mappings):
        for name, value in mappings.items():
            self.cs.setControlChannel(name, value)

    def get_channels(self, channels):
        return [self.cs.controlChannel(channel)[0] for channel in channels]

    def render_one_frame(self):
        """Performs one k-rate update of the compiled csd"""
        result = self.cs.performKsmps()
        if result != 0:
            logging.critical(f"URGENT. Result: {result} | k: {self.k}")
            self.cleanup()
            sys.exit(result)
        self.k += 1
        """
        TODO: return a code from render() that env can use to determine whether or not to apply a step (say done?)
        if self.k > end:
            STOP
            self.cleanup ?
            self.k = 0
        """

    def cleanup(self, exit=False):
        exit_code = self.cs.cleanup()
        self.cs.reset()
        self.k = 0
        if exit:
            del self.cs
            sys.exit(exit_code)


if __name__ == "__main__":

    from effect import Effect
    from tools import timestamp
    import os
    from template_handler import TemplateHandler

    EFFECTS_TEMPLATE_DIR = "rave/effects"
    effect = Effect("bandpass")
    mapping = effect.random_mapping()
    effect_csd = TemplateHandler(
        f"{effect.name}.csd.jinja2", template_dir=EFFECTS_TEMPLATE_DIR).compile(mapping)

    base_csd = TemplateHandler(
        "base_no_score.csd.jinja2", template_dir=EFFECTS_TEMPLATE_DIR)

    csd = base_csd.compile(
        input="rave/input_audio/amen_trim.wav",
        output=f"rave/bounces/amen_trim_{effect.name}_{timestamp()}.wav",
        sample_rate=44100,
        ksmps=64,
        flags="-W",
        effect=effect_csd,
        analyzer="",
    )

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
        print(player.k)
    player.cleanup()
