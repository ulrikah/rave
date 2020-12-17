import pdb
import time
import ctcsound
from random import randint, random
import sys


class RandomLine(object):
    def __init__(self, base, range):
        self.curVal = 0.0
        self.reset()
        self.base = base
        self.range = range

    def reset(self):
        self.dur = randint(256, 512)
        self.end = random()
        self.slope = (self.end - self.curVal) / self.dur

    def getValue(self):
        self.dur -= 1
        if(self.dur < 0):
            self.reset()
        retVal = self.curVal
        self.curVal += self.slope
        return self.base + (self.range * retVal)


CSD_FILE = "hello.csd"
cs = ctcsound.Csound()
status = cs.compile_("csound", "-m7", CSD_FILE)

if not status == ctcsound.CSOUND_SUCCESS:
    print("Compilation error")
    sys.exit()

ampChannel, _ = cs.channelPtr(
    "amp", ctcsound.CSOUND_CONTROL_CHANNEL | ctcsound.CSOUND_INPUT_CHANNEL)
freqChannel, _ = cs.channelPtr(
    "freq", ctcsound.CSOUND_CONTROL_CHANNEL | ctcsound.CSOUND_INPUT_CHANNEL)

amp = RandomLine(.4, .2)
freq = RandomLine(400, 80)

ampChannel[0] = amp.getValue()
freqChannel[0] = freq.getValue()

duration = 10
t_start = time.time()
cs.scoreEvent("i", [1, 0, duration])
while (cs.performKsmps() == 0):
    ampChannel[0] = amp.getValue()
    freqChannel[0] = freq.getValue()
    if int(time.time() - t_start) > duration:
        break
cs.reset()
