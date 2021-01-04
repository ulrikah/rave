import ctcsound
import timeit
import os

FILENAME = "timeit.wav"
NUMBER = 1000

orc = """
sr=44100
ksmps=32
nchnls=1
0dbfs=1

instr 1
aout poscil p3, p4
aout moogladder aout, 2000, 0.25
out aout
endin
"""

sco = """
i1 0 9 256
i1 3 6 266
i1 2 3 276
"""

t = timeit.timeit("""
cs = ctcsound.Csound()
cs.setOption(f"-o{FILENAME}")
cs.setOption("-W")

cs.compileOrc(orc)
cs.readScore(sco)

cs.start()
while cs.performBuffer() == 0:
    pass
cs.cleanup()
cs.reset()
os.remove(FILENAME)
""", globals=globals(), number=NUMBER)


print("")
print("Timer resulted in")
print("\t", "total time:", round(t, 2), "s")
print("\t", "avg time per loop:", round(t / NUMBER, 2), "s")
