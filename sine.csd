<CsoundSynthesizer>
<CsOptions>
-o sine220.wav -W
</CsOptions>
<CsInstruments>
sr = 44100
ksmps = 32
nchnls = 1
0dbfs = 1

instr 1
    aout poscil .8, p4
    out aout
endin

</CsInstruments>
<CsScore>
i1 0 3 220
</CsScore>
</CsoundSynthesizer>