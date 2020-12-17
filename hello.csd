<CsoundSynthesizer>
<CsOptions>
-odac
</CsOptions>
<CsInstruments>

sr=44100
ksmps=32
nchnls=2
0dbfs=1

instr 1 
kamp chnget "amp"
kfreq chnget "freq"
printk 0.5, kamp
printk 0.5, kfreq
aout poscil kamp, kfreq
aout moogladder aout, 2000, 0.25
outs aout, aout
endin

</CsInstruments>
</CsoundSynthesizer>
