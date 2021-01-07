<CsoundSynthesizer>
<CsOptions>
-odac
</CsOptions>
<CsInstruments>
sr = 44100
ksmps = 32
nchnls = 1

gisig	 ftgen	0,0, 257, 9, .5,1,270	; define a simple sigmoid
gisine   ftgen 1, 0, 16384, 10, 1	;sine wave
gisquare ftgen 2, 0, 16384, 10, 1, 0 , .33, 0, .2 , 0, .14, 0 , .11, 0, .09 ;odd harmonics
gisaw    ftgen 3, 0, 16384, 10, 0, .2, 0, .4, 0, .6, 0, .8, 0, 1, 0, .8, 0, .6, 0, .4, 0,.2 ;even harmonics

instr	1 ; play audio from disk

kSpeed  init     1           ; playback speed
iSkip   init     0           ; inskip into file (in seconds)
iLoop   init     0           ; looping switch (0=off 1=on)

kdist line 0, 3, 1; ramp from 0 to 1 in 3 secs

; read audio from disk using diskin2 opcode
asnd    diskin2  "test_audio/amen.wav", kSpeed, iSkip, iLoop
aout	distort	asnd, kdist, gisig	; gradually increase the distortion
out aout

endin

</CsInstruments>
<CsScore>
i 1 0 6
e
</CsScore>
</CsoundSynthesizer>