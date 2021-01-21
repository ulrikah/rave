<CsoundSynthesizer>
<CsOptions>
-o dac
</CsOptions>
<CsInstruments>
sr = 44100
ksmps = 32
nchnls = 2
0dbfs = 1

#define IP_ADDRESS	# "127.0.0.1" #
#define PORT 		# 4321 #

turnon 1000
	
instr 1000
	kValue1Received init 0.0
	kValue2Received init 0.0
	kValue3Received init 0.0
	Stext sprintf "%i", $PORT
	ihandle OSCinit $PORT
	kAction OSClisten ihandle, "/rave/features", "fff",
                 kValue1Received, kValue2Received, kValue3Received
		if (kAction == 1) then	
			printk2 kValue2Received
			printk2 kValue1Received
			
		endif
	aSine poscil3 kValue1Received, kValue2Received, -1
	; a bit reverbration
	aInVerb = aSine*kValue3Received
	aWetL, aWetR freeverb aInVerb, aInVerb, 0.4, 0.8
outs aWetL+aSine, aWetR+aSine
endin

</CsInstruments>
f 1 0 1024 10 1
f 2 0 8 -2      0 2 4 7 9 11 0 2
e 3600
</CsoundSynthesizer>