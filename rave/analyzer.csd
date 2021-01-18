<CsoundSynthesizer>
<CsOptions>
-n -i test_audio/amen.wav
</CsOptions>
<CsInstruments>
sr = 44100
ksmps = 64
nchnls = 1
0dbfs = 1

#define IP_ADDRESS	# "127.0.0.1" #
#define PORT 		# 4321 #

turnon 1000

instr 1000

    #include "analyze_chn_init.inc"

    kwhen           init 0
    ki              init 0
    ki              += 1
    km              trigger ki%2, 1, 1
    
    ; ***************
    ; raw audio from input
    a1 in 

    ; ***************
    ; pre-emphasis EQ for transient detection,
    ; allowing better sensitivity to utterances starting with a sibliant.
        kpreEqHiShelfFq	    chnget "preEqHiShelfFq"
        kpreEqHiShelfGain	chnget "preEqHiShelfGain"
        
        a1preEq             pareq a1, kpreEqHiShelfFq, ampdb(kpreEqHiShelfGain), 0.7,  2

    ; ***************
    ; amplitude tracking
        krms_preEq	        rms a1preEq     ; simple level measure (with transient pre emphasis)
        krms_preEq          = krms_preEq*2
        krms		        rms a1          ; simple level measure 
        krms                = krms*2
        krms_dB             = dbfsamp(krms)
      
    OSCsend kwhen, $IP_ADDRESS, $PORT, "/rave/features", "fff", krms, krms, krms_dB
    String sprintfk "rms : %f | db : %f", krms, krms_dB
    puts String, kwhen
    kwhen += km
endin

</CsInstruments>
</CsoundSynthesizer>