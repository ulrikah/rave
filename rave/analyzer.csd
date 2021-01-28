;    Copyright Oeyvind Brandtsegg 
;
;    This file contains a lot of code from the Feature-Extract-Modulator package (https://github.com/Oeyvind/featexmod).
;    It is licensed under thhe GNU General Public License.
;

<CsoundSynthesizer>
<CsOptions>
-i input_audio/amen.wav
; -i adc
-n
</CsOptions>
<CsInstruments>
sr = 44100
ksmps = 64
nchnls = 1
0dbfs = 1

; ***************
; globals

gi1         ftgen   1, 0, 32, -2, 0  ; analysis signal display

giSine	    ftgen	0, 0, 65536, 10, 1			; sine wave
gifftsize 	= 1024
            chnset gifftsize, "fftsize"
giFftTabSize	= (gifftsize / 2)+1
gifna     	ftgen   1 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs analysis
gifnf     	ftgen   2 ,0 ,giFftTabSize, 7, 0, giFftTabSize, 0   	; for pvs analysis

giSinEnv    ftgen   0, 0, 8192, 19, 1, 0.5, 270, 0.5        ; sinoid transient envelope shape for autocorr

; ***************
; OSC

#define IP_ADDRESS	# "127.0.0.1" #
#define PORT 		# 4321 #

instr 1

    #include "analyze_chn_init.inc"

    kwhen           init 0
    ki              init 0
    ki              += 1
    km              trigger ki%2, 1, 1
    
    ; ***************
    ; raw audio from input
    a1 in
    
    ; output
    outs a1, a1

    ; ***************
    ; pre-emphasis EQ for transient detection,
    ; allowing better sensitivity to utterances starting with a sibliant.
        kpreEqHiShelfFq	    chnget "preEqHiShelfFq"
        kpreEqHiShelfGain	chnget "preEqHiShelfGain"
        
        a1preEq             pareq a1, kpreEqHiShelfFq, ampdb(kpreEqHiShelfGain), 0.7,  2

    ; ***************
    ; amplitude tracking
        krms_preEq	    rms a1preEq			    	; simple level measure (with transient pre emphasis)
        krms_preEq      = krms_preEq*2
        krms		    rms a1			    		; simple level measure 
        krms            = krms*2
        krms_dB         = dbfsamp(krms)
                    
        kAttack		    = 0.001				        ; envelope follower attack
        kRelease        = 0.5;chnget "amp_transientDecTime"           ; envelope follower release
        a_env		    follow2	a1preEq, kAttack, kRelease	; envelope follower
        k_env		    downsamp a_env	

        knoiseFloor_dB	chnget "inputNoisefloor"
        kgate		    = (krms_dB < knoiseFloor_dB ? 0 : 1)	;  gate when below noise floor (for sampholding centroid and pitch etc)
        isecond_dB      = 9
        kgate2		    = (krms_dB < knoiseFloor_dB+isecond_dB ? 0 : 1)	;  gate when close to noise floor (for sampholding centroid and pitch etc)
        klowscaler      = limit(dbfsamp(krms)-knoiseFloor_dB, 0, isecond_dB)/isecond_dB      ; scaler to fade out different things towards the noise floor
        krms_dB_n       = (krms_dB/abs(knoiseFloor_dB))+1

        icrestrate      = 10
        kcrestmetro     metro icrestrate
        krms_max        init 0
        krms_max        max krms_max, krms
        krms_max        = (kcrestmetro > 0 ? 0 : krms_max)
        kcrestindex     init 0
        kcrestindex     = (kcrestindex+kcrestmetro)%2
        kcrestArr[]     init 2
        kcrestArr[kcrestindex] = krms_max
        kcrest_max      maxarray kcrestArr ; get max out of N last values
        kenv_crest0     divz kcrest_max, krms, 1
        kcrestrise      = (kgate2 > 0 ? 1 : 1) ; STATIC .. was slow response when signal is low
        kcrestfall      = (kgate2 > 0 ? 3 : 3) ; STATIC .. was slow response when signal is low
        kcrestA         = 0.001^(1/(kcrestrise*kr))
        kcrestB         = 0.001^(1/(kcrestfall*kr))
        kenv_crest      init 0
        kenv_crest      = (kenv_crest0>kenv_crest?(kenv_crest0+(kcrestA*(kenv_crest-kenv_crest0))):(kenv_crest0+(kcrestB*(kenv_crest-kenv_crest0))))
        kenv_crest1     = (dbamp(kenv_crest))/50

    ; ***************
    ; pitch tracking
    
        kpitch_low      chnget "pitch_low"
        kpitch_high     chnget "pitch_high"
        kpitch_low      init 100
        kpitch_high     init 1000          
    
        kd 		        = 0.1
        kloopf		    = 20
        kloopq		    = 0.3
        acps, alockp	plltrack a1, kd, kloopf, kloopq, kpitch_low, kpitch_high, ampdbfs(knoiseFloor_dB-8)
        kcps		    downsamp acps

        kmedianSize	    chnget "pitchFilterSize"
        kcps	        mediank	kcps, kmedianSize, 256
        kcps            limit kcps, kpitch_low, kpitch_high
        kcps            tonek kcps, 50

        kcps		samphold kcps, kgate2
        ; ksemitone       = limit:k((log2(kcps/440)*12)+69, 0, 127)

    ; ***************
    ; spectral analysis

        iwtype 			= 1
        fsin 			pvsanal	a1, gifftsize, gifftsize/2, gifftsize, iwtype
        kflag   		pvsftw	fsin,gifna,gifnf          	; export  amps  and freqs to table,
            
        kupdateRate		= 200
        kmetro			metro kupdateRate
        kdoflag			init 0
        kdoflag			= (kdoflag + kmetro);*kgate

        ; copy pvs data from table to array
        ; analyze spectral features
        kArrA[]  		init    giFftTabSize
        kArrAprev[]  		init    giFftTabSize
        kArrAnorm[]  		init    giFftTabSize
        kArrF[]  		init    giFftTabSize
        kArrCorr[]  		init    giFftTabSize
        kflatness		init 0

    if (kdoflag > 0) && (kflag > 0) then

        kArrAprev[]		= kArrA
                        copyf2array kArrA, gifna
                        copyf2array kArrF, gifnf	
        ksumAmp			sumarray kArrA
        kmaxAmp			maxarray kArrA
        ksumAmp         = (ksumAmp == 0 ? 1 : ksumAmp)
        kArrAnorm       = kArrA/ksumAmp
        kcentroid       pvscent fsin
        kArrCorr		= kArrA*kArrAprev
        kspread		    = sumarray(((kArrF+(kcentroid*-1))^2)*kArrAnorm)^0.5
        kskewness	    divz sumarray(((kArrF+(kcentroid*-1))^3)*kArrAnorm), kspread^3, 1
        kurtosis	    divz sumarray(((kArrF+(kcentroid*-1))^4)*kArrAnorm), kspread^4, 1	
        kcrest			divz kmaxAmp, ksumAmp/giFftTabSize, 1  
        kArrAlog[]      = kArrA
        kArrAlog[0]     = 1
        kArrAlog[1]     = 0
        klogmin         minarray kArrAlog
        while klogmin == 0 do
            klogmin,klogndx minarray kArrAlog
            kArrAlog[klogndx] = 1
        od
        kflatness		divz exp(sumarray(log(kArrAlog))/giFftTabSize),  (ksumAmp/giFftTabSize), 0
        kflux           = 1-(divz(sumarray(kArrCorr),(sqrt(sumarray(kArrA^2))*sqrt(sumarray(kArrAprev^2))),0))
        kdoflag 		= 0


    ; ** filter hack to keep spectral signals at the value analyzed while sound level above noise floor
        kcentroid		samphold kcentroid, kgate
        kcentroid2		samphold kcentroid, kgate2
        kcentroid       = (kgate2 > 0 ? kcentroid : kcentroid2-((kcentroid-kcentroid2)*((1-klowscaler)*0.25)))
        kspread		    samphold kspread, kgate
        kspread2		samphold kspread, kgate2
        kspread         = (kgate2 > 0 ? kspread : kspread2-((kspread-kspread2)*((1-klowscaler)*0.25)))
        kskewness		samphold kskewness, kgate
        kskewness2		samphold kskewness, kgate2
        kskewness       = (kgate2 > 0 ? kskewness : kskewness2-((kskewness-kskewness2)*((1-klowscaler)*0.25)))
        kurtosis		samphold kurtosis, kgate
        kurtosis2		samphold kurtosis, kgate2
        kurtosis        = (kgate2 > 0 ? kurtosis : kurtosis2-((kurtosis-kurtosis2)*((1-klowscaler)*0.25)))
        kflatness		samphold kflatness, kgate
        kflatness2		samphold kflatness, kgate2
        kflatness       = (kgate2 > 0 ? kflatness : kflatness2-((kflatness-kflatness2)*((1-klowscaler)*0.25)))
        kcrest		    samphold kcrest, kgate
        kcrest2		    samphold kcrest, kgate2
        kcrest          = (kgate2 > 0 ? kcrest : kcrest2-((kcrest-kcrest2)*((1-klowscaler)*0.25)))
        kflux		    samphold kflux, kgate
        kflux2		    samphold kflux, kgate2
        kflux           = (kgate2 > 0 ? kflux : kflux2-((kflux-kflux2)*((1-klowscaler)*0.25)))

    endif

    ; post filtering of spectral tracks
        kcentroidf              tonek kcentroid, 20
        kfluxf                  tonek kflux, 20

    ; ***************
    ; normalization
        kpitch_n        = limit(divz(kcps-kpitch_low, kpitch_high-kpitch_low, 1), 0, 1)    ; normalized and offset
        kcentroid_n     = kcentroidf / (sr*0.15)
        kflux_n         = kfluxf * 4
        imfccscale      = 1/200


    ; ***************
    ; limiter to (0, 1) range
        krms            limit krms, 0, 1
        kflux_l         limit kflux_n, 0, 1
      
    OSCsend kwhen, $IP_ADDRESS, $PORT, "/rave/features", "ffff", krms, kpitch_n, kcentroid_n, kflux_l
    ; String sprintfk "rms : %f | pitch : %f | cent : %f | flux : %f", krms, kpitch_n, kcentroid_n, kflux_l
    ; puts String, kwhen
    kwhen += km
endin

</CsInstruments>
<CsScore>
i 1 0 5.3
</CsScore>
</CsoundSynthesizer>