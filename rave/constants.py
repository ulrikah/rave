import os

# sound
LIVE = "adc"
NO_SOUND = "--nosound"
DAC = "dac"
AUDIO_INPUT_DIR = "rave/input_audio"
AUDIO_OUTPUT_DIR = "rave/bounces"
CSD_DIR = "rave/csd"
SAMPLE_RATE = 44100
KSMPS = 64
WAVE_FILE_FLAG = "-W"  # write audio output as WAVE file instead of AIFF
CSD_JINJA_SUFFIX = ".csd.jinja2"
CSD_JINJA_GLOBALS_SUFFIX = f".globals{CSD_JINJA_SUFFIX}"

# effects
EFFECT_TEMPLATE_DIR = "rave/effects"
EFFECT_BASE = f"base{CSD_JINJA_SUFFIX}"

# analyser
ANALYSER_DIR = "rave/feature_extractors"
ANALYSER_BASE = f"base_analyser{CSD_JINJA_SUFFIX}"


# ray
RAY_RESULTS_DIR = "rave/ray_results"


assert os.path.isdir(AUDIO_INPUT_DIR)
assert os.path.isdir(AUDIO_OUTPUT_DIR)
assert os.path.isdir(EFFECT_TEMPLATE_DIR)
assert os.path.isdir(ANALYSER_DIR)
assert os.path.isdir(RAY_RESULTS_DIR)