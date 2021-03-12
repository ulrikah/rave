import os

"""
NOTE:
For some reason, ray fails when paths are relative.
The current fix is to append the project ROOT path to all other dir paths,
but there has to be something better
"""
PROJECT_ROOT = os.getcwd()

AUDIO_INPUT_DIR = os.path.join(PROJECT_ROOT, "rave/input_audio")
AUDIO_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "rave/bounces")
CSD_DIR = os.path.join(PROJECT_ROOT, "rave/csd")
EFFECT_TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "rave/effects")
ANALYSER_DIR = os.path.join(PROJECT_ROOT, "rave/feature_extractors")
RAY_RESULTS_DIR = os.path.join(PROJECT_ROOT, "rave/ray_results")

# sound
LIVE = "adc"
NO_SOUND = "--nosound"
DAC = "dac"
ADC = "adc"
SAMPLE_RATE = 44100
KSMPS = 64
WAVE_FILE_FLAG = "-W"  # write audio output as WAVE file instead of AIFF
CSD_JINJA_SUFFIX = ".csd.jinja2"
CSD_JINJA_GLOBALS_SUFFIX = f".globals{CSD_JINJA_SUFFIX}"
DEBUG_SUFFIX = "_debug"  # channels, files and misc that are used for debugging

# effects
EFFECT_BASE = f"base{CSD_JINJA_SUFFIX}"

# analyser
ANALYSER_BASE = f"base_analyser{CSD_JINJA_SUFFIX}"

# osc
OSC_ADDRESS = "127.0.0.1"
OSC_FEATURE_PORT = 4321
OSC_MAPPING_PORT = 1234
OSC_MAPPING_ROUTE = "/rave/mapping"

if __name__ == "__main__":
    assert os.path.isdir(AUDIO_INPUT_DIR)
    assert os.path.isdir(AUDIO_OUTPUT_DIR)
    assert os.path.isdir(EFFECT_TEMPLATE_DIR)
    assert os.path.isdir(ANALYSER_DIR)
    assert os.path.isdir(RAY_RESULTS_DIR)
