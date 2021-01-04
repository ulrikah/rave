import librosa
import matplotlib.pyplot as plt
import subprocess
import datetime


def now():
    """
    Util for timestamping filenames and logs

    filename = f"bounce_{now()}.wav"
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def plot_wav(wav_file, sr=44100):
    """
    Plots a static wave file with matplotlib
    """
    samples = librosa.load(wav_file, sr)[0]
    plt.plot(samples)
    plt.show()


def play_wav(wav_file):
    """
    Plays back a static wave file by calling sox in a subprocess
    """
    subprocess.run(["play", wav_file])
