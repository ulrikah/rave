import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

import subprocess
import datetime
import wave
import glob
import sys


def k_to_sec(ksmps=64, sr=44100):
    """How many seconds are there in 1 k with sample rate sr?"""
    return (ksmps / sr)


def k_per_sec(ksmps=64, sr=44100):
    """How many ksmps are there in 1 second of sample rate sr?"""
    return (sr / ksmps)


def get_duration(wav_file):
    """
    Returns the duration of a WAV file in seconds

    NB! Requires sox to be installed
    TODO: find better way of determining duration
    """
    dur_bytes = subprocess.check_output(["soxi", "-D", wav_file])
    return float(dur_bytes.decode("utf-8").strip())


def timestamp():
    """
    Util for timestamping filenames and logs

    filename = f"bounce_{timestamp()}.wav"
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def plot_melspectrogram(S: np.ndarray, sr=44100):
    """
    Plots the mel spectrogram
    """
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(
        S_dB, x_axis='time', y_axis='mel', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')
    plt.show()


def plot_mfcc(mfccs: np.ndarray, sr=44100):
    """
    Plots the mel cepstral coefficients
    """
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='MFCC')
    plt.show()


def plot_1d(y: np.ndarray, title='RMS'):
    """
    Plots 1D numpy arrays along
    """
    fig, ax = plt.subplots()
    plt.plot(y)
    ax.set(title=title)
    plt.show()


def plot_wav(wav_file: str, save_to=None, sr=44100):
    """
    Plots a static wave file with matplotlib
    """
    samples = librosa.load(wav_file, sr)[0]
    plt.plot(samples)
    if save_to is not None:
        plt.savefig(save_to)
    else:
        plt.show()


def plot_wavs_on_top_of_eachother(wav_files: [str], save_to=None, sr=44100):
    for wav_file in wav_files:
        samples = librosa.load(wav_file, sr)[0]
        plt.plot(samples)
    plt.legend(wav_files)
    if save_to is not None:
        plt.savefig(save_to)
    else:
        plt.show()


def play_wav(wav_file: str):
    """
    Plays back a static wave file by calling sox in a subprocess
    """
    subprocess.run(["play", wav_file])


if __name__ == "__main__":
    path = "rave/bounces/*render*.wav"
    wav_paths = glob.glob(path)
    plot_wavs_on_top_of_eachother(wav_paths, save_to="rave/plots/wavs.jpg")
