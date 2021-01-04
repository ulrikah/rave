import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import datetime


def now():
    """
    Util for timestamping filenames and logs

    filename = f"bounce_{now()}.wav"
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def plot_melspectrogram(S: np.ndarray, sr=22050):
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


def plot_mfcc(mfccs: np.ndarray, sr=22050):
    """
    Plots the mel cepstral coefficients
    """
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='MFCC')
    plt.show()


def plot_wav(wav_file: str, sr=22050):
    """
    Plots a static wave file with matplotlib
    """
    samples = librosa.load(wav_file, sr)[0]
    plt.plot(samples)
    plt.show()


def play_wav(wav_file: str):
    """
    Plays back a static wave file by calling sox in a subprocess
    """
    subprocess.run(["play", wav_file])
