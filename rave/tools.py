import librosa
import librosa.display
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import subprocess
import datetime

from rave.constants import KSMPS, SAMPLE_RATE

# https://stackoverflow.com/questions/37470734/matplotlib-giving-error-overflowerror-in-draw-path-exceeded-cell-block-limit
matplotlib.rcParams["agg.path.chunksize"] = 10000


def scale(v, a, b, c, d):
    """
    Scales a value v in range [a, b] to range [c, d]
    """
    return (v - a) * (d - c) / (b - a) + c


def sec_per_k(ksmps=KSMPS, sr=SAMPLE_RATE):
    """How many seconds are there in 1 k with sample rate sr?"""
    return ksmps / sr


def k_per_sec(ksmps=KSMPS, sr=SAMPLE_RATE):
    """How many k are there in 1 second of sample rate sr?"""
    return sr / ksmps


def get_duration(wav_file):
    """
    Returns the duration of a WAV file in seconds

    NB! Requires sox to be installed
    """
    dur_bytes = subprocess.check_output(["soxi", "-D", wav_file])
    return float(dur_bytes.decode("utf-8").strip())


def timestamp(millis=True):
    """
    Util for timestamping filenames and logs

    filename = f"bounce_{timestamp()}.wav"
    """
    fmt = "%Y-%m-%d_%H-%M-%S"
    if millis:
        fmt += "_%f"
    return datetime.datetime.now().strftime(fmt)


def plot_melspectrogram(S: np.ndarray, sr=44100):
    """
    Plots the mel spectrogram
    """
    fig, ax = plt.subplots()
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr, ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title="Mel-frequency spectrogram")
    plt.show()


def plot_mfcc(mfccs: np.ndarray, sr=44100):
    """
    Plots the mel cepstral coefficients
    """
    fig, ax = plt.subplots()
    img = librosa.display.specshow(mfccs, x_axis="time", ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title="MFCC")
    plt.show()


def plot_spectrogram(y: np.ndarray, sr=44100):
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, ax=ax, x_axis="time", y_axis="log")
    fig.colorbar(img, ax=ax)
    plt.show()


def plot_1d(y: np.ndarray, title="RMS"):
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
    plt.figure(figsize=(12, 10))
    for wav_file in wav_files:
        samples = librosa.load(wav_file, sr)[0]
        plt.plot(samples, linewidth=0.5, markersize=12)
    plt.ylim(-1, 1)
    plt.legend(wav_files)
    if save_to is not None:
        plt.savefig(save_to)
        print(save_to)
    else:
        plt.show()

    plt.clf()
    plt.cla()
    plt.close()


def plot_wavs(wav_files: [str], save_to=None, sr=44100):
    fig, axs = plt.subplots(len(wav_files), sharex=True, sharey=True)

    plt.ylim(-1, 1)

    for i, wav_file in enumerate(wav_files):
        samples = librosa.load(wav_file, sr)[0]

        # Line labels
        n_samples = len(samples)
        n_xticks = 10
        xtick_values = np.linspace(0, n_samples, num=n_xticks)
        xtick_labels = [str(round(tick, 2)) for tick in xtick_values / SAMPLE_RATE]

        axs[i].plot(samples, linewidth=0.5, markersize=12)

        plt.yticks([])
        plt.xticks(xtick_values, xtick_labels)
        plt.xlabel("seconds")

    if save_to is not None:
        plt.savefig(save_to)
        print(save_to)
    else:
        plt.show()

    plt.clf()
    plt.cla()
    plt.close()


def play_wav(wav_file: str):
    """
    Plays back a static wave file by calling sox in a subprocess
    """
    subprocess.run(["play", wav_file])


def beep():
    return subprocess.run(["play", "-q", "-n", "synth", "1", "sin", "220"])


if __name__ == "__main__":

    paths = [
        "/Users/ulrikah/fag/thesis/results/live_inf/live_inference.wav",
        "/Users/ulrikah/fag/thesis/results/live_inf/offline_inference.wav",
    ]

    plot_wavs(paths)