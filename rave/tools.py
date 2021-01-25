import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import datetime


def get_duration(wav_file):
    """
    Returns the length of a WAV file in seconds

    NB! Requires sox to be installed
    TODO: find better way of determining length
    """
    dur_bytes = subprocess.check_output(["soxi", "-D", wav_file])
    return float(dur_bytes.decode("utf-8").strip())


def timestamp():
    """
    Util for timestamping filenames and logs

    filename = f"bounce_{timestamp()}.wav"
    """
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


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


def plot_1d(y: np.ndarray, title='RMS'):
    """
    Plots 1D numpy arrays along
    """
    fig, ax = plt.subplots()
    plt.plot(y)
    ax.set(title=title)
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


if __name__ == "__main__":
    dur = get_duration("rave/input_audio/amen.wav")
    assert type(dur) == float
    assert dur == 5.564717, "File length doesn't match soxi's output"
