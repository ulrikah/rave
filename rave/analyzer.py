import librosa
import numpy as np
from .utils import plot_melspectrogram, plot_mfcc


class Analyzer:
    def __init__(self):
        self.features = []

    def analyze_wav(self, wav_file):
        """
        TO DO: analyze audio in buffers
        """
        y, sr = librosa.load(wav_file, sr=22050)
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        return [S, mfccs]


if __name__ == "__main__":
    source = "input_audio/noise.wav"
    target = "input_audio/amen.wav"
    analyzer = Analyzer()
    noise_mfccs = analyzer.analyze_wav(source)
    amen_mfccs = analyzer.analyze_wav(target)
    import pdb
    pdb.set_trace()
