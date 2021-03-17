import numpy as np


from rave.sound import Sound
from rave.analyser import Analyser
from rave.constants import DEVIATION_LIMIT


class Standardizer:
    DEVIATION_LIMIT = 4.0

    def __init__(self, sounds: [Sound], analyser: Analyser):
        self.sounds = sounds
        self.analyser = analyser
        self.stats = self.get_stats()

    def get_stats(self):
        """
        Computes mean and standard deviation for all the sounds
        """
        features = []
        for sound in self.sounds:
            sound.prepare_to_render(analyser=self.analyser)
            done = False
            while not done:
                done = sound.render()
                frame_features = sound.player.get_channels(
                    self.analyser.analysis_features
                )
                if (max(frame_features) > 1.0) or (min(frame_features) < 0.0):
                    # NOTE: hacky way of filtering out outliars since Csound params are supposed to be limited (?)
                    # TODO: log how often this happens and try to debug it
                    continue
                else:
                    features.append(frame_features)
        features = np.array(features, dtype=float)
        stats = {}
        for i, feature_name in enumerate(self.analyser.analysis_features):
            stats[feature_name] = {
                "mean": np.mean(features[:, i]),
                "std": np.std(features[:, i]),
                "min": np.min(features[:, i]),
                "max": np.max(features[:, i]),
            }
        return stats

    def get_standardized_value(self, feature: str, value: float):
        if not self.stats:
            raise ValueError("Statistics have not been calculated yet")
        if not feature in self.stats.keys():
            raise ValueError("Statistics have not been calculated for this feature")

        if self.stats[feature]["std"] == 0.0:
            standardized_value = value - self.stats[feature]["mean"]
        else:
            standardized_value = (value - self.stats[feature]["mean"]) / self.stats[
                feature
            ]["std"]
            standardized_value = max(
                min(standardized_value, self.DEVIATION_LIMIT), -self.DEVIATION_LIMIT
            )
        return standardized_value


if __name__ == "__main__":
    sounds = [Sound("noise.wav"), Sound("amen_trim.wav")]
    a = Analyser(["rms", "mfcc"])
    s = Standardizer(sounds, a)

    new_sound = Sound("drums_7s.wav")
    features = np.empty(shape=(len(a.analysis_features),))
    new_sound.prepare_to_render(analyser=a)
    done = False
    while not done:
        done = new_sound.render()
        frame_features = np.array(new_sound.player.get_channels(a.analysis_features))
        if (frame_features > 1.0).any() or (frame_features < 0.0).any():
            # NOTE: hacky way of filtering out outliars since Csound params are supposed to be limited (?)
            # TODO: log how often this happens and try to debug it
            continue
        else:
            features = np.vstack((features, frame_features))
    # remove initial row
    features = features[1:, :]

    for row in features[
        100:1000,
    ]:
        for i, feature in enumerate(a.analysis_features):
            value = row[i]
            print(feature, "\t", s.get_standardized_value(feature, value))
        print("\n")
