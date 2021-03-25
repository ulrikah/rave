import numpy as np
from itertools import combinations

from rave.sound import Sound
from rave.analyser import Analyser
from rave.constants import DEVIATION_LIMIT
from rave.metrics import AbstractMetric, EuclideanDistance


class Standardizer:
    def __init__(
        self, sounds: [Sound], analyser: Analyser, metric: AbstractMetric = None
    ):
        self.sounds = sounds
        self.analyser = analyser
        self.metric = metric
        # not supported yet due to some unexpected peaks in the analysis signal
        if "mfcc" in map(
            lambda feature: feature["name"], self.analyser.feature_extractors
        ):
            raise ValueError("mfcc is not supported")
        self.stats = self.get_stats()

    def get_stats(self):
        """
        Computes mean and standard deviation for all the sounds
        """

        # array where every element is the feature matrix of one sound
        sounds_features = []

        for sound in self.sounds:
            sound_features = []
            sound.prepare_to_render(analyser=self.analyser)
            done = False
            while not done:
                done = sound.render()
                frame_features = sound.player.get_channels(
                    self.analyser.analysis_features
                )
                sound_features.append(frame_features)
            sounds_features.append(np.array(sound_features))

        stats = {}
        # stats for reward calculations
        if self.metric is not None:
            rewards = []
            for sound_a, sound_b in combinations(sounds_features, 2):
                assert sound_a.shape == sound_b.shape
                rows = sound_a.shape[0]
                for i in range(rows):
                    rewards.append(self.metric.calculate_reward(sound_a[i], sound_b[i]))
            rewards = np.array(rewards)
            stats["reward"] = {
                "mean": np.mean(rewards),
                "std": np.std(rewards),
                "min": np.min(rewards),
                "max": np.max(rewards),
            }

        # matrix of all features concatenated together
        all_features = np.concatenate(
            [_sound_features for _sound_features in sounds_features], dtype=float
        )
        # stats for every feature
        for i, feature_name in enumerate(self.analyser.analysis_features):
            stats[feature_name] = {
                "mean": np.mean(all_features[:, i]),
                "std": np.std(all_features[:, i]),
                "min": np.min(all_features[:, i]),
                "max": np.max(all_features[:, i]),
            }

        return stats

    def get_standardized_value(self, feature: str, value: float):
        if not self.stats:
            raise ValueError("Statistics have not been calculated yet")
        if feature not in self.stats.keys():
            raise ValueError("Statistics have not been calculated for this feature")

        if self.stats[feature]["std"] == 0.0:
            standardized_value = value - self.stats[feature]["mean"]
        else:
            standardized_value = (value - self.stats[feature]["mean"]) / self.stats[
                feature
            ]["std"]
            standardized_value = max(
                min(standardized_value, DEVIATION_LIMIT), -DEVIATION_LIMIT
            )
        return standardized_value

    def get_standardized_reward(self, reward: float):
        if not self.stats or "reward" not in self.stats.keys():
            raise ValueError("Statistics have not been calculated for the reward")
        reward_stats = self.stats["reward"]
        if reward_stats["std"] == 0.0:
            standardized_reward = reward - reward_stats["mean"]
        else:
            standardized_reward = (reward - reward_stats["mean"]) / reward_stats["std"]
        return standardized_reward


if __name__ == "__main__":
    sounds = [Sound("noise_5s.wav"), Sound("drums_5s.wav")]
    a = Analyser(["rms", "pitch", "spectral"])
    m = EuclideanDistance()
    s = Standardizer(sounds, a, m)

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
