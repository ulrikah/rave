import numpy as np
from collections import deque

from rave.sound import Sound
from rave.analyser import Analyser
from rave.constants import DEVIATION_LIMIT


class Standardizer:
    def __init__(
        self, sounds: [Sound], analyser: Analyser, reward_norm_batch_size=None
    ):
        self.sounds = sounds
        self.analyser = analyser
        if reward_norm_batch_size is not None:
            if not reward_norm_batch_size > 0:
                raise ValueError(
                    "The batch used for reward normalisation must be greater than zero"
                )
            self.reward_norm_batch_size = reward_norm_batch_size
            self.reward_norm_batch = deque(maxlen=reward_norm_batch_size)

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
        if len(self.reward_norm_batch) == self.reward_norm_batch_size:
            mean = np.mean(self.reward_norm_batch)
            std = np.std(self.reward_norm_batch)
            if std == 0.0:
                standardized_reward = reward - mean
            else:
                standardized_reward = (reward - mean) / std
        else:
            # We don't have enough reward samples to standardize
            standardized_reward = reward

        # append the new reward after mean and std calculation
        self.reward_norm_batch.append(reward)
        return standardized_reward


if __name__ == "__main__":
    sounds = [Sound("noise_5s.wav"), Sound("drums_5s.wav")]
    a = Analyser(["rms", "pitch", "spectral"])
    s = Standardizer(sounds, a)
    print(s.stats)

    # new_sound = Sound("drums_7s.wav")
    # features = np.empty(shape=(len(a.analysis_features),))
    # new_sound.prepare_to_render(analyser=a)
    # done = False
    # while not done:
    #     done = new_sound.render()
    #     frame_features = np.array(new_sound.player.get_channels(a.analysis_features))
    #     if (frame_features > 1.0).any() or (frame_features < 0.0).any():
    #         # NOTE: hacky way of filtering out outliars since Csound params are supposed to be limited (?)
    #         # TODO: log how often this happens and try to debug it
    #         continue
    #     else:
    #         features = np.vstack((features, frame_features))
    # # remove initial row
    # features = features[1:, :]

    # for row in features[
    #     100:1000,
    # ]:
    #     for i, feature in enumerate(a.analysis_features):
    #         value = row[i]
    #         print(feature, "\t", s.get_standardized_value(feature, value))
    #     print("\n")
