import numpy as np

from rave.standardizer import Standardizer
from rave.sound import Sound
from rave.analyser import Analyser


def get_standardizer(
    sounds=[Sound("noise.wav")], analyser=Analyser(["rms"]), reward_norm_batch_size=100
):
    return Standardizer(sounds, analyser, reward_norm_batch_size=reward_norm_batch_size)


def test_rewards_are_standardized_when_batch_is_full():
    reward_norm_batch_size = 10
    standardizer = get_standardizer(reward_norm_batch_size=reward_norm_batch_size)
    assert standardizer.reward_norm_batch_size == 10
    assert standardizer.reward_norm_batch.maxlen == 10
    rewards = np.linspace(0, 1, num=reward_norm_batch_size + 1)
    for reward in rewards[:-1]:
        assert reward == standardizer.get_standardized_reward(reward)
    assert rewards[-1] < standardizer.get_standardized_reward(rewards[-1])


def test_successive_rewards_of_same_magnitude_have_different_standardized_values():
    # This should happen because the mean increases when the new reward is used to calculate mean and stddev
    reward_norm_batch_size = 10
    standardizer = get_standardizer(reward_norm_batch_size=reward_norm_batch_size)
    rewards = np.linspace(0, 1, num=reward_norm_batch_size)
    for reward in rewards:
        _ = standardizer.get_standardized_reward(reward)
    highest_reward = 1.0

    highest_reward_standardized_1 = standardizer.get_standardized_reward(rewards[-1])
    highest_reward_standardized_2 = standardizer.get_standardized_reward(rewards[-1])

    assert highest_reward_standardized_1 > highest_reward
    assert highest_reward_standardized_2 < highest_reward_standardized_1


def test_standardization_of_new_mfcc_extraction():
    analyser = Analyser(["rms", "mel"])
    sound = Sound("amen_5s.wav")
    standardizer = get_standardizer(sounds=[sound], analyser=analyser)

    sound.prepare_to_render(analyser=analyser)
    sound.features = []

    done = False

    while not done:
        done = done or sound.render()
        raw_features = sound.player.get_channels(analyser.analysis_features)
        standardized_features = np.array(
            [
                standardizer.get_standardized_value(
                    analyser.analysis_features[i], feature_value
                )
                for i, feature_value in enumerate(raw_features)
            ]
        )
        sound.features.append(standardized_features)

    mel_features = np.array(sound.features)[:, 1:]  # ignore RMS
    assert -1.0 < mel_features.mean() < 1.0
    assert -1.0 < mel_features.std() < 1.0
    assert mel_features.max() <= 4.0
    assert mel_features.min() >= -4.0
