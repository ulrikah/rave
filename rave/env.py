import gym
import numpy as np
import torch

import subprocess
import os
import sys
from enum import Enum

from rave.metrics import EuclideanDistance, AbstractMetric
from rave.effect import Effect
from rave.analyser import Analyser
from rave.sound import Sound
from rave.mediator import Mediator
from rave.tools import timestamp, play_wav

AMEN = "amen_trim.wav"
NOISE = "noise.wav"
SINE = "sine220.wav"


class Mode(Enum):
    LIVE = "live"
    STATIC = "static"


CROSS_ADAPTIVE_DEFAULT_CONFIG = {
    "effect": Effect("bandpass"),
    "metric": EuclideanDistance(),
    "mode": Mode.STATIC,
    "source": NOISE,
    "target": AMEN,
    "feature_extractors": ["rms", "pitch", "spectral"],
}


class CrossAdaptiveEnv(gym.Env):
    """
    Environment for learning crossadaptive processing with reinforcement learning
    """

    def __init__(self, config=CROSS_ADAPTIVE_DEFAULT_CONFIG):
        self.source_input = config["source"]
        self.target_input = config["target"]
        self.effect = config["effect"]
        self.metric = config["metric"]
        self.mode = config["mode"]
        self.feature_extractors = config["feature_extractors"]

        if not len(self.feature_extractors) > 0:
            raise ValueError(
                "The environment doesn't work without any feature extractors"
            )
        analyser = Analyser(self.feature_extractors)
        self.analysis_features = analyser.analysis_features

        self.source = Sound(self.source_input)
        self.target = Sound(self.target_input)

        # an observation = 1 x audio frame from both source and target => 2 x length of features
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(len(self.analysis_features) * 2,)
        )

        # an action = a combination of effect parameters
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(len(self.effect.parameters),)
        )

        self.source.prepare_to_render(effect=self.effect, analyser=analyser)

        self.target.prepare_to_render(effect=None, analyser=analyser)

        if self.mode == Mode.LIVE:
            self.mediator = Mediator()

        self.actions = []
        self.rewards = []
        self.source_features = np.zeros(shape=len(self.analysis_features))
        self.target_features = np.zeros(shape=len(self.analysis_features))

    def action_to_mapping(self, action: np.ndarray):
        assert len(action) == len(
            self.effect.parameters
        ), "Number of params doesn't match length of action"
        mapping = {}
        for i, p in enumerate(self.effect.parameters):
            mapping[p.name] = self.map_action_to_effect_parameter(
                action[i],
                p.mapping.min_value,
                p.mapping.max_value,
                p.mapping.skew_factor,
            )
        return mapping

    @staticmethod
    def map_action_to_effect_parameter(x, min_value, max_value, skew_factor):
        """
        Scaling outputs of the neural network in the [0, 1] range to a desired range with a skew factor.
        The skew factor is a way of creating non-linear mappings. The mapping can be made linear by setting
        the skew factor to 1.0.

        This mapping trick is an idea borrowed from Jordal (2017) and Walsh (2008).
        """
        return min_value + (max_value - min_value) * np.exp(np.log(x) / skew_factor)

    def step(self, action: np.ndarray):
        """
        Algorithm:
            Use new action to generate new frames of audio
            Analyse the new frames in CSound and send back features
            Use features to calculate reward
        """
        assert self.action_space.contains(action)
        self.actions.append(action)

        mapping = self.action_to_mapping(action)

        source_done = self.source.render(mapping=mapping)
        target_done = self.target.render()

        if self.mode == Mode.LIVE:
            source_features, target_features = self.mediator.get_features()
        else:
            source_features = self.source.player.get_channels(self.analysis_features)
            target_features = self.target.player.get_channels(self.analysis_features)

        self.source_features = source_features
        self.target_features = target_features

        reward = self.calculate_reward(source_features, target_features)
        done = source_done

        if done:
            self.render()
            self.reset()
        return self.get_state(), reward, done, {}

    def get_state(self):
        return np.concatenate((self.source_features, self.target_features))

    def reset(self):
        if self.mode == Mode.LIVE:
            self.mediator.clear()
        self.actions = []
        self.rewards = []
        return self.get_state()

    def calculate_reward(self, source, target):
        assert source.shape == target.shape
        reward = self.metric.calculate_reward(source, target)
        self.rewards.append(reward)
        return reward

    def close(self):
        if self.mode == Mode.LIVE:
            self.mediator.terminate()
        for sound in [self.source, self.target]:
            if sound.player is not None:
                sound.player.cleanup()

    def render(self):
        """
        Renders a file with all the actions from the episode
        """
        done = False
        source = Sound(
            self.source_input,
            output_file_path=f"{self.effect.name}_render_{timestamp()}_{os.path.basename(self.source_input)}",
            loop=False,
        )
        source.prepare_to_render(effect=self.effect)
        for action in self.actions:
            mapping = self.action_to_mapping(action)
            done = source.render(mapping=mapping)
            if done:
                break
        return source.output


if __name__ == "__main__":
    env = CrossAdaptiveEnv()

    N = 10000
    for i in range(N):
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        if i % 1000 == 0:
            print(f"\nREWARD: {reward} \n")
    path = env.render()
    play_wav(path)
