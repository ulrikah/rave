import gym
import numpy as np
import torch

import subprocess
import os
import sys
from enum import Enum

from metrics import EuclideanDistance, AbstractMetric
from effect import Effect
from sound import Sound
from mediator import Mediator
from tools import timestamp, play_wav

AMEN = "amen_trim.wav"
NOISE = "noise.wav"
SINE = "sine220.wav"
ANALYSIS_CHANNELS = ["rms", "pitch_n", "centroid", "flux"]


class Mode(Enum):
    LIVE = "live"
    STATIC = "static"


DEFAULT_CONFIG = {
    "effect": Effect("bandpass"),
    "metric": EuclideanDistance(),
    "mode": Mode.STATIC,
    "source": NOISE,
    "target": AMEN
}


class CrossAdaptiveEnv(gym.Env):
    """
    Environment for learning crossadaptive processing with reinforcement learning
    """

    def __init__(self, config=DEFAULT_CONFIG):
        self.source_input = config["source"]
        self.target_input = config["target"]
        self.effect = config["effect"]
        self.metric = config["metric"]
        self.mode = config["mode"]

        self.source = Sound(self.source_input)
        self.target = Sound(self.target_input)

        # an observation = 1 x audio frame from both source and target
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(len(ANALYSIS_CHANNELS) * 2,))

        # an action = a combination of effect parameters
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(len(self.effect.parameters),))

        self.source.apply_effect(
            effect=self.effect, analyze=True)
        self.target.apply_effect(
            effect=None, analyze=True)

        if self.mode == Mode.LIVE:
            self.mediator = Mediator()

        self.actions = []
        self.rewards = []
        self.source_features = np.zeros(shape=len(ANALYSIS_CHANNELS))
        self.target_features = np.zeros(shape=len(ANALYSIS_CHANNELS))

    def action_to_mapping(self, action: np.ndarray):
        assert len(action) == len(
            self.effect.parameters), "Number of params doesn't match length of action"
        mapping = {}
        for i, p in enumerate(self.effect.parameters):
            fp = [p.mapping.min_value, p.mapping.max_value]
            mapping[p.name] = np.interp(action[i], [0.0, 1.0], fp)
        return mapping

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
            source_features = self.source.player.get_channels(
                ANALYSIS_CHANNELS)
            target_features = self.target.player.get_channels(
                ANALYSIS_CHANNELS)

        self.source_features = source_features
        self.target_features = target_features

        reward = self.calculate_reward(source_features, target_features)
        done = source_done or target_done

        if done:
            self.render()
            self.reset()
        return self.get_state(), reward, done, {}

    def get_state(self):
        return np.concatenate(self.source_features, self.target_features)

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
            self.source_input, output_file_path=f"{self.effect.name}_render_{timestamp()}_{self.source_input}", loop=False)
        source.apply_effect(effect=self.effect)
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
