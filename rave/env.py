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
from tools import timestamp

AMEN = "amen_trim.wav"
NOISE = "noise.wav"
ANALYSIS_CHANNELS = ["rms", "pitch_n", "centroid", "flux"]


class Mode(Enum):
    LIVE = "live"
    STATIC = "static"


class Env(gym.Env):
    """
    Environment for learning crossadaptive processing with reinforcement learning
    """

    def __init__(self, effect: Effect, metric: AbstractMetric, mode: Mode):
        # TODO: define target and source sounds from settings or similar
        self.source_input = NOISE
        self.target_input = AMEN
        self.source = Sound(self.source_input)
        self.target = Sound(self.target_input)

        self.effect = effect
        self.metric = metric
        self.mode = mode
        self.mapping = self.effect.random_mapping()
        self.mappings = []

        lows = np.array([p.mapping.min_value for p in effect.parameters])
        highs = np.array([p.mapping.max_value for p in effect.parameters])
        self.action_space = gym.spaces.Box(
            low=lows, high=highs, dtype=np.float)

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(len(ANALYSIS_CHANNELS),))

        self.source.apply_effect(
            effect=self.effect, analyzer_osc_route="/rave/source/features")
        self.target.apply_effect(
            effect=None, analyzer_osc_route="/rave/target/features")

        if self.mode == Mode.LIVE:
            self.mediator = Mediator()

    def step(self, action: np.ndarray):
        """
        Algorithm:
            Use new action to generate new frames of audio
            Analyse the new frames in CSound and send back features
            Use features to calculate reward
        """
        assert self.action_space.contains(action)

        self.mapping = self.effect.mapping_from_array(action)
        self.mappings.append(self.mapping)
        source_done = self.source.render(mapping=self.mapping)
        target_done = self.target.render()

        if self.mode == Mode.LIVE:
            source, target = self.mediator.get_features()
        else:
            source = torch.tensor(
                self.source.player.get_channels(ANALYSIS_CHANNELS))
            target = torch.tensor(
                self.target.player.get_channels(ANALYSIS_CHANNELS))

        reward = self.calculate_reward(source, target)
        done = source_done or target_done
        if done:
            self.render()
            self._reset_mappings()
        return self.get_state(), reward, done, {}

    def get_state(self):
        return self.mapping

    def _reset_mappings(self):
        self.mapping = self.effect.random_mapping()
        self.mappings = [self.mapping]

    def reset(self):
        if self.mode == LIVE:
            self.mediator.clear()
        self._reset_mappings()
        return self.get_state()

    def calculate_reward(self, source, target):
        assert source.shape == target.shape
        return self.metric.calculate_reward(source, target)

    def close(self):
        if self.mode == Mode.Live:
            self.mediator.terminate()
        raise NotImplementedError

    def render(self):
        """
        Renders a file with all the mappings from the episode
        """
        done = False
        source = Sound(self.source_input,
                       output_file_path=f"{self.effect.name}_render_{timestamp()}.wav", loop=False)
        source.apply_effect(effect=self.effect)
        for mapping in self.mappings[:-1]:
            done = source.render(mapping=mapping)
            if done:
                break


if __name__ == "__main__":
    effect = Effect("bandpass")
    metric = EuclideanDistance()
    env = Env(effect, metric, mode=Mode.STATIC)

    N = 10000
    for i in range(N):
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
        if i % 1000 == 0:
            print(f"\nREWARD: {reward} \n")

    # TODO: where do we send the next audio?
