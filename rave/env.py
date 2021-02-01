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
        self.source = Sound(NOISE)
        self.target = Sound(AMEN)

        self.effect = effect
        self.metric = metric
        self.mode = mode
        self.mapping = None  # or init to random?

        lows = np.array([p.mapping.min_value for p in effect.parameters])
        highs = np.array([p.mapping.max_value for p in effect.parameters])
        self.action_space = gym.spaces.Box(
            low=lows, high=highs, dtype=np.float)

        # TODO: set this dynamically based on analysis params
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,))

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
        self.source.render(mapping=self.mapping)
        self.target.render()

        if self.mode == Mode.LIVE:
            source, target = self.mediator.get_features()
        else:
            source = torch.tensor(
                self.source.player.get_channels(ANALYSIS_CHANNELS))
            target = torch.tensor(
                self.target.player.get_channels(ANALYSIS_CHANNELS))

        reward = self.calculate_reward(source, target)
        return self.get_state(), reward, False, {}

    def get_state(self):
        assert self.mapping is not None
        return self.mapping

    def reset(self):
        if self.mode == LIVE:
            self.mediator.clear()
        raise NotImplementedError

    def calculate_reward(self, source, target):
        assert source.shape == target.shape
        return self.metric.calculate_reward(source, target)

    def close(self):
        if self.mode == Mode.Live:
            self.mediator.terminate()
        raise NotImplementedError

    def render(self):
        raise NotImplementedError


if __name__ == "__main__":
    effect = Effect("distortion")
    metric = EuclideanDistance()
    env = Env(effect, metric, mode=Mode.STATIC)
    for i in range(100):
        action = env.action_space.sample()
        state, reward, _, _ = env.step(action)
        print(reward)

    # TODO: where do we send the next audio?
