import gym
import numpy as np

import subprocess
import os
import sys

from metrics import EuclideanDistance, AbstractMetric
from effect import Effect
from sound import Sound
from mediator import Mediator

AMEN = "amen_trim.wav"
NOISE = "noise.wav"


class Env(gym.Env):
    """
    Environment for learning crossadaptive processing with reinforcement learning
    """

    def __init__(self, effect: Effect, metric: AbstractMetric):
        # TODO: define target and source sounds from settings or similar
        self.source = Sound(NOISE)
        self.target = Sound(AMEN)

        self.effect = effect
        self.metric = metric
        self.mediator = Mediator()
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

        # TODO: get features should be dependent on whether mode is LIVE or STATIC, i.e. wheter features are sent over channels or OSC
        # source, target = self.mediator.get_features()
        # reward = self.calculate_reward(source, target)

        # return self.get_state(), reward, False, {}  # gym.Env format

    def get_state(self):
        return self.mapping

    def reset(self):
        self.mediator.clear()

    def calculate_reward(self, source, target):
        return self.metric.calculate_reward(source, target)

    def close(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError


if __name__ == "__main__":
    effect = Effect("distortion")
    metric = EuclideanDistance()
    env = Env(effect, metric)
    action = env.action_space.sample()
    # state, reward, _, _ = env.step(action)
    env.step(action)

    # where do we send the next audio?

    env.mediator.terminate()  # TODO: better to wrap this in a context using with
