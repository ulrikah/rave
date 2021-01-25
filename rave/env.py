import gym
import numpy as np

import subprocess
import os

from metrics import EuclideanDistance, AbstractMetric
from effect import Effect
from sound import Sound
from mediator import Mediator


class Env(gym.Env):
    """
    Environment for learning crossadaptive processing with reinforcement learning
    """

    def __init__(self, effect: Effect, metric: AbstractMetric):
        # TODO: define target and source sounds
        self.source = Sound()
        self.target = None

        self.effect = effect
        self.metric = metric
        self.mediator = Mediator()

        lows = np.array([p.mapping.min_value for p in effect.parameters])
        highs = np.array([p.mapping.max_value for p in effect.parameters])
        self.action_space = gym.spaces.Box(
            low=lows, high=highs, dtype=np.float)

        # TODO: set this dynamically based on analysis params
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,))

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action)
        """
        Algorithm:
            Use new action to generate new frames of audio
            Analyse the new frames in CSound and send back features
            Use features to calculate reward
        """
        mapping = self.effect.mapping_from_array(action)
        self.source.apply_effect(self.effect, mapping)

    def get_state(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def calculate_reward(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError


if __name__ == "__main__":
    effect = Effect("distortion")
    metric = EuclideanDistance()
    env = Env(effect, metric)
    action = env.action_space.sample()
    env.step(action)
    subprocess.run(["csound", "rave/analyzer.csd"])
    print("q", env.mediator.q.qsize())
    env.mediator.terminate()  # TODO: better to wrap this in a context using with
