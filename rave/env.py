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
from rave.constants import DAC

AMEN = "amen_trim.wav"
NOISE = "noise.wav"
SINE = "sine220.wav"


CROSS_ADAPTIVE_DEFAULT_CONFIG = {
    "effect": Effect("bandpass"),
    "metric": EuclideanDistance(),
    "source": NOISE,
    "target": AMEN,
    "feature_extractors": ["rms", "pitch", "spectral"],
    "live_mode": False,
    "eval_interval": 1,
    "render_to_dac": False,
    "debug": False,
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
        self.live_mode = config["live_mode"]
        self.feature_extractors = config["feature_extractors"]
        self.feature_extractors = config["feature_extractors"]
        self.render_to_dac = config["render_to_dac"]
        self.debug = config["debug"]

        # how often the model should evaluate
        self.eval_interval = config["eval_interval"]
        if self.eval_interval is not None:
            self.step_index = 0

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

        if self.live_mode:
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
            if self.debug:
                mapping[f"{p.name}_debug"] = action[i]
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
            1. Use new action to generate a new frame of audio for both source and target
            2. Analyse the new frame in Csound and send back features
            3. Use features to calculate reward
        """
        assert self.action_space.contains(action)
        self.actions.append(action)

        mapping = self.action_to_mapping(action)

        source_done = self.source.render(mapping=mapping)
        target_done = self.target.render()

        if self.live_mode:
            source_features, target_features = self.mediator.get_features()
        else:
            source_features = self.source.player.get_channels(self.analysis_features)
            target_features = self.target.player.get_channels(self.analysis_features)

        self.source_features = source_features
        self.target_features = target_features

        reward = self.calculate_reward(source_features, target_features)

        """
        An episode is either over at the end of every interval (if using this mechanism),
        or when the rendering of the source sound is complete. However, we only reset and
        bounce when the source sound is complete
        """

        done = source_done

        if self.eval_interval is not None:
            self.step_index += 1
            should_evaluate = (
                self.step_index != 0 and self.step_index % self.eval_interval == 0
            )
            if should_evaluate:
                self.step_index = 0
                done = True

        if source_done:
            self.render()
            self._reset_internal_state()

        return self.get_state(), reward, done, {}

    def get_state(self):
        return np.concatenate((self.source_features, self.target_features))

    def _reset_internal_state(self):
        if self.live_mode:
            self.mediator.clear()
        self.actions = []
        self.rewards = []

    def reset(self):
        # NOTE: rrlib calls reset internally, so to control the behavior this only returns the state
        return self.get_state()

    def calculate_reward(self, source, target):
        assert source.shape == target.shape
        reward = self.metric.calculate_reward(source, target)
        self.rewards.append(reward)
        return reward

    def close(self):
        if self.live_mode:
            self.mediator.terminate()
        for sound in [self.source, self.target]:
            if sound.player is not None:
                sound.player.cleanup()

    def render(self):
        """
        Renders a file with all the actions from the episode
        """
        done = False
        if self.render_to_dac:
            output = DAC
        else:
            output = f"{timestamp()}_render_{self.effect.name}_{os.path.basename(self.source_input)}"

        source = Sound(
            self.source_input,
            output=output,
            loop=False,
        )
        source.prepare_to_render(effect=self.effect, add_debug_channels=self.debug)
        for action in self.actions:
            mapping = self.action_to_mapping(action)
            done = source.render(mapping=mapping)
            if done:
                break
        return source
