import os
import gym
from gym.spaces import Discrete, Box
from ray import tune
from ray.rllib.agents.ppo import PPOTrainer
from env import CrossAdaptiveEnv
from effect import Effect
from env import Mode
from metrics import EuclideanDistance

RAY_RESULTS_DIR = "rave/ray_results"
assert os.path.isdir(RAY_RESULTS_DIR)

if __name__ == "__main__":
    cross_adaptive_config = {
        "effect": Effect("bandpass"),
        "metric": EuclideanDistance(),
        "mode": Mode.STATIC
    }

    tune.run(
        PPOTrainer,
        local_dir=RAY_RESULTS_DIR,
        config={
            "env": CrossAdaptiveEnv,
            "num_workers": 1,  # amount of CPU/GPU's to put in action
            "framework": "torch",
            "env_config": cross_adaptive_config
        }
    )
