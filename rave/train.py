import os
import ray
from ray.rllib.agents import ppo
from ray.rllib.agents import sac
from ray.tune.logger import pretty_print

from rave.env import CrossAdaptiveEnv, CROSS_ADAPTIVE_DEFAULT_CONFIG
from rave.effect import Effect
from rave.metrics import EuclideanDistance, NormalizedEuclidean
from rave.tools import timestamp

RAY_RESULTS_DIR = "rave/ray_results"
assert os.path.isdir(RAY_RESULTS_DIR)


def train():
    env_config = CROSS_ADAPTIVE_DEFAULT_CONFIG
    env_config["metric"] = NormalizedEuclidean()
    env_config["effect"] = Effect("dist_lpf")
    env_config["feature_extractors"] = ["rms", "pitch", "spectral"]

    config = sac.DEFAULT_CONFIG.copy()
    config["env"] = CrossAdaptiveEnv
    config["env_config"] = env_config
    config["framework"] = "torch"
    config["log_level"] = "WARN"
    config["num_cpus_per_worker"] = 4

    agent = sac.SACTrainer(config=config)
    analysis = ray.tune.run(
        sac.SACTrainer,
        config=config,
        # stop={
        #     # "training_iteration": 10
        #     # "episode_reward_mean": 0.8,
        # },
        local_dir=RAY_RESULTS_DIR,
    )


if __name__ == "__main__":
    train()
