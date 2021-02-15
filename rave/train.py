import os
import ray
from ray.rllib.agents import ppo
from ray.rllib.agents import sac
from ray.tune.logger import pretty_print
import env
from effect import Effect
from metrics import EuclideanDistance

RAY_RESULTS_DIR = "rave/ray_results"
assert os.path.isdir(RAY_RESULTS_DIR)


def train():
    config = sac.DEFAULT_CONFIG.copy()
    config["env"] = env.CrossAdaptiveEnv
    config["env_config"] = env.DEFAULT_CONFIG
    config["num_workers"] = 0
    config["framework"] = "torch"
    config["log_level"] = "WARN"

    agent = sac.SACTrainer(config=config)
    analysis = ray.tune.run(
        sac.SACTrainer,
        config=config,
        # stop={
        #     # "training_iteration": 10
        #     # "episode_reward_mean": 0.8,
        # },
        local_dir=RAY_RESULTS_DIR
    )


if __name__ == "__main__":
    train()
