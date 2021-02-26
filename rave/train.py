import os
import argparse
import ray
from ray.rllib.agents import sac
from ray.tune.logger import pretty_print

from rave.env import CrossAdaptiveEnv, CROSS_ADAPTIVE_DEFAULT_CONFIG
from rave.effect import Effect
from rave.metrics import metric_from_name
from rave.tools import timestamp
from rave.config import parse_config_file

RAY_RESULTS_DIR = "rave/ray_results"
assert os.path.isdir(RAY_RESULTS_DIR)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config_file",
        action="store",
        default="default.toml",
        help="Path to a config file",
    )

    return parser.parse_args()


def train(config: dict):
    ray.init(local_mode=config["ray"]["local_mode"])

    env_config = {
        "effect": Effect(config["env"]["effect"]),
        "metric": metric_from_name(config["env"]["metric"]),
        "feature_extractors": config["env"]["feature_extractors"],
        "source": config["env"]["source"],
        "target": config["env"]["target"],
        "live_mode": config["env"]["live_mode"],
    }

    agent_config = {
        **sac.DEFAULT_CONFIG.copy(),
        "env": CrossAdaptiveEnv,
        "env_config": env_config,
        "framework": "torch",
        "num_cpus_per_worker": config["ray"]["num_cpus_per_worker"],
        "log_level": config["ray"]["log_level"],
    }

    agent = sac.SACTrainer(config=agent_config)
    analysis = ray.tune.run(
        sac.SACTrainer,
        config=agent_config,
        stop=config["agent"]["stop"],
        local_dir=config["ray"]["local_dir"],
        checkpoint_at_end=config["agent"]["checkpoint_at_end"],
        checkpoint_freq=config["agent"]["checkpoint_freq"],
        name=f'{agent._name}_{env_config["effect"].name}_{"_".join(env_config["feature_extractors"])}_{timestamp()}',
    )


if __name__ == "__main__":
    args = args()
    config = parse_config_file(args.config_file)
    train(config)