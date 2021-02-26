import ray
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents import sac
from gym import Env
import numpy as np

import argparse

from rave.env import CrossAdaptiveEnv, CROSS_ADAPTIVE_DEFAULT_CONFIG
from rave.effect import Effect
from rave.metrics import metric_from_name
from rave.config import parse_config_file


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--checkpoint",
        dest="checkpoint_path",
        action="store",
        help="Path to a checkpoint from a training session",
    )
    parser.add_argument(
        "--config",
        dest="config_file",
        action="store",
        default="default.toml",
        help="Path to a config file",
    )

    parser.add_argument(
        "-s",
        "--source",
        dest="source_sound",
        action="store",
        default=None,
        help="Specify the source input to use. Should either by a .wav file or adc for live input",
    )

    parser.add_argument(
        "-t",
        "--target",
        dest="target_sound",
        action="store",
        default=None,
        help="Specify the target input to use. Should either by a .wav file or adc for live input",
    )
    return parser.parse_args()


def inference(
    config_path: str,
    checkpoint_path: str,
    source_sound: str = None,
    target_sound: str = None,
):
    """
    Runs inference on a pretrained agent

    Args:
        config: config dict
        checkpoint_path: path to checkpoint from which to load the pretrained agent
        source_sound: an input sound source
        target_sound: a target sound source to evaluate the model against
    """

    # NOTE: går an å teste om man egentlig trenger å sette config, siden jeg allerede gjør agent.restore()

    ray.init(local_mode=config["ray"]["local_mode"])

    env_config = {
        "effect": Effect(config["env"]["effect"]),
        "metric": metric_from_name(config["env"]["metric"]),
        "feature_extractors": config["env"]["feature_extractors"],
        "source": source_sound if source_sound else config["env"]["source"],
        "target": target_sound if target_sound else config["env"]["target"],
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

    env = CrossAdaptiveEnv(env_config)

    agent = sac.SACTrainer(config=agent_config)
    agent.restore(checkpoint_path)

    episode_index = 0
    episode_reward = []
    done = False
    obs = env.reset()
    while episode_index < 5:
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward.append(reward)
        if done:
            episode_index += 1
            print("\n" * 5)
            print("DOOOOONE", episode_index)
            print("mean episode reward:", np.mean(episode_reward))
            print("\n" * 5)
            episode_reward = []


if __name__ == "__main__":
    args = args()
    print(args)

    config = parse_config_file(args.config_file)
    inference(
        config,
        checkpoint_path=args.checkpoint_path,
        source_sound=args.source_sound,
        target_sound=args.target_sound,
    )
