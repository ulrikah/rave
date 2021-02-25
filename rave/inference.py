import ray
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents import sac
from gym import Env
import numpy as np

import argparse

from rave.env import CrossAdaptiveEnv, CROSS_ADAPTIVE_DEFAULT_CONFIG
from rave.effect import Effect


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
        "-i",
        "--input",
        dest="input_sound",
        action="store",
        default="noise.wav",
        help="Specify the input to use. Should either by a .wav file or adc for live input",
    )
    return parser.parse_args()


def inference(checkpoint_path: str, input_sound: str):
    """
    Runs inference on a sound source against a pretrained agent

    Args:
        checkpoint_path: path to checkpoint from which to load the pretrained agent
        input_sound: a source sound to run inference against
    """

    # NOTE: dette burde være definert et felles sted, i en YAML eller lignende, for å matche det som agenten har blitt trent på
    # NOTE: går også an å teste om man egentlig trenger å sette config, siden jeg allerede gjør agent.restore()
    env_config = CROSS_ADAPTIVE_DEFAULT_CONFIG
    env_config["effect"] = Effect("dist_lpf")
    env_config["feature_extractors"] = ["rms"]
    env_config["source"] = input_sound

    config = sac.DEFAULT_CONFIG.copy()
    config["env"] = CrossAdaptiveEnv
    config["env_config"] = env_config
    config["framework"] = "torch"
    config["log_level"] = "WARN"
    config["num_cpus_per_worker"] = 4

    env = CrossAdaptiveEnv(env_config)

    agent = sac.SACTrainer(config=config)
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
    ray.init(local_mode=True)
    inference(args.checkpoint_path, args.input_sound)
