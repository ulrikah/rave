import ray
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents import sac
from gym import Env

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
    Runs inference against a pretrained agent

    Args:
        checkpoint_path: path to checkpoint from which to load the pretrained agent
        input_sound: a source sound to run inference against
    """

    # TOOD: dette burde ikke trenge å være med
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

    # run until episode ends
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward


if __name__ == "__main__":
    args = args()
    ray.init(local_mode=True)
    inference(args.checkpoint_path, args.input_sound)