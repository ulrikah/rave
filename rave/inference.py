import ray
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents import sac
from ray.rllib.agents import ppo
import numpy as np

import argparse

from rave.env import CrossAdaptiveEnv
from rave.config import parse_config_file
from rave.mediator import Mediator


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckp",
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

    parser.add_argument(
        "--dac",
        dest="render_to_dac",
        action="store_true",
        default=False,
        help="If the inference should output directly to the speakers",
    )
    parser.add_argument(
        "--live",
        dest="live_mode",
        action="store_true",
        default=False,
        help="Live mode or not",
    )
    return parser.parse_args()


def inference(
    config_path: str,
    checkpoint_path: str,
    source_sound: str = None,
    target_sound: str = None,
    render_to_dac=False,
    live_mode=True,
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
        "effect": config["env"]["effect"],
        "metric": config["env"]["metric"],
        "feature_extractors": config["env"]["feature_extractors"],
        "source": source_sound if source_sound else config["env"]["source"],
        "targets": [target_sound] if target_sound else config["env"]["targets"],
        "eval_interval": None,
        "render_to_dac": render_to_dac,
        "standardize_rewards": False,  # NOTE: experimental feature
        "debug": config["env"]["debug"],
    }

    learning_rate = (
        config["agent"]["learning_rate"]
        if "learning_rate" in config["agent"].keys()
        else 3e-3
    )

    hidden_layers = config["agent"]["hidden_layers"]
    tanh = "tanh"

    common_config = {
        "env": CrossAdaptiveEnv,
        "env_config": env_config,
        "framework": "torch",
        "num_cpus_per_worker": config["ray"]["num_cpus_per_worker"],
        "log_level": config["ray"]["log_level"],
        # "observation_filter": "MeanStdFilter",  # NB! some (old) runs didn't not include this, so that might cause crashes
        "num_workers": 0,
        "train_batch_size": 256,
        "explore": False,
    }

    def sac_trainer():
        agent_name = "SAC"
        sac_config = {
            **sac.DEFAULT_CONFIG.copy(),
            **common_config.copy(),
            "learning_starts": 10000 if not checkpoint_path else 0,
            "target_entropy": -24,  # set empirically after trials with dist_lpf
            "optimization": {
                "actor_learning_rate": learning_rate,
                "critic_learning_rate": learning_rate,
                "entropy_learning_rate": learning_rate,
            },
            # Model options for the Q network(s).
            "Q_model": {
                "fcnet_activation": tanh,
                "fcnet_hiddens": hidden_layers,
            },
            # Model options for the policy function.
            "policy_model": {
                "fcnet_activation": tanh,
                "fcnet_hiddens": hidden_layers,
            },
        }
        return sac.SACTrainer, sac_config, agent_name

    def ppo_trainer():
        agent_name = "PPO"
        ppo_config = {
            **ppo.DEFAULT_CONFIG.copy(),
            **common_config.copy(),
            "lr": learning_rate,
            "model": {
                "fcnet_hiddens": hidden_layers,
                "fcnet_activation": tanh,
            },
            "sgd_minibatch_size": 64,
            # Coefficient of the entropy regularizer. Unused if a schedule if set
            "entropy_coeff": 0.0,
            # Decay schedule for the entropy regularizer.
            "entropy_coeff_schedule": None,
        }
        return ppo.PPOTrainer, ppo_config, agent_name

    agent = config["agent"]["agent"]
    available_trainers = ["sac", "ppo"]
    no_agent_error = ValueError(f"{agent} not available")
    if agent not in available_trainers:
        raise no_agent_error
    elif agent == "sac":
        trainer, agent_config, agent_name = sac_trainer()
    elif agent == "ppo":
        trainer, agent_config, agent_name = ppo_trainer()

    env = CrossAdaptiveEnv(env_config)
    agent = trainer(config=agent_config)
    agent.restore(checkpoint_path)

    if live_mode:
        run_live_inference(agent, env)
    else:
        run_offline_inference(agent, env)


def run_live_inference(
    agent: Trainer,
    env: CrossAdaptiveEnv,
):
    mediator = Mediator()

    episode_index = 0
    while episode_index < 1500:
        source_features, target_features = mediator.get_features()
        if source_features is None or target_features is None:
            continue

        standardized_source = np.array(
            [
                env.standardizer.get_standardized_value(
                    env.analyser.analysis_features[i], feature_value
                )
                for i, feature_value in enumerate(source_features)
            ]
        )
        standardized_target = np.array(
            [
                env.standardizer.get_standardized_value(
                    env.analyser.analysis_features[i], feature_value
                )
                for i, feature_value in enumerate(target_features)
            ]
        )
        obs = np.concatenate((standardized_source, standardized_target))
        print(np.round(obs, decimals=2))
        action = agent.compute_action(obs)
        # action = env.action_space.sample()
        mapping = env.action_to_mapping(action)
        # print(mapping)
        mediator.send_effect_mapping(mapping)
        episode_index += 1
    mediator.terminate()
    print("\n\n\tDONE\n\n")


def run_offline_inference(agent: Trainer, env: CrossAdaptiveEnv):
    # NOTE: something is wrong here. For some reason, all the action values are too close to the bound
    done = False
    obs = env.reset()
    while not done:
        action = agent.compute_action(obs)
        # TODO: standardize action
        # it might be difficult to standardize the action in live mode, but offline inference essentially work
        obs, _, done, _ = env.step(action)


if __name__ == "__main__":
    args = args()
    config = parse_config_file(args.config_file)
    inference(
        config,
        checkpoint_path=args.checkpoint_path,
        source_sound=args.source_sound,
        target_sound=args.target_sound,
        render_to_dac=args.render_to_dac,
        live_mode=args.live_mode,
    )
