import os
from pathlib import Path
import argparse
import ray
from ray import tune
from ray.rllib.agents import sac
from ray.tune.logger import pretty_print
from ray.tune.progress_reporter import CLIReporter

from rave.env import CrossAdaptiveEnv
from rave.effect import Effect
from rave.metrics import metric_from_name
from rave.tools import timestamp
from rave.config import parse_config_file
from rave.constants import RAY_RESULTS_DIR


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        dest="config_file",
        action="store",
        default="default.toml",
        help="Path to a config file",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint_path",
        action="store",
        default=None,
        help="Path to a checkpoint from a training session (for resuming training)",
    )

    return parser.parse_args()


def train(config: dict, checkpoint_path: str = None):
    ray.init(local_mode=config["ray"]["local_mode"])

    env_config = {
        "effect": Effect(config["env"]["effect"]),
        "metric": metric_from_name(config["env"]["metric"]),
        "feature_extractors": config["env"]["feature_extractors"],
        "source": config["env"]["source"],
        "target": config["env"]["target"],
        "eval_interval": config["env"]["eval_interval"],
        "render_to_dac": False,
        "debug": config["env"]["debug"],
    }

    learning_rate = 3e-3
    agent_config = {
        **sac.DEFAULT_CONFIG.copy(),
        "env": CrossAdaptiveEnv,
        "env_config": env_config,
        "framework": "torch",
        "num_cpus_per_worker": config["ray"]["num_cpus_per_worker"],
        "log_level": config["ray"]["log_level"],
        # Model options for the Q network(s).
        "Q_model": {
            "fcnet_activation": config["agent"]["activation"],
            "fcnet_hiddens": config["agent"]["hidden_layers"],
        },
        # Model options for the policy function.
        "policy_model": {
            "fcnet_activation": config["agent"]["activation"],
            "fcnet_hiddens": config["agent"]["hidden_layers"],
        },
        "optimization": {
            "actor_learning_rate": learning_rate,
            "critic_learning_rate": learning_rate,
            "entropy_learning_rate": learning_rate,
        },
    }

    if checkpoint_path:
        # NOTE
        # hacky way to find the corresponding Tune 'name' of the restored experiment since
        # the checkpoint is always three levels deeper
        path = Path(checkpoint_path)
        name = path.parent.parent.parent.name
    else:

        agent_name = sac.__name__.split(".")[-1].upper()  # i.e. 'SAC or 'PPO
        name = f'{agent_name}_{config["name"]}_{timestamp(millis=False)}'

    progress_reporter = CLIReporter(max_report_frequency=15)

    ###############
    # Hyperparameter search

    hidden_layer_sizes = [4, 8, 16, 32]
    learning_rates = [3e-3, 3e-4, 3e-5]

    agent_config = tune.grid_search(
        [
            {
                **agent_config.copy(),
                "optimization": {
                    "actor_learning_rate": lr,
                    "critic_learning_rate": lr,
                    "entropy_learning_rate": lr,
                },
            }
            for lr in learning_rates
        ]
        # [
        #     {
        #         **agent_config,
        #         "Q_model": {
        #             "fcnet_hiddens": size,
        #         },
        #         "policy_model": {
        #             "fcnet_hiddens": size,
        #         },
        #     }
        #     for size in hidden_layer_sizes
        # ]
    )
    ###############

    analysis = tune.run(
        sac.SACTrainer,
        config=agent_config,
        local_dir=RAY_RESULTS_DIR,
        checkpoint_at_end=config["agent"]["checkpoint_at_end"],
        checkpoint_freq=config["agent"]["checkpoint_freq"],
        name=name,
        restore=checkpoint_path,  # None is default
        progress_reporter=progress_reporter,
        stop={"timesteps_total": 50000},
    )


if __name__ == "__main__":
    args = args()
    config = parse_config_file(args.config_file)
    config["name"] = Path(args.config_file).stem
    train(config, args.checkpoint_path)
