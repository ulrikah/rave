from pathlib import Path
import argparse
import ray
from ray import tune
from ray.rllib.agents import sac
from ray.tune.progress_reporter import CLIReporter

from rave.env import CrossAdaptiveEnv
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

    parser.add_argument(
        "--label",
        dest="label",
        action="store",
        default=None,
        help="Label the training with a specific name",
    )

    return parser.parse_args()


def train(config: dict, checkpoint_path: str = None):
    ray.init(local_mode=config["ray"]["local_mode"])

    env_config = {
        "effect": config["env"]["effect"],
        "metric": config["env"]["metric"],
        "feature_extractors": config["env"]["feature_extractors"],
        "source": config["env"]["source"],
        "targets": config["env"]["targets"],
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
        "learning_starts": 10000,
        "optimization": {
            "actor_learning_rate": learning_rate,
            "critic_learning_rate": learning_rate,
            "entropy_learning_rate": learning_rate,
        },
        # Model options for the Q network(s).
        "Q_model": {
            "fcnet_activation": "tanh",
            "fcnet_hiddens": config["agent"]["hidden_layers"],
        },
        # Model options for the policy function.
        "policy_model": {
            "fcnet_activation": "tanh",
            "fcnet_hiddens": config["agent"]["hidden_layers"],
        },
    }

    if checkpoint_path:
        # NOTE: hacky way to find the corresponding Tune 'name' of the
        # restored experiment since the checkpoint is always three levels deeper
        path = Path(checkpoint_path)
        name = path.parent.parent.parent.name
    else:
        agent_name = sac.__name__.split(".")[-1].upper()  # i.e. 'SAC or 'PPO'
        name = f'{config["name"]}_{agent_name}_{timestamp(millis=False)}'

    progress_reporter = CLIReporter(max_report_frequency=15)

    analysis = tune.run(
        sac.SACTrainer,
        config=agent_config,
        local_dir=RAY_RESULTS_DIR,
        checkpoint_at_end=config["agent"]["checkpoint_at_end"],
        checkpoint_freq=config["agent"]["checkpoint_freq"],
        name=name,
        restore=checkpoint_path,  # None is default
        progress_reporter=progress_reporter,
    )
    print(analysis)


if __name__ == "__main__":
    args = args()
    config = parse_config_file(args.config_file)
    if args.label:
        config["name"] = args.label
    else:
        config["name"] = Path(args.config_file).stem
    train(config, args.checkpoint_path)
