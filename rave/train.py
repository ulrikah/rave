import os
import ray
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print
import env
from effect import Effect
from metrics import EuclideanDistance

RAY_RESULTS_DIR = "rave/ray_results"
assert os.path.isdir(RAY_RESULTS_DIR)


# common config: https://docs.ray.io/en/master/rllib-training.html#common-parameters
# PPO config: https://docs.ray.io/en/master/rllib-algorithms.html#ppo
config = ppo.DEFAULT_CONFIG.copy()

config["env"] = env.CrossAdaptiveEnv
config["env_config"] = env.DEFAULT_CONFIG
config["num_workers"] = 0
config["framework"] = "torch"
# config["model"] : {}, # https://github.com/ray-project/ray/blob/master/rllib/models/catalog.py
config["normalize_actions"] = True
config["log_level"] = "WARN"
# config["log_sys_usage"] = True


def train_naively():
    ray.init()
    agent = ppo.PPOTrainer(config=config)
    env = env.CrossAdaptiveEnv(env.DEFAULT_CONFIG)
    obs = env.reset()
    done = False

    for i in range(100):
        agent
        action = agent.compute_action(obs)
        obs, reward, done, _ = env.step(action)
        print({"reward": reward, "action": action, "obs": obs})


if __name__ == "__main__":
    analysis = ray.tune.run(
        ppo.PPOTrainer,
        config=config,
        stop={
            "episode_reward_mean": 0.8
        },
        local_dir=RAY_RESULTS_DIR,
        num_samples=100)


# import os
# import ray
# from ray.rllib.agents import sac
# from ray.tune.logger import pretty_print
# import env
# from effect import Effect
# from metrics import EuclideanDistance

# RAY_RESULTS_DIR = "rave/ray_results"
# assert os.path.isdir(RAY_RESULTS_DIR)

# # common config: https://docs.ray.io/en/master/rllib-training.html#common-parameters
# # PPO config: https://docs.ray.io/en/master/rllib-algorithms.html#ppo
# config = sac.DEFAULT_CONFIG.copy()

# config["num_workers"] = 0
# config["framework"] = "torch"
# config["monitor"] = False
# config["log_level"] = "WARN"
# config["log_sys_usage"] = True

# config["env"] = env.CrossAdaptiveEnv
# config["env_config"] = env.DEFAULT_CONFIG

# # config["model"] : {}, # https://github.com/ray-project/ray/blob/master/rllib/models/catalog.py
# config["normalize_actions"] = True
# # config["lr"] = 1e-5


# def train():
#     agent = sac.SACTrainer(config=config)

#     for i in range(1000):
#         result = agent.train()
#         print(pretty_print(result))


# if __name__ == "__main__":
#     ray.init()
#     train()
