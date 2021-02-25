import ray
from ray.rllib.agents.trainer import Trainer
from ray.rllib.agents import sac
from gym import Env

from rave.env import CrossAdaptiveEnv, CROSS_ADAPTIVE_DEFAULT_CONFIG
from rave.effect import Effect


def inference(checkpoint_path: str):
    """
    Runs inference against a pretrained agent

    Args:
        checkpoint_path: path to checkpoint from which to load the pretrained agent
    """

    # TOOD: dette burde ikke trenge å være med
    env_config = CROSS_ADAPTIVE_DEFAULT_CONFIG
    env_config["effect"] = Effect("dist_lpf")
    env_config["feature_extractors"] = ["rms"]
    env_config["source"] = "MoreNight_flips_a_dirty_Scottish_voicemail.wav"

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
    ray.init(local_mode=True)
    checkpoint_path = "/Users/ulrikah/fag/thesis/rave/rave/ray_results/SAC_2021-02-25_11-12-41/SAC_CrossAdaptiveEnv_fe899_00000_0_2021-02-25_11-12-41/checkpoint_400/checkpoint-400"
    inference(checkpoint_path)
