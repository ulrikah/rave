import numpy as np


class AbstractMetric:
    def __init__(self):
        self.reward_range = None

    def calculate_reward(self, source: np.ndarray, target: np.ndarray):
        raise NotImplementedError("Must be implemented in subclass")

    @staticmethod
    def is_in_range(reward: np.ndarray, reward_range: tuple):
        return reward >= reward_range[0] and reward <= reward_range[1]


class EuclideanDistance(AbstractMetric):
    def __init__(self):
        super().__init__()
        self.reward_range = (0.0, 1.0)

    def calculate_reward(self, source: np.ndarray, target: np.ndarray):
        """
        Computes the euclidean distance between two feature vectors

        Normalizes output so that the closer distance gets higher reward
        """
        euclidean_distance = np.linalg.norm(source - target)
        reward = 1.0 / (1.0 + euclidean_distance)
        assert self.is_in_range(
            reward, self.reward_range), f"Reward {reward} is outside the requested range {self.reward_range}"
        return reward


if __name__ == "__main__":
    s = np.arange(-0.1, 0.3, step=0.1)
    t1 = np.arange(-0.1, 0.3, step=0.1) + 0.1
    t2 = np.arange(-0.1, 0.3, step=0.1) + 0.01  # closer than t1
    metric = EuclideanDistance()
    reward1 = metric.calculate_reward(s, t1)
    reward2 = metric.calculate_reward(s, t2)
    reward3 = metric.calculate_reward(s, s)
    assert reward1 < reward2
    assert reward3 == 1.0
