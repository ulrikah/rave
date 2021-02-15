import numpy as np


class AbstractMetric:
    def __init__(self):
        self.reward_range = None

    def calculate_reward(self, source: np.ndarray, target: np.ndarray):
        raise NotImplementedError("Must be implemented in subclass")

    @staticmethod
    def is_in_range(reward: np.ndarray, reward_range: tuple):
        return reward >= reward_range[0] and reward <= reward_range[1]


class AbsoluteValueNorm(AbstractMetric):
    def __init__(self):
        super().__init__()
        self.reward_range = (0.0, 1.0)

    def calculate_reward(self, source: np.ndarray, target: np.ndarray):
        """
        Computes the absolute value (L1) norm between two feature vectors

        Normalizes output so that closer distances result in higher rewards
        """
        l1_norm = np.mean(np.abs(source - target))
        reward = 1.0 / (1.0 + l1_norm)
        return reward


class EuclideanDistance(AbstractMetric):
    def __init__(self):
        super().__init__()
        self.reward_range = (0.0, 1.0)

    def calculate_reward(self, source: np.ndarray, target: np.ndarray):
        """
        Computes the euclidean distance (L2 norm) between two feature vectors

        Normalizes output so that closer distances result in higher rewards
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
    ones = np.ones(4)
    zeros = np.zeros(4)
    for _metric in [EuclideanDistance, AbsoluteValueNorm]:
        metric = _metric()
        r1 = metric.calculate_reward(s, t1)
        r2 = metric.calculate_reward(s, t2)
        r3 = metric.calculate_reward(s, s)
        import pdb
        pdb.set_trace()
        r4 = metric.calculate_reward(ones, zeros)
        print(r1, r2, r3)
        assert r1 < r2
        assert r3 == 1.0
