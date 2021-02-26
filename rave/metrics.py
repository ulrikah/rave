import numpy as np


def scale(v, a, b, c, d):
    """
    Scales a value v in range [a, b] to range [c, d]
    """
    return (v - a) * (d - c) / (b - a) + c


class AbstractMetric:
    def __init__(self):
        self.reward_range = None

    def calculate_reward(self, source: np.ndarray, target: np.ndarray):
        raise NotImplementedError("Must be implemented in subclass")

    @staticmethod
    def is_in_range(reward: np.ndarray, reward_range: tuple):
        return reward >= reward_range[0] and reward <= reward_range[1]


def metric_from_name(name: str) -> AbstractMetric:
    # raises a KeyError if no corresponding metric is found
    return {
        "l1": AbsoluteValueNorm,
        "l2": EuclideanDistance,
        "l2_normalized": NormalizedEuclidean,
    }[name]()


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


class NormalizedEuclidean(AbstractMetric):
    def __init__(self):
        super().__init__()
        self.reward_range = (0.0, 1.0)

    def normalize(self, reward, feature_length: int):
        # TODO: only do this once (problem is if features change)
        one = np.ones(feature_length)
        zero = np.zeros(feature_length)
        a = 1.0 / (1.0 + np.linalg.norm(one - zero))
        b = 1.0 / (1.0 + np.linalg.norm(one - one))
        c, d = self.reward_range
        return scale(reward, a, b, c, d)

    def calculate_reward(self, source: np.ndarray, target: np.ndarray):
        """
        Computes the euclidean distance (L2 norm) between two feature vectors

        Normalizes output so that closer distances result in higher rewards, and
        so that the lower and higher bounds are truly (0.0, 1.0)
        """
        euclidean_distance = np.linalg.norm(source - target)
        reward = 1.0 / (1.0 + euclidean_distance)
        scaled_reward = self.normalize(reward, source.size)
        assert self.is_in_range(
            scaled_reward, self.reward_range
        ), f"Reward {scaled_reward} is outside the requested range {self.reward_range}"
        return scaled_reward


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
            reward, self.reward_range
        ), f"Reward {reward} is outside the requested range {self.reward_range}"
        return reward


if __name__ == "__main__":
    s = np.arange(-0.1, 0.3, step=0.1)
    t1 = np.arange(-0.1, 0.3, step=0.1) + 0.1
    t2 = np.arange(-0.1, 0.3, step=0.1) + 0.01  # closer than t1
    ones = np.ones(4)
    zeros = np.zeros(4)
    for _metric in [NormalizedEuclidean]:
        metric = _metric()
        r1 = metric.calculate_reward(s, t1)
        r2 = metric.calculate_reward(s, t2)
        r3 = metric.calculate_reward(s, s)
        r4 = metric.calculate_reward(ones, zeros)
        r5 = metric.calculate_reward(zeros, zeros)
        r6 = metric.calculate_reward(ones, ones)
        print(r1, r2, r3)
        assert r1 < r2
        assert r3 == 1.0
        assert r4 == 0.0, r4
        assert r5 == r6 == 1.0
