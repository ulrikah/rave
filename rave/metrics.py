import numpy as np


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
        "l2": InvertedEuclideanDistance,
        "dissimilarity": EuclideanDistance,
    }[name]()


class AbsoluteValueNorm(AbstractMetric):
    def __init__(self):
        super().__init__()
        self.reward_range = (0.0, 1.0)

    def calculate_reward(self, source: np.ndarray, target: np.ndarray):
        """
        Computes the absolute value (L1) norm between two feature vectors

        Scales output so that closer distances result in higher rewards
        """
        l1_norm = np.mean(np.abs(source - target))
        reward = 1.0 / (1.0 + l1_norm)
        return reward


class InvertedEuclideanDistance(AbstractMetric):
    def __init__(self):
        super().__init__()
        self.reward_range = (0.0, 1.0)

    def calculate_reward(self, source: np.ndarray, target: np.ndarray):
        """
        Computes the euclidean distance (L2 norm) between two feature vectors

        Scales output so that closer distances result in higher rewards
        """
        euclidean_distance = np.linalg.norm(source - target)
        reward = 1.0 / (1.0 + euclidean_distance)
        assert self.is_in_range(
            reward, self.reward_range
        ), f"Reward {reward} is outside the requested range {self.reward_range}"
        return reward


class EuclideanDistance(AbstractMetric):
    def __init__(self):
        super().__init__()

    def calculate_reward(self, source: np.ndarray, target: np.ndarray):
        """
        Computes the euclidean distance (L2 norm) between two feature vectors

        The larger the distance, the higher the reward => optimize for dissimilarity
        """
        euclidean_distance = np.linalg.norm(source - target)
        reward = euclidean_distance
        return reward
