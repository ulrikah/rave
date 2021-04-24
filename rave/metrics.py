import numpy as np


class AbstractMetric:
    def __init__(self):
        self.reward_range = None

    def calculate_reward(self, dry: np.ndarray, wet: np.ndarray, target: np.ndarray):
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
        "relative": RelativeGain,
    }[name]()


class AbsoluteValueNorm(AbstractMetric):
    def __init__(self):
        super().__init__()
        self.reward_range = (0.0, 1.0)

    def calculate_reward(self, dry: np.ndarray, wet: np.ndarray, target: np.ndarray):
        """
        Computes the absolute value (L1) norm between wet and target feature vectors

        Scales output so that closer distances result in higher rewards
        """
        l1_norm = np.mean(np.abs(wet - target))
        reward = 1.0 / (1.0 + l1_norm)
        return reward


class InvertedEuclideanDistance(AbstractMetric):
    def __init__(self):
        super().__init__()
        self.reward_range = (0.0, 1.0)

    def calculate_reward(self, dry: np.ndarray, wet: np.ndarray, target: np.ndarray):
        """
        Computes the euclidean distance between wet and target feature vectors

        Scales output so that closer distances result in higher rewards
        """
        euclidean_distance = np.linalg.norm(wet - target)
        reward = 1.0 / (1.0 + euclidean_distance)
        assert self.is_in_range(
            reward, self.reward_range
        ), f"Reward {reward} is outside the requested range {self.reward_range}"
        return reward


class EuclideanDistance(AbstractMetric):
    def __init__(self):
        super().__init__()

    def calculate_reward(self, dry: np.ndarray, wet: np.ndarray, target: np.ndarray):
        """
        Computes the euclidean distance between wet and target feature vectors

        The larger the distance, the higher the reward => optimize for dissimilarity
        """
        euclidean_distance = np.linalg.norm(wet - target)
        reward = euclidean_distance
        return reward


class RelativeGain(AbstractMetric):
    def __init__(self):
        super().__init__()

    def calculate_reward(self, dry: np.ndarray, wet: np.ndarray, target: np.ndarray):
        """
        1. Computes the euclidean distance between dry and target
        2. Computes the euclidean distance between wet and target
        3. Measure how much better wet performed than dry by taking the difference.
            The difference will be a proxy for how much better/worse wet performed than dry
        """
        euc_dry = np.linalg.norm(dry - target)
        euc_wet = np.linalg.norm(wet - target)
        reward = euc_dry - euc_wet
        return reward
