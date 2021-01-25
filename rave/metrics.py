import torch
import torch.linalg


class AbstractMetric:
    def __init__(self):
        self.reward_range = None

    def calculate_reward(self, source: torch.Tensor, target: torch.Tensor):
        raise NotImplementedError("Must be implemented in subclass")

    @staticmethod
    def is_in_range(reward: torch.Tensor, reward_range: tuple):
        return reward >= reward_range[0] and reward <= reward_range[1]


class EuclideanDistance(AbstractMetric):
    def __init__(self):
        super().__init__()
        self.reward_range = (0.0, 1.0)

    def calculate_reward(self, source: torch.Tensor, target: torch.Tensor):
        """
        Computes the euclidean distance between two feature vectors

        Normalizes output so that the closer distance gets higher reward
        """
        euclidean_distance = torch.linalg.norm(source - target)
        reward = 1.0 / (1.0 + euclidean_distance)
        assert self.is_in_range(
            reward, self.reward_range), f"Reward {reward} is outside the requested range {self.reward_range}"
        return reward


if __name__ == "__main__":
    s = torch.arange(-0.1, 0.3, step=0.1)
    t1 = torch.arange(-0.1, 0.3, step=0.1) + 0.1
    t2 = torch.arange(-0.1, 0.3, step=0.1) + 0.01  # closer than t1
    metric = EuclideanDistance()
    reward1 = metric.calculate_reward(s, t1)
    reward2 = metric.calculate_reward(s, t2)
    assert reward1 < reward2
