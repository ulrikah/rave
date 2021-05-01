import time
import torch
import torch.nn as nn
from rave.sound import Sound
from rave.effect import Effect
from rave.analyser import Analyser


class AbstractTimeable:
    def call(self):
        raise NotImplementedError("Must be implemented in subclass")


class TimeableTest(AbstractTimeable):
    def call(self):
        return 2 ** 2


class ForwardPass(AbstractTimeable):
    def __init__(self, in_size=6, out_size=8):
        self.model = nn.Sequential(
            nn.Linear(in_size, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, out_size),
            nn.Tanh(),
        )
        self.data = torch.randn(in_size)

    def call(self):
        return self.model.forward(self.data)


class EffectAndAnalysis(AbstractTimeable):
    def __init__(self):
        self.analyser = Analyser(["rms", "pitch", "spectral"])
        self.effect = Effect("dist_lpf")
        self.sound = Sound("noise.wav")
        self.sound.prepare_to_render(self.effect, self.analyser)

    def call(self):
        mapping = self.effect.random_mapping()
        self.sound.render(mapping)
        return self.sound.player.get_channels(self.analyser.analysis_features)


def timer(timeable: AbstractTimeable, n_runs: int = 100):
    start = time.perf_counter()
    for _ in range(n_runs):
        timeable.call()
    end = time.perf_counter()
    elapsed_time_mean = (end - start) / n_runs

    return {"start": start, "end": end, "mean": elapsed_time_mean}


if __name__ == "__main__":
    # timeable = TimeableTest()
    # timeable = ForwardPass()
    timeable = EffectAndAnalysis()
    results = []
    for _ in range(3):
        result = timer(timeable, n_runs=10000)
        results.append(result)

    for result in results:
        print(f"{result['mean'] * 1000:.10f} ms")
