import numpy as np
from osc_server import OscServer


class Analyzer:
    def __init__(self):
        self.features = np.zeros(4)  # TODO: create a ring buffer for this one?
        self.osc_server = OscServer()
        self.osc_server.register_handler(
            "/rave/features", self.parse_osc_features)
        self.inc = 1

    def parse_osc_features(self, address, *features):
        """
        Parsing features from OSC messages
        """
        parsed_features = np.fromiter(features, float)
        self.features = np.vstack((self.features, parsed_features))
        self.inc += 1

        if self.inc % 100 == 0:
            print(f"\n{self.inc} iters\n")
            print(np.mean(self.features[-100:], axis=0))


if __name__ == "__main__":
    analyzer = Analyzer()
    analyzer.osc_server.serve()
