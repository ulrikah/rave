import numpy as np
import time
from queue import SimpleQueue
from osc_server import OscServer
from threading import Thread


class Mediator:
    def __init__(self, run=True):
        self.osc_server = OscServer()
        self.source_q = SimpleQueue()
        self.target_q = SimpleQueue()

        self.osc_server.register_handler(
            "/rave/source/features", self.add_source_features)
        self.osc_server.register_handler(
            "/rave/target/features", self.add_target_features)
        self.osc_server_thread = None
        if run:
            self.run()

    def add_source_features(self, address, *features):
        self.source_q.put(features)

    def add_target_features(self, address, *features):
        self.target_q.put(features)

    def get_features(self):
        """
        Pops an array of features off both queues and converts them to numpy arrays
        """
        return np.array(self.get_source_features()), np.array(self.get_target_features())

    def get_source_features(self):
        if not self.source_q.empty():
            return self.source_q.get()
        else:
            raise IOError("Source queue doesn't contain any more features")

    def get_target_features(self):
        if not self.target_q.empty():
            return self.target_q.get()
        else:
            raise IOError("Target queue doesn't contain any more features")

    def clear(self):
        for q in [self.source_q, self.target_q]:
            while not q.empty():
                q.get()
            assert q.qsize() == 0

    def run(self):
        self.osc_server_thread = Thread(target=self.osc_server.serve)
        self.osc_server_thread.start()

    def terminate(self):
        if self.osc_server_thread is not None:
            self.osc_server.terminate()
            self.osc_server_thread.join()
            self.osc_server_thread = None
        else:
            raise Exception("Attempting to terminate thread before it started")
