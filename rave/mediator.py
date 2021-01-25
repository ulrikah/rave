import numpy as np
import time
from queue import SimpleQueue
from osc_server import OscServer
from threading import Thread


class Mediator:
    def __init__(self, run=True):
        self.osc_server = OscServer()
        self.osc_server.register_handler(
            "/rave/features", self.add_features)
        self.q = SimpleQueue()
        self.osc_server_thread = None
        if run:
            self.run()

    def add_features(self, address, *features):
        self.q.put(features)

    def pop_features(self):
        if not self.q.empty():
            return self.q.get()
        else:
            print("Queue doesn't contain any more features")
            raise IOError

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


if __name__ == "__main__":
    print("Starting mediator in separate thread")
    mediator = Mediator()
    mediator.run()
    print("Sleeping for 1 seconds")
    time.sleep(1)
    print("Terminating OSC server thread")
    mediator.terminate()
    print("Current queue size", mediator.q.qsize())
    print("Features", mediator.pop_features())
    print("Current queue size", mediator.q.qsize())
