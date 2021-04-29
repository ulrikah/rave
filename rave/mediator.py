import queue
from threading import Thread

from rave.osc_server import OscServer
from rave.osc_client import OscClient
from rave.constants import (
    OSC_ADDRESS,
    OSC_FEATURE_PORT,
    OSC_MAPPING_PORT,
    OSC_MAPPING_ROUTE,
    OSC_SOURCE_FEATURES_ROUTE,
    OSC_TARGET_FEATURES_ROUTE,
    DEBUG_SUFFIX,
)


class Mediator:
    def __init__(self, run=True, monitor=False):
        self.osc_server = OscServer(ip_adress=OSC_ADDRESS, port=OSC_FEATURE_PORT)
        self.osc_client = OscClient(ip_adress=OSC_ADDRESS, port=OSC_MAPPING_PORT)
        self.source_q = queue.LifoQueue(maxsize=1)
        self.target_q = queue.LifoQueue(maxsize=1)

        self.osc_server.register_handler(
            OSC_SOURCE_FEATURES_ROUTE, self.add_source_features
        )
        self.osc_server.register_handler(
            OSC_TARGET_FEATURES_ROUTE, self.add_target_features
        )
        self.osc_server_thread = None
        self.monitor = monitor
        if self.monitor:

            def log_handler(address, *args):
                print("[UNHANDLED]", address, *args)

            self.osc_server.register_default_handler(log_handler)

        # placeholders for future values
        self.source = None
        self.target = None

        if run:
            self.run()

    def add_source_features(self, address, *features):
        if self.monitor:
            print(address, *features)
        try:
            self.source_q.put(features, block=False)
        except queue.Full:
            _ = self.source_q.get()
            self.source_q.put(features, block=False)

    def add_target_features(self, address, *features):
        if self.monitor:
            print(address, *features)
        try:
            self.target_q.put(features, block=False)
        except queue.Full:
            _ = self.target_q.get()
            self.target_q.put(features, block=False)

    def get_features(self):
        """
        Pops an array of features off both queues and converts them to numpy arrays

        NB: this does not pop features of the stack if only one of the queues have
        new entries. For those cases, None is returned
        """

        source = self.get_source_features()
        target = self.get_target_features()
        if source is None or target is None:
            return None, None
        else:
            return source, target

    def get_source_features(self, blocking=True):
        try:
            return self.source_q.get(block=blocking)
        except queue.Empty:
            return None

    def get_target_features(self, blocking=True):
        try:
            return self.target_q.get(block=blocking)
        except queue.Empty:
            return None

    def send_effect_mapping(self, mapping: dict):
        self.osc_client.send_message(
            OSC_MAPPING_ROUTE,
            [
                value
                for (key, value) in mapping.items()
                if not key.endswith(DEBUG_SUFFIX)
            ],
        )

    def clear(self, q):
        "Empties a queue"
        while not q.empty():
            q.get(block=False)
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


if __name__ == "__main__":
    mediator = Mediator(monitor=True)
