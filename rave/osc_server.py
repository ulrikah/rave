import time
from types import FunctionType

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer


class OscServer:
    def __init__(self, ip_adress="127.0.0.1", port=4321):
        self.ip_adress = ip_adress
        self.port = port

        self.dispatcher = Dispatcher()
        self.server = None

    def register_handler(
        self, route: str, handler_function: FunctionType, default=False
    ):
        if default:
            self.dispatcher.set_default_handler(handler_function)
        else:
            self.dispatcher.map(route, handler_function)

    def terminate(self):
        self.server.shutdown()

    def serve(self):
        print(f"Listening for OSC messages on {self.ip_adress}:{self.port}")
        self.server = BlockingOSCUDPServer((self.ip_adress, self.port), self.dispatcher)
        self.server.serve_forever()


if __name__ == "__main__":

    def log_handler(address, *args):
        print("[UNHANDLED ADDRESS]", address, *args)

    def features_handler(address, *features):
        rms, pitch, centroid, flux = features
        print(
            f"{address} | rms {rms} | pitch {pitch} | centroid {centroid} | flux {flux}"
        )

    server = OscServer()
    server.register_handler("/rave/features", features_handler)
    server.serve()
