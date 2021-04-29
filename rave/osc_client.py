from pythonosc.udp_client import SimpleUDPClient
import time
import numpy as np


class OscClient:
    def __init__(self, ip_adress="127.0.0.1", port=4321):
        self.ip_adress = ip_adress
        self.port = port
        self.client = SimpleUDPClient(address=self.ip_adress, port=self.port)
        self.send_message = self.client.send_message


if __name__ == "__main__":
    c = OscClient(port=1234)
    while True:
        c.send_message("/rave/mapping", np.random.uniform(low=0.5, high=3.0, size=6))
        time.sleep(0.1)
