from pythonosc.udp_client import SimpleUDPClient


class OscClient:
    def __init__(self, ip_adress="127.0.0.1", port=4321):
        self.ip_adress = ip_adress
        self.port = port
        self.client = SimpleUDPClient(address=self.ip_adress, port=self.port)
        self.send_message = self.client.send_message


if __name__ == "__main__":
    c = OscClient()
    c.send_message("/rave/mapping", [0.3, 0.1])
