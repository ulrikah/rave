import time
from random import random, randint, uniform

from osc4py3.as_eventloop import *
from osc4py3 import oscbuildparse

PORT = 4321
IP_ADDRESS = "127.0.0.1"


if __name__ == "__main__":
    SLEEP_TIME = 0.5
    osc_startup()
    osc_udp_client(IP_ADDRESS, PORT, "client")

    finished = False
    while not finished:
        msg = oscbuildparse.OSCMessage(
            "/rave/features", ",fff", [uniform(0.3, 0.4), randint(80, 120), uniform(0.6, 0.8)])
        osc_send(msg, "client")
        print("Sleeping for", SLEEP_TIME, "second")
        time.sleep(SLEEP_TIME)
        osc_process()

    osc_terminate()
