import time

from osc4py3.as_eventloop import *
from osc4py3 import oscmethod as osm

IP_ADDRESS = "127.0.0.1"
PORT = 4321
ARGSCHEME = osm.OSCARG_ADDRESS + osm.OSCARG_DATAUNPACK


def log_handler(address, *args):
    print("[UNHANDLED ADDRESS]", address, *args)


def features_handler(address, *features):
    print(address)
    rms, pitch, centroid, flux = features


if __name__ == "__main__":
    SLEEP_TIME = 1
    osc_startup()
    osc_udp_server(IP_ADDRESS, PORT, "listener")
    # osc_method("/rave/*", log_handler, argscheme=ARGSCHEME)
    osc_method("/rave/features", features_handler, argscheme=ARGSCHEME)

    finished = False
    while not finished:
        try:
            # print("Sleeping for", SLEEP_TIME, "second")
            # time.sleep(SLEEP_TIME)
            osc_process()

        except KeyboardInterrupt:
            print("\nClosing OSC server.")
            osc_terminate()
            finished = True
