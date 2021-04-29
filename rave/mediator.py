import queue
from threading import Thread

from rave.osc_server import OscServer
from rave.osc_client import OscClient
from rave.tools import sec_per_k
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

    QUEUE_SIZE = 1
    MAX_K_DIFF = (
        5  # Max amount of KSMPS between timestamp of source and target features
    )

    def __init__(self, run=True, monitor=False):
        self.osc_server = OscServer(ip_adress=OSC_ADDRESS, port=OSC_FEATURE_PORT)
        self.osc_client = OscClient(ip_adress=OSC_ADDRESS, port=OSC_MAPPING_PORT)
        self.source_q = queue.LifoQueue(maxsize=self.QUEUE_SIZE)
        self.target_q = queue.LifoQueue(maxsize=self.QUEUE_SIZE)

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
        s_time = source[0]
        t_time = target[0]
        diff = abs(s_time - t_time)
        threshold = sec_per_k() * self.MAX_K_DIFF
        if diff > threshold:
            return None, None
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
    mediator = Mediator()
    i = 0
    diffs = []
    while i < 10000:
        s, t = mediator.get_features()
        if s and t:
            diff = abs(s[0] - t[0])
            diffs.append(diff)
            i += 1
    mediator.terminate()
    import pdb

    pdb.set_trace()

    # source = np.array(
    #     [
    #         (
    #             1619772022.1248074,
    #             0.21896128356456757,
    #             0.0784977525472641,
    #             0.7613061666488647,
    #             0.6819378733634949,
    #             1.0,
    #             0.8089755177497864,
    #         ),
    #         (
    #             1619772022.05805,
    #             0.22176681458950043,
    #             0.10552229732275009,
    #             0.7885875701904297,
    #             0.7171474099159241,
    #             1.0,
    #             0.7252354025840759,
    #         ),
    #         (
    #             1619772022.0551474,
    #             0.21750089526176453,
    #             0.10548053681850433,
    #             0.8131749033927917,
    #             0.7365936040878296,
    #             1.0,
    #             0.692665696144104,
    #         ),
    #         (
    #             1619772022.052245,
    #             0.2228269875049591,
    #             0.10538239777088165,
    #             0.8067473769187927,
    #             0.7134986519813538,
    #             1.0,
    #             0.6537395119667053,
    #         ),
    #         (
    #             1619772022.0493424,
    #             0.21884021162986755,
    #             0.10515178740024567,
    #             0.8099315166473389,
    #             0.6881665587425232,
    #             1.0,
    #             0.7265108823776245,
    #         ),
    #         (
    #             1619772022.04644,
    #             0.22035256028175354,
    #             0.1046098843216896,
    #             0.803701639175415,
    #             0.6795605421066284,
    #             1.0,
    #             0.7186996340751648,
    #         ),
    #         (
    #             1619772022.0435374,
    #             0.22350053489208221,
    #             0.10461147874593735,
    #             0.8029029369354248,
    #             0.7468311786651611,
    #             1.0,
    #             0.760991632938385,
    #         ),
    #         (
    #             1619772022.0406349,
    #             0.22149261832237244,
    #             0.104615218937397,
    #             0.7894060611724854,
    #             0.7351065278053284,
    #             1.0,
    #             0.7539457082748413,
    #         ),
    #         (
    #             1619772022.0377324,
    #             0.21536774933338165,
    #             0.10462401062250137,
    #             0.7960370779037476,
    #             0.6986750960350037,
    #             1.0,
    #             0.7395306825637817,
    #         ),
    #         (
    #             1619772022.0348299,
    #             0.2130621373653412,
    #             0.10464466363191605,
    #             0.7855211496353149,
    #             0.7201325297355652,
    #             1.0,
    #             0.7833290696144104,
    #         ),
    #     ]
    # )

    # target = np.array(
    #     [
    #         (
    #             1619772022.4701133,
    #             0.1142989844083786,
    #             0.19184510409832,
    #             0.8294172883033752,
    #             0.4814106225967407,
    #             0.22753292322158813,
    #             0.7027481198310852,
    #         ),
    #         (
    #             1619772022.403356,
    #             0.0720544308423996,
    #             0.145279198884964,
    #             0.9377073645591736,
    #             0.4352615773677826,
    #             0.30730700492858887,
    #             0.7353980541229248,
    #         ),
    #         (
    #             1619772022.4004536,
    #             0.07418153434991837,
    #             0.14553137123584747,
    #             0.9563738703727722,
    #             0.40971651673316956,
    #             0.30815669894218445,
    #             0.7030508518218994,
    #         ),
    #         (
    #             1619772022.397551,
    #             0.07852686941623688,
    #             0.1455979347229004,
    #             0.9658654928207397,
    #             0.4080480635166168,
    #             0.2789640426635742,
    #             0.6518739461898804,
    #         ),
    #         (
    #             1619772022.3946486,
    #             0.08211104571819305,
    #             0.1457543820142746,
    #             0.9582415819168091,
    #             0.4284486472606659,
    #             0.3117845058441162,
    #             0.7166059017181396,
    #         ),
    #         (
    #             1619772022.391746,
    #             0.08443763852119446,
    #             0.14612197875976562,
    #             0.9674806594848633,
    #             0.3942376375198364,
    #             0.29192814230918884,
    #             0.725587010383606,
    #         ),
    #         (
    #             1619772022.3888435,
    #             0.08618234843015671,
    #             0.14698581397533417,
    #             0.9609271287918091,
    #             0.40236830711364746,
    #             0.29270270466804504,
    #             0.7128711938858032,
    #         ),
    #         (
    #             1619772022.385941,
    #             0.08947855234146118,
    #             0.14640142023563385,
    #             0.9627712965011597,
    #             0.44876018166542053,
    #             0.3385215103626251,
    #             0.6888264417648315,
    #         ),
    #         (
    #             1619772022.3830385,
    #             0.09086105227470398,
    #             0.14502817392349243,
    #             0.9615370035171509,
    #             0.44992756843566895,
    #             0.28781193494796753,
    #             0.6464289426803589,
    #         ),
    #         (
    #             1619772022.380136,
    #             0.09588079154491425,
    #             0.1444154977798462,
    #             0.9709896445274353,
    #             0.4195929169654846,
    #             0.24589166045188904,
    #             0.7149625420570374,
    #         ),
    #     ]
    # )

    # import pdb

    # pdb.set_trace()
