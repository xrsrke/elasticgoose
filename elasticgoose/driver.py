from threading import Thread

from elasticgoose.discovery import NodeDiscovery


class ElasticDriver:
    def __init__(self, discovery_script: NodeDiscovery):
        self._discovery_script = discovery_script
        self._discovery_thread = Thread(target=self._discovery_script.discover, daemon=True)
