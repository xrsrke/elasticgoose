from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class HostInfo:
    name: str
    slots: int


class NodeDiscovery(ABC):
    @abstractmethod
    def discover(self):
        raise NotImplementedError("Honk honk! This isn't implemented yet.")


class NodeDiscoveryScript(NodeDiscovery):
    def __init__(self, script: str):
        self.script = script

    def discover(self):
        pass
