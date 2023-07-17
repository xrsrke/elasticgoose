import random
import socketserver
from typing import Tuple

from elasticgoose.constant import NETWORK_MAX_PORT, NETWORK_MIN_PORT


def find_port_and_create_a_socket_server(server_factory: socketserver.BaseServer) -> Tuple[socketserver.BaseServer, int]:
    """Find a free port and return a server instance bound to it."""
    min_port = NETWORK_MIN_PORT
    max_port = NETWORK_MAX_PORT
    start_port = random.randint(min_port, max_port)

    for port in range(start_port, max_port):
        try:
            addr = ("", port)
            server = server_factory(addr)
            return server, port
        except Exception:
            continue


class BaseService:
    def __init__(self):
        self._server = find_port_and_create_a_socket_server(lambda add: socketserver.TCPServer(add, self._handler))

    def _probe(self, addresses):
        pass

    def _handler(self):
        pass
