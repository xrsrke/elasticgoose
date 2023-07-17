import socketserver
from elasticgoose.network import find_port


def test_find_port():
    server, port = find_port(lambda addr: socketserver.TCPServer(addr, None))

    assert isinstance(server, socketserver.TCPServer)
    assert isinstance(port, int)
