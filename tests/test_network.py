import socketserver
from elasticgoose.network import find_port_and_create_a_socket_server


def test_find_port():
    server, port = find_port_and_create_a_socket_server(lambda addr: socketserver.TCPServer(addr, None))

    assert isinstance(server, socketserver.TCPServer)
    assert isinstance(port, int)
