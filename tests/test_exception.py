from elasticgoose.exception import NodeInternalError, NodesUpdatedInterupt


def test_node_internal_error():
    try:
        raise NodeInternalError()
    except NodeInternalError as e:
        assert isinstance(e, RuntimeError)


def test_nodes_updated_interupt():
    try:
        raise NodesUpdatedInterupt()
    except NodesUpdatedInterupt as e:
        assert isinstance(e, RuntimeError)
