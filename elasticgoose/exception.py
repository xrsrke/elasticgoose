class NodesUpdatedInterupt(RuntimeError):
    """Raised when there're a new nodes added or removed from the cluster.
    This will trigger elastic mode."""


class NodeInternalError(RuntimeError):
    """Raised when a node encounters an internal error.
    This will trigger fault-tolernace mode."""
