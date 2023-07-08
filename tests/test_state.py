from elasticgoose.state import ModelStateHandler, get_handlers

from torch import nn


def test_get_handlers():
    """Test that the ModelStateHandler works as expected."""
    model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
    states = {"model": model, "epoch": 0}

    handlers, remainders = get_handlers(states)

    assert len(handlers) == 1
    assert len(remainders) == 1

    assert isinstance(handlers["model"], ModelStateHandler)
    assert remainders["epoch"] == 0
