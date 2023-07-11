from elasticgoose.state import ModelStateHandler, get_handlers, State

import torch
from torch import nn


def test_get_handlers():
    """Test that the ModelStateHandler works as expected."""
    model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
    states = {"model": model, "epoch": 0, "processed_idxs": [1, 2, 3]}

    handlers, remainders = get_handlers(states)

    assert len(handlers) == 1
    assert len(remainders) == 2

    assert isinstance(handlers["model"], ModelStateHandler)
    assert remainders.keys() == {"epoch", "processed_idxs"}
    assert remainders["epoch"] == 0
    assert remainders["processed_idxs"] == [1, 2, 3]


def test_model_state_handler():
    model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
    MODEL_WEIGHTS = model.state_dict()

    NEW_MODEL = nn.Sequential(nn.Linear(2, 2))
    NEW_MODEL.load_state_dict({
        "0.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "0.bias": torch.tensor([5.0, 6.0]),
    })

    handler = ModelStateHandler(model)
    handler.set_value(NEW_MODEL)

    # TODO: seems there's a bug after you set a new value
    # you shouldn't store it as a backup unless .commit() is called


def test_init_state():
    model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    state = State(
        model, optim,
        epoch=0, batch=0
    )

    assert state.model == model
    assert state.optim == optim
    assert state.epoch == 0
    assert state.batch == 0


def test_sync_state():
    model = nn.Sequential(nn.Linear(2, 2))
    MODEL_WEIGHTS = model.state_dict().values()
    EPOCH = 2
    BATCH = 5

    NEW_MODEL = nn.Sequential(nn.Linear(2, 2))
    NEW_MODEL.load_state_dict({
        "0.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "0.bias": torch.tensor([5.0, 6.0]),
    })

    # optim = torch.optim.Adam(model.parameters(), lr=0.01)

    state = State(
        model,
        # optim,
        epoch=EPOCH, batch=BATCH
    )
    # state.sync()

    # modify the model and then restore
    model.load_state_dict(NEW_MODEL.state_dict())
    state.batch += 1
    state.epoch += 1

    state.restore()

    for new_weight, orig_weight in zip(model.parameters(), MODEL_WEIGHTS):
        assert torch.allclose(new_weight, orig_weight)
    assert state.epoch == EPOCH
    assert state.batch == BATCH


    # modify the model and then commit
    model.load_state_dict(NEW_MODEL.state_dict())
    state.batch += 1
    state.epoch += 1

    state.commit()
    state.restore()

    for new_weight, orig_weight in zip(model.parameters(), NEW_MODEL.parameters()):
        assert torch.allclose(new_weight, orig_weight)
    assert state.epoch == EPOCH + 1
    assert state.batch == BATCH + 1
