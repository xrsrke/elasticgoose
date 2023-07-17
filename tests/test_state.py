import torch
from torch import nn

from elasticgoose.state import ModelStateHandler, State, get_handlers


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


def test_commit_and_restore_model_state_handler():
    model = nn.Sequential(nn.Linear(2, 2))
    MODEL_WEIGHTS = model.state_dict().values()

    NEW_MODEL = nn.Sequential(nn.Linear(2, 2))
    NEW_MODEL.load_state_dict(
        {
            "0.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "0.bias": torch.tensor([5.0, 6.0]),
        }
    )

    handler = ModelStateHandler(model)

    assert handler.value == model

    # set a new value, but haven't committed and then restore
    handler.set_value(NEW_MODEL)
    handler.restore()

    for w1, w2 in zip(handler.value.parameters(), MODEL_WEIGHTS):
        assert torch.allclose(w1, w2)

    # set a new value, then commit and restore
    handler.set_value(NEW_MODEL)
    handler.commit()
    handler.restore()

    assert handler.value == NEW_MODEL

    for w1, w2 in zip(handler.value.parameters(), NEW_MODEL.parameters()):
        assert torch.allclose(w1, w2)

    # TODO: test commit and then delete the original object

    # TODO: seems there's a bug after you set a new value
    # you shouldn't store it as a backup unless .commit() is called


def test_init_state():
    model = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    state = State(model, optim, epoch=0, batch=0)

    assert state.model == model
    assert state.optim == optim
    assert state.epoch == 0
    assert state.batch == 0


def test_commit_and_restore_state_single_node():
    model = nn.Sequential(nn.Linear(2, 2))
    MODEL_WEIGHTS = model.state_dict().values()
    EPOCH, BATCH = 2, 5

    NEW_MODEL = nn.Sequential(nn.Linear(2, 2))
    NEW_MODEL.load_state_dict(
        {
            "0.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "0.bias": torch.tensor([5.0, 6.0]),
        }
    )

    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    state = State(model, optim, epoch=EPOCH, batch=BATCH)

    # update the model, but hanve't committed and then restore
    model.load_state_dict(NEW_MODEL.state_dict())
    state.batch += 1
    state.epoch += 1

    state.restore()

    for w1, w2 in zip(model.parameters(), MODEL_WEIGHTS):
        assert torch.allclose(w1, w2)
    assert state.epoch == EPOCH
    assert state.batch == BATCH

    # update the model, then commit and restore
    model.load_state_dict(NEW_MODEL.state_dict())
    state.batch += 1
    state.epoch += 1

    state.commit()
    state.restore()

    for w1, w2 in zip(model.parameters(), NEW_MODEL.parameters()):
        assert torch.allclose(w1, w2)
    assert state.epoch == EPOCH + 1
    assert state.batch == BATCH + 1

    # TODO: update through State.attribute
    # for both special handlers and regular states


def test_sync_state_multi_process():
    pass
