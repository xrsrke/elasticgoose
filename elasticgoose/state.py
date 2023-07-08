from typing import Dict, Union, Tuple, Optional

import torch
from torch import nn
from torch import optim


SpecialState = Union[nn.Module, optim.Optimizer]
NonSpecialState = torch.Tensor

StateType = Union[SpecialState, NonSpecialState]


class StateHandler:
    """Handle syncronization of state across workers. """
    def __init__(self, value: StateType):
        self.value = value

    def save(self):
        """Save the current value to host memory"""
        raise NotImplementedError("Honk honk! This isn't implemented yet.")

    def restore(self):
        """Restore the last commited across workers"""
        raise NotImplementedError("Honk honk! This isn't implemented yet.")

    def sync(self):
        """Syncronize state across workers."""
        raise NotImplementedError("Honk honk! This isn't implemented yet.")

    def set_value(self, value: StateType):
        """Set the current value of the state."""
        self.value = value
        raise NotImplementedError("Honk honk! This isn't implemented yet.")


class ModelStateHandler(StateHandler):
    def __init__(self, model: nn.Module):
        super().__init__(model.state_dict())


_HANDLER_REGISTRY = [
    (nn.Module, ModelStateHandler),
    # optim.Optimizer, OptimizerStateHandler,
]


def get_handler_registry():
    return _HANDLER_REGISTRY


def get_handler(v: StateType) -> Optional[SpecialState]:
    for handler_type, handler_cls in get_handler_registry():
        if isinstance(v, handler_type):
            return handler_cls(v)
    return None


def get_handlers(states: Dict[str, StateType]) -> Tuple[Dict[str, SpecialState], Dict[str, NonSpecialState]]:
    handlers = {}
    remainders = {}

    for key, value in states.items():
        handler = get_handler(value)
        if handler is None:
            remainders[key] = value
        else:
            handlers[key] = handler

    return handlers, remainders
