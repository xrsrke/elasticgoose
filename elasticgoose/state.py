from typing import Dict, Union, Tuple, Optional
from abc import ABC, abstractmethod
import copy

import torch
from torch import nn
from torch import optim


SpecialState = Union[nn.Module, optim.Optimizer]
NonSpecialState = torch.Tensor

StateType = Union[SpecialState, NonSpecialState]


class StateHandler(ABC):
    """Handle syncronization of state across workers."""
    def __init__(self, value: StateType):
        self.value = value

    @abstractmethod
    def save(self):
        """Save the current value to host memory."""
        raise NotImplementedError("Honk honk! This isn't implemented yet.")

    @abstractmethod
    def restore(self):
        """Restore the last commited across workers."""
        raise NotImplementedError("Honk honk! This isn't implemented yet.")

    @abstractmethod
    def sync(self):
        """Syncronize state across workers."""
        raise NotImplementedError("Honk honk! This isn't implemented yet.")

    def set_value(self, value: StateType):
        """Set the current value of the state."""
        self.value = value
        self.save()


class ModelStateHandler(StateHandler):
    """Handle syncronization of model state across workers."""
    def __init__(self, model: nn.Module):
        super().__init__(value=model)

    def save(self):
        self._model_state = copy.deepcopy(self.value.state_dict())

    def restore(self):
        self.value = self.value.load_state_dict(self._model_state)

    def sync(self):
        pass


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
