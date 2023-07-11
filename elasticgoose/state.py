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
    def commit(self):
        """Save the current value for backup."""
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
        self.commit()


class ModelStateHandler(StateHandler):
    """Handle syncronization of model state across workers."""
    def __init__(self, model: nn.Module):
        super().__init__(value=model)

        # store this as an initial commit
        self._model_state = copy.deepcopy(self.value.state_dict())

    def commit(self):
        self._model_state = copy.deepcopy(self.value.state_dict())

    def restore(self):
        self.value.load_state_dict(self._model_state)

    def sync(self):
        pass


class OptimizerStateHandler(StateHandler):
    def __init__(self, optim: optim.Optimizer):
        super().__init__(value=optim)

    def commit(self):
        pass

    def restore(self):
        pass

    def sync(self):
        pass


_HANDLER_REGISTRY = [
    (nn.Module, ModelStateHandler),
    (optim.Optimizer, OptimizerStateHandler)
]


def get_handler_registry():
    return _HANDLER_REGISTRY


def _get_handler(v: StateType) -> Optional[SpecialState]:
    for handler_type, handler_cls in get_handler_registry():
        if isinstance(v, handler_type):
            return handler_cls(v)
    return None


def get_handlers(states: Dict[str, StateType]) -> Tuple[Dict[str, StateHandler], Dict[str, NonSpecialState]]:
    handlers = {}
    remainders = {}

    for key, value in states.items():
        handler = _get_handler(value)
        if handler is None:
            remainders[key] = value
        else:
            handlers[key] = handler

    return handlers, remainders


class ObjectState(ABC):
    @staticmethod
    def commit(self):
        """Commit the current value for backup."""
        raise NotImplementedError("Honk honk! This isn't implemented yet.")

    @staticmethod
    def restore(self):
        """Restore the last commited across workers."""
        raise NotImplementedError("Honk honk! This isn't implemented yet.")

    @abstractmethod
    def reset(self):
        """Reset the state to the initial state."""
        raise NotImplementedError("Honk honk! This isn't implemented yet.")

    @abstractmethod
    def sync(self):
        """Syncronize state across workers."""
        raise NotImplementedError("Honk honk! This isn't implemented yet.")


class RegularState(ObjectState):
    """For states that don't require special handlers."""
    def __init__(self, states: Dict[str, NonSpecialState]):
        self._saved_states = states
        for key, value in self._saved_states.items():
            setattr(self, key, value)

    def commit(self):
        new_states = {}
        for key in self._saved_states.keys():
            new_states[key] = getattr(self, key)
        self._saved_states = new_states

    def restore(self):
        for key, value in self._saved_states.items():
            setattr(self, key, value)

    def reset(self):
        pass

    def sync(self):
        pass


class State(RegularState):
    """A wrapper for state that can be synced across workers."""
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        optim: Optional[optim.Optimizer] = None,
        **kwargs
    ):
        kwargs.update({"model": model, "optim": optim})
        handlers, regular_states = get_handlers(kwargs)

        self._handlers: Dict[str, StateHandler] = handlers
        RegularState.__init__(self, states=regular_states)

        for name, handler in self._handlers.items():
            setattr(self, name, handler.value)

    def commit(self):
        """Commit the current value for backup."""
        for handler in self._handlers.values():
            handler.commit()
        RegularState.commit(self)

    def restore(self):
        """Restore the last commited across workers."""
        # restore syncronous states that requires
        # a special handler
        for handler in self._handlers.values():
            handler.restore()
        RegularState.restore(self)

    def reset(self):
        # TODO: implement it
        pass

    def sync(self):
        for handler in self._handlers.values():
            handler.sync()
        RegularState.sync(self)
