import abc
from typing import Optional

from ..mujoco.state import MjState


class Monitor(abc.ABC):
    """
    Generic monitor
    """

    frequency: float
    """frequency of call"""
    
    _value: Optional[float]
    """The episodic value for this monitor (if any)"""

    _delta: Optional[float]
    """The value for the last time step (if any)"""

    def __init__(self, frequency=None):
        self.frequency = frequency
        if frequency is not None:
            self.period = 1 / frequency
            self._next_call = 0
            self._i = 0

        self._value = None
        self._delta = None

    def start(self, state: MjState):
        # print(f"[kgd-debug|Monitor>{self.__class__.__name__}] start")
        if self.frequency is not None:
            self._next_call = state.time
            self._i = 0

    def stop(self, state: MjState):
        # print(f"[kgd-debug|Monitor>{self.__class__.__name__}] stop")
        pass

    def _step(self, state: MjState): pass

    @classmethod
    def name(cls): return cls.__name__.lower().replace("monitor", "")

    @property
    def value(self): return self._value

    @property
    def delta(self): return self._delta

    def __call__(self, state: MjState):
        time = state.data.time
        if self.frequency is None or time < self._next_call:
            return
        else:
            self._next_call = time + self.period

        self._step(state)
        # print(f"[kgd-debug|Monitor>{self.__class__.__name__}] step")
        # print(f"[kgd-debug|Monitor>{self.__class__.__name__}] i={self._i} t={time} v={self._value}",
        #       f"d={self._delta}" if hasattr(self, "_delta") else "")

        if self.frequency is not None:
            self._i += 1
