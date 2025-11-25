import abc

from ..mujoco.state import MjState


class Monitor(abc.ABC):
    def __init__(self, frequency=None):
        self.frequency = frequency
        if frequency is not None:
            self.period = 1 / frequency
            self._next_call = 0

        self._value = None

    def start(self, state: MjState): pass
    def stop(self, state: MjState): pass
    def _step(self, state: MjState): pass

    @property
    def value(self): return self._value

    def __call__(self, state: MjState):
        time = state.data.time
        if self.period is None or time < self._next_call:
            return
        else:
            self._next_call = time + self.period

        self._step(state)
