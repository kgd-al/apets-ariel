import numpy as np

from ._monitor import Monitor
from ..mujoco.state import MjState


class SpeedMonitor(Monitor):
    def __init__(self, robot_name: str, *, frequency=None, pos_slice=slice(0, 3), signed=False):
        super().__init__(frequency=frequency)
        self.robot_name = robot_name
        self._pos0, self._pos1 = None, None
        self._slice = pos_slice
        self._signed = signed

    def start(self, state: MjState):
        self._pos1 = state.data.body(f"{self.robot_name}_core").xpos
        self._pos0 = self._pos1.copy()

    def stop(self, state: MjState):
        if (t := state.data.time) > 0:
            self._value = float(self._compute_delta(self._pos0, self._pos1, self._signed, self._slice) / t)

        else:
            self._value = 0

    @staticmethod
    def _compute_delta(pos0, pos1, _signed: bool, _slice: slice):
        if _signed:
            return sum(v for v in (pos1 - pos0)[_slice])
        else:
            return np.sqrt(sum(v ** 2 for v in (pos1 - pos0)[_slice]))

    @property
    def current_position(self): return self._pos1


class XSpeedMonitor(SpeedMonitor):
    def __init__(self, robot_name: str, stepwise: bool = False):
        frequency = 1000 if stepwise else None
        super().__init__(robot_name, frequency=frequency, pos_slice=slice(0, 1), signed=True)
        self.stepwise = stepwise
        if self.stepwise:
            self._prev_pos, self._delta = None, None

    def start(self, state: MjState):
        super().start(state)
        self._prev_pos = self._pos0.copy()

    def _step(self, state: MjState):
        self._delta = self._compute_delta(self._prev_pos, self._pos1, self._signed, self._slice)
        self._prev_pos = self._pos1.copy()

    @property
    def delta(self): return self._delta


class YSpeedMonitor(SpeedMonitor):
    def __init__(self, robot_name: str):
        super().__init__(robot_name, pos_slice=slice(1, 2), signed=True)


class XYSpeedMonitor(SpeedMonitor):
    def __init__(self, robot_name: str):
        super().__init__(robot_name, pos_slice=slice(0, 2), signed=False)

