from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ._monitor import Monitor
from ..mujoco.state import MjState


class SpeedMonitor(Monitor):
    def __init__(self, name: str, pos_slice=slice(0, 3)):
        super().__init__(frequency=None)
        self.name = name
        self._pos0, self._pos1 = None, None
        self._slice = pos_slice

    def start(self, state: MjState):
        self._pos1 = state.data.body(f"{self.name}_core").xpos
        self._pos0 = self._pos1.copy()

    def stop(self, state: MjState):
        self._value = np.sqrt(sum(v**2 for v in (self._pos1 - self._pos0)[self._slice]))
        self._value /= state.data.time

    @property
    def current_position(self): return self._pos1

class XSpeedMonitor(SpeedMonitor):
    def __init__(self, name: str):
        super().__init__(name, slice(0, 1))


class YSpeedMonitor(SpeedMonitor):
    def __init__(self, name: str):
        super().__init__(name, slice(1, 2))


class XYSpeedMonitor(SpeedMonitor):
    def __init__(self, name: str):
        super().__init__(name, slice(0, 2))

