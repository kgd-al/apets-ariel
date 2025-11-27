from typing import List, Dict

import mujoco
from mujoco import MjModel, MjData

from .state import MjState
from ..config import BaseConfig
from ..controllers.abstract import Controller
from ..monitors import Monitor


class MjcbCallbacks:
    def __init__(
        self,
        state: MjState,
        controllers: List[Controller],
        monitors: Dict[str, Monitor],
        config: BaseConfig,
    ):
        self.state, self.model, self.data = state, state.model, state.data

        self.controllers = controllers
        self._control_step = 1 / config.control_frequency
        self._next_control_step = 0

        self.max_duration = config.duration
        self.monitors = monitors

        self.config = config

    def __enter__(self):
        if self.monitors:
            for m in self.monitors.values():
                m.start(self.state)
            self.monitor()

            mujoco.set_mjcb_passive(self.monitor)
        mujoco.set_mjcb_control(self.control)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.monitor()
            for m in self.monitors.values():
                m.stop(self.state)

        mujoco.set_mjcb_passive(None)
        mujoco.set_mjcb_control(None)

    def monitor(self, model: MjModel = None, data: MjData = None):
        assert (model is None and data is None) or (model == self.model and data == self.data)
        if self.data.time < self.max_duration:
            for m in self.monitors.values():
                m(self.state)

    def control(self, model: MjModel = None, data: MjData = None):
        assert (model is None and data is None) or (model == self.model and data == self.data)
        if self.data.time >= self._next_control_step:
            for c in self.controllers:
                c(self.state)

            self._next_control_step += self._control_step

    @property
    def metrics(self):
        return {name: monitor.value for name, monitor in self.monitors.items()}
