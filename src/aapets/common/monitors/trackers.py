from abc import ABC
from pathlib import Path

import numpy as np
import pandas as pd

from aapets.common.controllers.abstract import Controller
from aapets.common.monitors import Monitor
from aapets.common.mujoco.state import MjState


class Tracker(Monitor, ABC):
    def __init__(self, frequency: float, robot_name: str, path: Path, suffix: str = "csv", *args, **kwargs):
        super().__init__(frequency, *args, **kwargs)
        if not suffix[-4:].lower() == ".csv":
            suffix += ".csv"
        if path.is_dir():
            path = path.joinpath(suffix)
        self._path = path
        self._robot_name = robot_name
        self._data = None

    def stop(self, state: MjState):
        pd.DataFrame(self._data).to_csv(self._path, index=False)


class PositionTracker(Tracker):
    def __init__(self, frequency: float, robot_name: str, path: Path, *args, **kwargs):
        super().__init__(frequency, robot_name, path, "positions", *args, **kwargs)

        self._bodies = None

    def start(self, state: MjState):
        super().start(state)

        self._bodies = state.get(
            state.get_names(self._robot_name, "body"), "body", "data")
        self._data = {
            b.name + "-" + d: [] for b in self._bodies for d in "xyzRPY"
        }

    def _step(self, state: MjState):
        for b in self._bodies:
            for i, d in enumerate("xyz"):
                self._data[b.name + "-" + d].append(b.xpos[i])
            for v, d in zip(self._euler(b.xquat), "RPY"):
                self._data[b.name + "-" + d].append(np.degrees(v))

    @staticmethod
    def _euler(quat):
        w, x, y, z = quat
        roll = np.atan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
        pitch = 2 * np.atan2(np.sqrt(1 + 2 * (w * y - x * z)),
                             np.sqrt(1 - 2 * (w * y - x * z))) - np.pi / 2
        yaw = np.atan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2)) - np.pi / 2
        return roll, pitch, yaw


class JointsTracker(Tracker):
    def __init__(self, frequency: float, robot_name: str, path: Path, *args, **kwargs):
        super().__init__(frequency, robot_name, path, "joints", *args, **kwargs)
        self._joints, self._actuators = None, None

    def start(self, state: MjState):
        super().start(state)

        self._joints = Controller.joints(state, self._robot_name, "data")
        self._actuators = Controller.actuators(state, self._robot_name, "data")
        self._data = {
            j.name + "-pos": [] for j in self._joints
        } | {
            a.name + "-ctrl": [] for a in self._actuators
        }

    def _step(self, state: MjState):
        for j, a in zip(self._joints, self._actuators):
            self._data[j.name + "-pos"].append(j.qpos[0].copy())
            self._data[a.name + "-ctrl"].append(a.ctrl[0].copy())
