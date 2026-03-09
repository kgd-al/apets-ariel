import math
from enum import auto, StrEnum, Enum
from pathlib import Path

import numpy as np
import pandas as pd

from ._monitor import Monitor
from ..controllers.abstract import Controller
from ..mujoco.state import MjState


def body_pos(state: MjState, robot_name: str):
    return state.data.body(f"{robot_name}_core").xpos


class SpeedMonitor(Monitor):
    def __init__(self, robot_name: str, *, frequency=None, pos_slice=slice(0, 3), signed=False):
        super().__init__(frequency=frequency)
        self.robot_name = robot_name
        self._pos0, self._pos1 = None, None
        self._slice = pos_slice
        self._signed = signed

    def start(self, state: MjState):
        super().start(state)
        self._pos1 = body_pos(state, self.robot_name)
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
        stepwise = True
        frequency = 20 if stepwise else None
        super().__init__(robot_name, frequency=frequency, pos_slice=slice(0, 1), signed=True)
        self.stepwise = stepwise
        if self.stepwise:
            self._prev_pos, self._delta = None, None

    def start(self, state: MjState):
        super().start(state)
        if self.stepwise:
            self._delta = 0
            self._prev_pos = self._pos0.copy()

    def _step(self, state: MjState):
        self._delta = self._compute_delta(self._prev_pos, self._pos1, self._signed, self._slice)
        self._prev_pos = self._pos1.copy()


class YSpeedMonitor(SpeedMonitor):
    def __init__(self, robot_name: str):
        super().__init__(robot_name, pos_slice=slice(1, 2), signed=True)


class XYSpeedMonitor(SpeedMonitor):
    def __init__(self, robot_name: str):
        super().__init__(robot_name, pos_slice=slice(0, 2), signed=False)


def krb(x, x_, c): return math.exp(c*(x-x_)**2)


class KernelRewardMonitor(Monitor):
    class AtomicRewards(Enum):
        Vx = auto()
        Vy = auto()
        Vz = auto()
        z = auto()

    __weights = {
        AtomicRewards.Vx: .5,
        AtomicRewards.Vy: .25,
        AtomicRewards.Vz: .125,
        AtomicRewards.z: .125
    }

    def __init__(self, robot_name: str, **kwargs):
        super().__init__(frequency=20)
        self._delta, self._value = 0, 0

        self._robot_name = robot_name
        self._prev_pos, self._pos = None, None
        self._original_height = None

    def start(self, state: MjState):
        super().start(state)
        self._value = 0
        self._prev_pos = None
        self._pos = body_pos(state, self._robot_name)
        self._original_height = self._pos[2].copy()

    def _step(self, state: MjState):
        if self._prev_pos is not None:
            velocity = self._pos - self._prev_pos
        else:
            velocity = [0, 0, 0]
        k = self.AtomicRewards
        k_rewards = {
            k.Vx: krb(velocity[0], .5, -25),
            k.Vy: krb(abs(velocity[1]), 0, -5),
            k.Vz: krb(velocity[2], 0, -5),
            k.z: krb(self._pos[2] - self._original_height, .05, -2e3)
        }
        # print(f"[kgd-debug] {self._prev_pos=}")
        # print(f"[kgd-debug] {self._pos=}")
        # print(f"[kgd-debug] {velocity=}")
        # print(f"[kgd-debug] {k_rewards=}")
        # print(f"[kgd-debug]: ", {f"{self.__weights[c]} * {v}" for c, v in k_rewards.items()})
        self._delta = sum(self.__weights[c] * float(v) for c, v in k_rewards.items())
        self._value += self._delta
        self._prev_pos = self._pos.copy()


class GymRewardMonitor(Monitor):
    class AtomicRewards(Enum):
        Fwd = auto()
        Ctrl = auto()
        Cont = auto()

    __weights = {
        AtomicRewards.Fwd: 1,
        AtomicRewards.Ctrl: -5e-2,
        AtomicRewards.Cont: -5e-3,
    }

    def __init__(self, robot_name: str, stepwise: bool = False):
        super().__init__(frequency=20)
        self._delta, self._value = 0, 0

        self._robot_name = robot_name
        self._prev_pos, self._pos = None, None

    def start(self, state: MjState):
        self._value = 0
        self._prev_pos = None
        self._pos = body_pos(state, self._robot_name)

    def _step(self, state: MjState):
        if self._prev_pos is not None:
            velocity = self._pos - self._prev_pos
        else:
            velocity = [0, 0, 0]

        contact_forces = np.clip(state.data.cfrc_ext, -1, 1)

        g = self.AtomicRewards
        g_rewards = {
            g.Fwd: velocity[0],
            g.Ctrl: np.sum(state.data.ctrl),
            g.Cont: np.sum(np.square(contact_forces))
        }
        # print(f"[kgd-debug] {self._prev_pos=}")
        # print(f"[kgd-debug] {self._pos=}")
        # print(f"[kgd-debug] {velocity=}")
        # print(f"[kgd-debug] {g_rewards=}")
        # print(f"[kgd-debug]: ", {f"{self.__weights[c]} * {v}" for c, v in g_rewards.items()})
        self._delta = sum(self.__weights[c] * float(v) for c, v in g_rewards.items())
        self._value += self._delta
        self._prev_pos = self._pos.copy()

