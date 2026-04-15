import math
import sys
from enum import auto, Enum, StrEnum
from pathlib import Path
from typing import List, Dict, Iterable, Optional

import numpy as np
from matplotlib import pyplot as plt
from mujoco import mj_rnePostConstraint

from ._monitor import Monitor
from ..misc.debug import kgd_debug
from ..mujoco.state import MjState


def body_pos(state: MjState, robot_name: str):
    return state.data.body(f"{robot_name}_core").xpos


class RewardRecorder:
    def __init__(self, path):
        self._path = path
        self._data = None

    def start(self, headers: Iterable[str]):
        self._data = {h: [] for h in headers}

    def step(self, data: Dict[str, float]):
        for h, v in data.items():
            self._data[h].append(v)

    def stop(self):
        fig, ax = plt.subplots()
        handles = []

        for h, d in self._data.items():
            if h == "R":
                ax2 = ax.twinx()
                handle, = ax2.plot(d, label=h, color='black', linestyle='dashed',)
                ax2.set_ylabel(h)
            else:
                handle, = ax.plot(d, label=h)
            handles.append(handle)

        ax.legend(handles=handles)
        fig.savefig(self._path, bbox_inches="tight")


class SpeedMonitor(Monitor):
    def __init__(self,
                 robot_name: str,
                 frequency=None,
                 pos_slice=slice(0, 3),
                 signed=False,
                 *args, **kwargs):

        super().__init__(frequency=frequency, *args, **kwargs)
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
    def __init__(self, robot_name: str, stepwise=True, record: Optional[Path] = None, *args, **kwargs):
        frequency = 20 if stepwise else None
        super().__init__(robot_name, frequency=frequency, pos_slice=slice(0, 1), signed=True, *args, **kwargs)
        self.stepwise = stepwise
        if self.stepwise:
            self._prev_pos, self._delta = None, None
        self._recorder = RewardRecorder(record) if record else None

    def start(self, state: MjState):
        super().start(state)
        if self.stepwise:
            self._delta = 0
            self._prev_pos = self._pos0.copy()
        if self._recorder is not None:
            self._recorder.start(["R"])

    def _step(self, state: MjState):
        # print(f"[kgd-debug:XSpeedMonitor:_step] {self._delta=} {self._prev_pos=} {self._pos1=}")
        self._delta = self._compute_delta(self._prev_pos, self._pos1, self._signed, self._slice)
        self._prev_pos = self._pos1.copy()
        if self._recorder is not None:
            self._recorder.step({"R": self._delta})

    def stop(self, state: MjState):
        super().stop(state)
        if self._recorder is not None:
            self._recorder.stop()


class YSpeedMonitor(SpeedMonitor):
    def __init__(self, robot_name: str, *args, **kwargs):
        super().__init__(robot_name, pos_slice=slice(1, 2), signed=True, *args, **kwargs)


class XYSpeedMonitor(SpeedMonitor):
    def __init__(self, robot_name: str, *args, **kwargs):
        super().__init__(robot_name, pos_slice=slice(0, 2), signed=False, *args, **kwargs)


def krb(x, x_, c): return math.exp(c*(x-x_)**2)


class KernelRewardMonitor(Monitor):
    class AtomicRewards(StrEnum):
        Vx = auto()
        Vy = auto()
        Vz = auto()
        z = auto()

    __weights = {
        AtomicRewards.Vx: .625,
        AtomicRewards.Vy: .125,
        AtomicRewards.Vz: .125,
        AtomicRewards.z: .125
    }

    __params = {
        AtomicRewards.Vx: (.5, -25),
        AtomicRewards.Vy: (0, -5),
        AtomicRewards.Vz: (0, -5),
        AtomicRewards.z: (.20, -2e3),
    }

    def __init__(self, robot_name: str, record: Optional[Path] = None, *args, **kwargs):
        super().__init__(frequency=20, *args, **kwargs)
        self._delta, self._value = 0, 0

        self._robot_name = robot_name
        self._prev_pos, self._pos = None, None
        self._prev_time = None
        self._time = None

        self._recorder = RewardRecorder(record) if record else None

    @property
    def value(self): return super().value / self._time if self._time > 0 else super().value

    def start(self, state: MjState):
        super().start(state)
        self._value = 0
        self._prev_pos = None
        self._pos = body_pos(state, self._robot_name)
        self._prev_time, self._time = state.time, state.time

        if self._recorder is not None:
            self._recorder.start(["R"] + [r.name for r in self.AtomicRewards])

    def _step(self, state: MjState):
        dt = state.time - self._prev_time
        self._prev_time = state.time

        if self._prev_pos is not None and dt > 0:
            velocity = (self._pos - self._prev_pos) / dt
        else:
            velocity = [0, 0, 0]
        k = self.AtomicRewards
        k_rewards = {
            r: krb(v, *self.__params[r]) for r, v in {
                k.Vx: velocity[0],
                k.Vy: abs(velocity[1]),
                k.Vz: velocity[2],
                k.z: self._pos[2]
            }.items()
        }
        weighted_rewards = {c.name: self.__weights[c] * float(v) for c, v in k_rewards.items()}

        # print(f"[kgd-debug] {self._prev_pos=}")
        # print(f"[kgd-debug] {self._pos=}")
        # print(f"[kgd-debug] {velocity=}")
        # print(f"[kgd-debug] {k_rewards=}")
        # print(f"[kgd-debug]: ", {f"{self.__weights[c]} * {v}" for c, v in k_rewards.items()})
        self._delta = sum(weighted_rewards.values())
        self._value += self._delta
        self._prev_pos = self._pos.copy()
        self._time = state.time

        if self._recorder is not None:
            self._recorder.step({"R": self._delta} | weighted_rewards)

    def stop(self, state: MjState):
        if self._recorder is not None:
            self._recorder.stop()


class GymRewardMonitor(Monitor):
    class AtomicRewards(StrEnum):
        Fwd = auto()
        Ctrl = auto()
        Cont = auto()

    __weights = {
        AtomicRewards.Fwd: 1,
        AtomicRewards.Ctrl: -5e-1,
        AtomicRewards.Cont: -5e-4,
    }

    def __init__(self, robot_name: str, stepwise: bool = False, record: Optional[Path] = None, *args, **kwargs):
        super().__init__(frequency=20, *args, **kwargs)
        self._delta, self._value = 0, 0

        self._robot_name = robot_name
        self._prev_pos, self._prev_time, self._pos = None, None, None
        self._prev_ctrl = None

        self._recorder = RewardRecorder(record) if record else None

    def start(self, state: MjState):
        self._value = 0
        self._prev_pos = None
        self._pos = body_pos(state, self._robot_name)
        self._prev_ctrl = self.__ctrl(state)

        self._prev_time = state.time

        if self._recorder is not None:
            self._recorder.start(["R"] + [r.name for r in self.AtomicRewards])

    @staticmethod
    def __ctrl(state: MjState):
        return np.clip(2 * state.data.ctrl / np.pi, -1, 1)

    def _step(self, state: MjState):
        mj_rnePostConstraint(state.model, state.data)
        
        dt = state.time - self._prev_time
        self._prev_time = state.time

        if self._prev_pos is not None and dt > 0:
            velocity = (self._pos - self._prev_pos) / dt
        else:
            velocity = [0, 0, 0]

        contact_forces = np.clip(state.data.cfrc_ext, -1, 1)

        ctrl = self.__ctrl(state)
        ctrl_delta = .5 * abs(ctrl - self._prev_ctrl)
        self._prev_ctrl = ctrl.copy()

        # kgd_debug(f"t={state.time} {ctrl=}")
        # kgd_debug(f"               {ctrl_delta=}")

        g = self.AtomicRewards
        g_rewards = {
            g.Fwd: float(velocity[0]),
            g.Ctrl: float(np.sum(np.square(ctrl_delta))),
            g.Cont: float(np.sum(np.square(contact_forces))),
        }
        if not np.isfinite(g_rewards[g.Cont]):
            g_rewards[g.Cont] = 0
        if any(not np.isfinite(r) for r in g_rewards.values()):
            err_msg = f"Invalid values in gym atomic reward {g_rewards}"
            g_rewards = {g.Fwd: -100, g.Ctrl: 100, g.Cont: 100}
            err_msg = f"{err_msg} >> {g_rewards}"
            print(err_msg, file=sys.stderr)
        weighted_rewards = {c.name: self.__weights[c] * float(v) for c, v in g_rewards.items()}
        # print(f"[kgd-debug] -----------------")
        # print(f"[kgd-debug] {self._prev_pos=}")
        # print(f"[kgd-debug] {self._pos=}")
        # print(f"[kgd-debug] {state.data.ctrl=}")
        # print(f"[kgd-debug] {velocity=}")
        # print(f"[kgd-debug] {g_rewards=}")
        # print(f"[kgd-debug]: ", {f"{self.__weights[c]} * {v}" for c, v in g_rewards.items()})
        self._delta = sum(weighted_rewards.values())
        # print(f"[kgd-debug]", f"{self._delta=}", {f"{self.__weights[c]} * {v}" for c, v in g_rewards.items()})
        self._value += self._delta
        self._prev_pos = self._pos.copy()

        if self._recorder is not None:
            self._recorder.step({"R": self._delta} | weighted_rewards)

    def stop(self, state: MjState):
        if self._recorder is not None:
            self._recorder.stop()


class GymAntKernelRewardMonitor(KernelRewardMonitor):
    AtomicRewards = KernelRewardMonitor.AtomicRewards
    __params = {
        AtomicRewards.Vx: (1, -25),
        AtomicRewards.Vy: (0, -5),
        AtomicRewards.Vz: (0, -5),
        AtomicRewards.z: (.35, -2e3),
    }


class GymAntGymRewardMonitor(GymRewardMonitor):
    AtomicRewards = GymRewardMonitor.AtomicRewards
    __weights = {
        AtomicRewards.Fwd: 1,
        AtomicRewards.Ctrl: -5e-1,
        AtomicRewards.Cont: -5e-4,
    }

