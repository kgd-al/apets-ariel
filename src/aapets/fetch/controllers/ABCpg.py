from typing import Optional, Sequence

import numpy as np
from mujoco import mju_rotVecQuat

from aapets.common.controllers import RevolveCPG
from aapets.common.mujoco.state import MjState


class ABCpg(RevolveCPG):
    def __init__(
            self,
            *args,
            state: MjState,
            scaling_power: float = 1,
            **kwargs
    ):
        super().__init__(*args, state=state, **kwargs)

        self._alpha, self._beta = None, None
        self.scaling_power = scaling_power

        self._sides = [
            np.sign(self._joints_pos[actuator.name][1]) for actuator in self._actuators
        ]

        self._verticals = [
            np.allclose(state.data.joint(name).xaxis, [0, 0, 1])
            for name in self._mapping.keys()
        ]

    @classmethod
    def name(cls): return "abcpg"

    @property
    def alpha(self): return self._alpha

    @property
    def beta(self): return self._beta

    def _set_actuators_states(self):
        lateral_scaling = (1 - abs(self._alpha)) ** self.scaling_power
        global_scaling = abs(self._beta) ** self.scaling_power

        forward = np.sign(self._beta)

        # print(f"{lateral_scaling=}, {forward_scaling=}, {forward=}")

        for i, (actuator, ctrl, side, vertical) in enumerate(zip(
                self._actuators, self._state, self._sides, self._verticals)):

            scaling = global_scaling

            # print(actuator.name, "initial scaling", scaling)
            if self._alpha * side * forward > 0:  # Same side
                scaling *= lateral_scaling
                # print(">>", scaling, f" * {lateral_scaling=}")

            if vertical and self._beta < 0:
                scaling *= -1
                # print(">>", scaling, f" * {forward_scaling=}")

            # scaling = 1 if vertical else -1
            # print(">",  scaling)

            actuator.ctrl[:] = scaling * ctrl * self._ranges[i]
