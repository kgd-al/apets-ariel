from enum import StrEnum
from typing import Tuple, Literal, List

import numpy as np
from mujoco import mju_rotVecQuat

from aapets.fetch.controllers.ABCpg import ABCpg
from aapets.common.mujoco.state import MjState


class FetcherCPG(ABCpg):
    """
    Simple state machine controlling a cpg.

    Has two potential targets: a ball and a human
    When not having the ball (A), the robot goes towards it
    When having the ball (B), the robot goes towards the human
    When too close to the human (C), the robot moves backward
    """

    def __init__(
            self,
            *args,
            body: str, targets: List[str],
            state: MjState,
            field_of_vision: float = 62.2,
            scaling_power: float = 1,
            **kwargs,
    ):
        super().__init__(
            *args,
            state=state,
            scaling_power=scaling_power,
            **kwargs,
        )

        self._body = state.data.body(body)
        self._targets = [state.data.body(target) for target in targets]
        self.__mj_state = state
        self.__target_idx = 0
        self.__has_ball = 0
        self.__beta_scaling_factor = .25  # Optimal distance

        self._fwd, self._tgt = np.array([0., 0., 0.]), np.array([0., 0., 0.])
        self._angle = None

        self.half_vision = np.deg2rad(field_of_vision) / 2

        self.overwritten = False

        self.compute_state()

    @classmethod
    def name(cls): return "fetcher"

    @property
    def body(self): return self._body

    @property
    def target(self): return self._targets[self.__target_idx]

    @property
    def forward(self) -> np.ndarray: return self._fwd

    @property
    def goal(self) -> np.ndarray: return self._tgt

    @property
    def angle(self): return self._angle

    @property
    def is_target_visible(self): return abs(self._angle) < self.half_vision

    def overwrite_modulators(self, alpha, beta):
        self.overwritten = True
        self._alpha, self._beta = np.clip(alpha, -1, 1), np.clip(beta, -1, 1)

    def release_overwrite(self):
        self.overwritten = False

    def set_has_ball(self, ball: bool):
        self.__has_ball = ball
        self.__target_idx = int(ball)

    def __beta_scaling(self, d: float):
        res = 1 - 2 / (1 + np.exp((10 / self.__beta_scaling_factor) * (d - self.__beta_scaling_factor)))
        print(f"beta_scaling({d=}) = {res}")
        return res

    def compute_state(self):
        mju_rotVecQuat(self._fwd, np.array([1., 0., 0.]), self.body.xquat)
        self._tgt[:2] = (self.target.xpos[:2] - self.body.xpos[:2])
        length = (self._tgt[:2] ** 2).sum() ** .5
        self._tgt[:2] /= length

        self._angle = np.arccos(np.clip(np.dot(self._fwd[:2], self._tgt[:2]), -1.0, 1.0))
        if np.cross(self._fwd[:2], self._tgt[:2]) < 0:
            self._angle *= -1

        if not self.overwritten:
            self._alpha = np.clip(self._angle / self.half_vision, -1, 1)

            if self.__target_idx == 1:  # human
                print(length)
                self._beta = self.__beta_scaling(length)
            else:
                self._beta = 1

    def _set_actuators_states(self):
        self.compute_state()
        super()._set_actuators_states()
