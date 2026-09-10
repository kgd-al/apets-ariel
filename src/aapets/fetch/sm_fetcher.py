import logging

import numpy as np
from mujoco import mju_rotVecQuat

from ..common.controllers.ABCpg import ABCpg
from ..common.mujoco.state import MjState
from .types import FetchTaskObjects, NewBodyParts


def cross2d(lhs, rhs):
    return lhs[..., 0] * rhs[..., 1] - lhs[..., 1] * rhs[..., 0]


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
            name: str,
            state: MjState,
            field_of_vision: float = 62.2,
            scaling_power: float = 1,
            **kwargs,
    ):
        super().__init__(
            *args,
            state=state,
            scaling_power=scaling_power,
            name=name.split("_")[0],
            **kwargs,
        )

        self._body = state.data.body(name)
        self.__ball = state.data.body(FetchTaskObjects.BALL)
        self._targets = [self.__ball]

        try:
            self.__human = state.data.body(FetchTaskObjects.HAND)
            self._targets.append(self.__human)
        except KeyError as e:
            logging.debug("Error was", e)
            self.__human = None

        self.__mouth = (
            state.data.actuator(NewBodyParts.MOUTH_SUCKER),
            state.data.sensor(NewBodyParts.MOUTH_SENSOR)
        )
        self.__mouth_off = 0

        self.__eyes = [
            g for i in range(state.model.ngeom)
            if NewBodyParts.SPIDER_EYES in (g := state.model.geom(i)).name
        ]

        self.__mj_state = state
        self.__time = state.time

        self.__beta_scaling_factor = .5  # Optimal distance
        self.__backtracking = 0  # Backtracking counter for proper deference

        self._fwd, self._tgt = np.array([0., 0., 0.]), np.array([0., 0., 0.])
        self._angle = None

        self.half_vision = np.deg2rad(field_of_vision) / 2

        self.overwritten = False

        self.compute_state()

    def reset(self, state: MjState):
        s = super().reset(state)
        self.__time = state.time
        return s

    @classmethod
    def name(cls): return "fetcher"

    @property
    def body(self): return self._body

    @property
    def target(self): return self._targets[self.target_idx]

    @property
    def target_idx(self): return int(self.has_ball) if self.__human is not None else 0

    @property
    def has_ball(self): return bool(self.__mouth[1].data[0])

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

    def release_ball(self, duration=0):
        self.__mouth[0].ctrl[:] = 0
        self.__mouth_off = duration

    def __beta_scaling(self, d: float):
        return 1 - 2 / (1 + np.exp((10 / self.__beta_scaling_factor) * (d - self.__beta_scaling_factor)))

    def compute_state(self):
        dt = (self.__mj_state.time - self.__time)
        self.__time = self.__mj_state.time

        if self.__mouth_off > 0:
            self.__mouth_off -= dt
        elif self.__mouth[1].data[0] and self.__mouth[0].ctrl[0] == 0:
            self.__mouth[0].ctrl[0] = 1

        mju_rotVecQuat(self._fwd, np.array([1., 0., 0.]), self.body.xquat)
        tgt = (self.target.xpos[:2] - self.body.xpos[:2])
        length = np.linalg.norm(tgt)
        if length != 0:
            self._tgt[:2] = tgt / length

        if length < 0.1:
            self.__backtracking = 0.5

        self._angle = np.arccos(np.clip(np.dot(self._fwd[:2], self._tgt[:2]), -1.0, 1.0))
        if cross2d(self._fwd[:2], self._tgt[:2]) < 0:
            self._angle *= -1

        if not self.overwritten:
            self._alpha = np.clip(self._angle / self.half_vision, -1, 1)

            if self.target_idx > 0:  # human
                self._beta = self.__beta_scaling(length)
            else:
                if self.__backtracking > 0:
                    self._beta = -1
                    self.__backtracking -= dt
                else: # Too close: backtrack!
                    self._beta = 1

        if self._beta <= 0:  # Back-pedalling
            eye_color = [0, 0, 1, 1]
        elif self.has_ball:  # Happy
            eye_color = [0, 1, 0, 1]
        elif self.is_target_visible:  # Tracking
            eye_color = [1, 1, 0, 1]
        else:  # Sad
            eye_color = [1, 0, 0, 1]
        for eye in self.__eyes:
            eye.rgba = eye_color

    def _set_actuators_states(self):
        self.compute_state()
        super()._set_actuators_states()
