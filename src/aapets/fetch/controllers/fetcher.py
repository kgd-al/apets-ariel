import numpy as np
from mujoco import mju_rotVecQuat

from aapets.fetch.controllers.ABCpg import ABCpg
from aapets.common.mujoco.state import MjState


class FetcherCPG(ABCpg):
    def __init__(
            self,
            *args,
            body: str, targets: list[str],
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
        self.__state = 0

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
    def target(self): return self._targets[self.__state]

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

    def compute_state(self):
        mju_rotVecQuat(self._fwd, np.array([1., 0., 0.]), self.body.xquat)
        self._tgt[:2] = (self.target.xpos[:2] - self.body.xpos[:2])
        self._tgt[:2] /= (self._tgt[:2] ** 2).sum() ** .5

        self._angle = np.arccos(np.clip(np.dot(self._fwd[:2], self._tgt[:2]), -1.0, 1.0))
        if np.cross(self._fwd[:2], self._tgt[:2]) < 0:
            self._angle *= -1

        if not self.overwritten:
            self._alpha = np.clip(self._angle / self.half_vision, -1, 1)
            self._beta = 1

    def _set_actuators_states(self):
        self.compute_state()
        super()._set_actuators_states()
