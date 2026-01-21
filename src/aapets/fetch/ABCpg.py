import numpy as np
from mujoco import mju_rotVecQuat

from aapets.common.controllers import RevolveCPG
from aapets.common.mujoco.state import MjState


class ABCpg(RevolveCPG):
    def __init__(
            self,
            *args,
            body: str, target: str,
            state: MjState,
            field_of_vision: float = 62.2,
            scaling_power: float = 7,
            **kwargs
    ):
        super().__init__(*args, state=state, **kwargs)

        self._modulators = [np.sign(self._joints_pos[actuator.name][1]) for actuator in self._actuators]

        self._body = state.data.body(body)
        self._target = state.data.body(target)
        self.__mj_state = state

        self._fwd, self._tgt = np.array([0., 0., 0.]), np.array([0., 0., 0.])
        self._angle = None

        self.half_vision = field_of_vision / 2
        self.scaling_power = scaling_power

        self.compute_state()

    @property
    def body(self): return self._body

    @property
    def target(self): return self._target

    @property
    def forward(self) -> np.ndarray: return self._fwd

    @property
    def goal(self) -> np.ndarray: return self._tgt

    @property
    def angle(self): return self._angle

    @property
    def is_target_visible(self): return abs(self._angle) < self.half_vision

    def compute_state(self):
        mju_rotVecQuat(self._fwd, np.array([1., 0., 0.]), self.body.xquat)
        self._tgt[:2] = (self.target.xpos[:2] - self.body.xpos[:2])
        self._tgt[:2] /= (self._tgt[:2] ** 2).sum() ** .5

        self._angle = np.arccos(np.clip(np.dot(self._fwd[:2], self._tgt[:2]), -1.0, 1.0))
        if np.cross(self._fwd[:2], self._tgt[:2]) < 0:
            self._angle *= -1

    def _set_actuators_states(self):
        self.compute_state()

        global_scaling = ((self.half_vision - abs(self._angle)) / self.half_vision) ** self.scaling_power

        for i, (actuator, ctrl, modulator) in enumerate(zip(self._actuators, self._state, self._modulators)):
            if self._angle * modulator > 0:  # Same side
                local_scaling = global_scaling
            else:
                local_scaling = 1
            actuator.ctrl[:] = local_scaling * ctrl * self._ranges[i]
