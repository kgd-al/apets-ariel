import numpy as np
from mujoco import mju_rotVecQuat

from aapets.common.controllers import RevolveCPG
from aapets.common.mujoco.state import MjState


class ABCpg(RevolveCPG):
    def __init__(
            self,
            *args,
            body: str, targets: list[str],
            state: MjState,
            field_of_vision: float = 62.2,
            scaling_power: float = 7,
            **kwargs
    ):
        super().__init__(*args, state=state, **kwargs)

        self._modulators = [np.sign(self._joints_pos[actuator.name][1]) for actuator in self._actuators]

        self._body = state.data.body(body)
        self._targets = [state.data.body(target) for target in targets]
        self.__mj_state = state
        self.__state = 0

        self._fwd, self._tgt = np.array([0., 0., 0.]), np.array([0., 0., 0.])
        self._angle = None

        self._alpha, self._beta = None, None

        self.half_vision = np.deg2rad(field_of_vision) / 2
        self.scaling_power = scaling_power

        self.overwritten = False

        self.compute_state()

    @classmethod
    def name(cls): return "abcpg"

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
    def alpha(self): return self._alpha

    @property
    def beta(self): return self._beta

    @property
    def is_target_visible(self): return abs(self._angle) < self.half_vision

    def overwrite_modulators(self, alpha, beta):
        self.overwritten = True
        self._alpha, self._beta = np.clip(alpha, -1, 1), np.clip(beta, 0, 1)

    def release_overwrite(self):
        self.overwritten = False

    def compute_state(self):
        mju_rotVecQuat(self._fwd, np.array([1., 0., 0.]), self.body.xquat)
        self._tgt[:2] = (self.target.xpos[:2] - self.body.xpos[:2])
        self._tgt[:2] /= (self._tgt[:2] ** 2).sum() ** .5

        self._angle = np.arccos(np.clip(np.dot(self._fwd[:2], self._tgt[:2]), -1.0, 1.0))
        if np.cross(self._fwd[:2], self._tgt[:2]) < 0:
            self._angle *= -1

    def _set_actuators_states(self):
        self.compute_state()

        print()
        if not self.overwritten:
            print(self._alpha, self.half_vision)
            self._alpha = np.clip(self._angle / self.half_vision, -1, 1)
            self._beta = 1
        print("alpha:", self._alpha, ", beta:", self._beta)

        lateral_scaling = (1 - abs(self._alpha)) ** self.scaling_power
        global_scaling = self._beta ** self.scaling_power
        # print(f"[kgd-debug] ABCpg.global_scaling:"
        #       f" {lateral_scaling} ({(self.half_vision - abs(self.angle)) / self.half_vision}) ^ {self.scaling_power}")

        print(f"{lateral_scaling=}, {global_scaling=}")

        for i, (actuator, ctrl, modulator) in enumerate(zip(self._actuators, self._state, self._modulators)):
            if self._alpha * modulator > 0:  # Same side
                scaling = lateral_scaling * global_scaling
            else:
                scaling = global_scaling
            actuator.ctrl[:] = scaling * ctrl * self._ranges[i]
