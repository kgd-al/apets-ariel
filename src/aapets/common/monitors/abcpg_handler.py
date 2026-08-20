import numpy as np

from mujoco import mju_rotVecQuat

from ..mujoco.state import MjState
from ..controllers import ABCpg
from ._monitor import MonitorBase


def cross2d(lhs, rhs):
    return lhs[..., 0] * rhs[..., 1] - lhs[..., 1] * rhs[..., 0]


class ABCPGHandler(MonitorBase):
    """ Tracks `target` and provides ab-commands to provided `abcpg` controlling `robot`
    """
    def __init__(
        self,
        controller: ABCpg,
        robot_name: str = "target",
        target_name: str = "target",
        field_of_vision: float = 62.2,
        frequency=20,
        debug=False,
        *args,
        **kwargs,
    ):
        super().__init__(frequency=frequency, *args, **kwargs)
        self.robot_name, self.target_name = robot_name, target_name
        self.controller = controller
        self.robot, self.target = None, None

        self.half_vision = np.deg2rad(field_of_vision) / 2
        self._fwd, self._tgt = np.array([0., 0., 0.]), np.array([0., 0., 0.])

        self.debug = debug

    def start(self, state: MjState):
        super().start(state)
        self.robot = state.data.body(self.robot_name + "_world")
        self.target = state.data.body(self.target_name)
        if False and self.debug:
            print(f"robot0={self.robot.xpos} target0={self.target.xpos} a/b={self.controller.alpha}/{self.controller.beta}")
            with np.printoptions(linewidth=1000, precision=1, threshold=1000000):
                print(f"weights:\n{self.controller._weight_matrix}")

    def _step(self, state: MjState):
        mju_rotVecQuat(self._fwd, np.array([1., 0., 0.]), self.robot.xquat)
        self._tgt[:2] = (self.target.xpos[:2] - self.robot.xpos[:2])
        length = (self._tgt[:2] ** 2).sum() ** .5
        self._tgt[:2] /= length

        self._angle = np.arccos(np.clip(np.dot(self._fwd[:2], self._tgt[:2]), -1.0, 1.0))
        if cross2d(self._fwd[:2], self._tgt[:2]) < 0:
            self._angle *= -1

        alpha = float(np.clip(self._angle / self.half_vision, -1, 1))
        beta = 1.0

        self.controller.set(alpha=alpha, beta=beta)
        if False and self.debug:
            print(f"abcpg_handler._step(t={state.time}): {alpha=} {beta=}")
        if False and self.debug:
            print(f"abcpg_handler._step(t={state.time})")
            print("> qpos:")
            print(state.data.qpos)
            print("> self.controller._state:")
            print(self.controller._state)
            print("> ctrl:")
            print(state.data.ctrl)
