import numpy as np

from .base import GenericFetchDynamics
from aapets.fetch.controllers.fetcher import FetcherCPG
from ..overlay import FetchOverlay
from ..types import InteractionMode, Keys as K
from ...common.mujoco.state import MjState


class DemoBallDynamics(GenericFetchDynamics):
    def __init__(
            self,
            state: MjState,
            overlay: FetchOverlay,
            robot: str, ball: str, human: str,
            brain: FetcherCPG):

        super().__init__(
            state, InteractionMode.BALL, overlay,
            robot, ball, human, brain
        )

        self.__ball_forces = {
            K.RIGHT: np.array((1, 0)), K.UP: np.array((0, 1)),
            K.LEFT: np.array((-1, 0)), K.DOWN: np.array((0, -1))
        }

    def _process_keys(self):
        forces = np.sum([
            force * self._is_pressed(k)
            for k, force in self.__ball_forces.items()
        ], axis=0)
        if any(forces != 0):
            self.ball.xfrc_applied[:2] = 100 * forces
        # print("[kgd-debug] ball.xfrc:", self.ball.xfrc_applied)
