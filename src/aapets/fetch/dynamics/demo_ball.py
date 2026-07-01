from typing import Optional, List

import numpy as np

from .base import GenericFetchDynamics
from ..sm_fetcher import FetcherCPG
from ..overlay import FetchOverlay
from ..types import InteractionMode, Keys as K, Config
from ...common.mujoco.state import MjState


class DemoBallDynamics(GenericFetchDynamics):
    def __init__(
            self,
            state: MjState,
            overlay: FetchOverlay,
            robot: str, ball: str, human: str,
            brain: FetcherCPG,
            config: Config
    ):

        super().__init__(
            state, InteractionMode.BALL, overlay,
            robot, ball, human, brain, config
        )

        self.__ball_forces = {
            K.RIGHT: np.array((1, 0)), K.UP: np.array((0, 1)),
            K.LEFT: np.array((-1, 0)), K.DOWN: np.array((0, -1))
        }
        self.__ball_force_magnitude = config.ball_strength

    def _process_keys(self):
        forces = np.sum([
            force * (self._key_pressed(k))
            for k, force in self.__ball_forces.items()
        ], axis=0)
        self.ball.xfrc_applied[:2] = self.__ball_force_magnitude * forces
