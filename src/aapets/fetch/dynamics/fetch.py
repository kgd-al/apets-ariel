
from .base import GenericFetchDynamics
from aapets.fetch.controllers.fetcher import FetcherCPG
from ..overlay import FetchOverlay
from ..types import InteractionMode
from ...common.mujoco.state import MjState


class FetchDynamics(GenericFetchDynamics):
    def __init__(self,
                 state: MjState, mode: InteractionMode,
                 overlay: FetchOverlay,
                 robot: str, ball: str, human: str,
                 brain: FetcherCPG):

        super().__init__(
            state, InteractionMode.HUMAN, overlay,
            robot, ball, human, brain
        )

    def _step(self, state: MjState):
        super()._step(state)

    def _process_keys(self):
        pass

