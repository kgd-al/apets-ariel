from abc import abstractmethod

import glfw
from mujoco import MjSpec
from mujoco.viewer import Handle

from aapets.fetch.controllers.fetcher import FetcherCPG
from ..overlay import FetchOverlay
from ...common.config import ViewerConfig
from ...common.monitors import Monitor
from ...common.mujoco.state import MjState
from ..types import InteractionMode, Keys
from ...common.world_builder import adjust_side_camera


class GenericFetchDynamics(Monitor):
    def __init__(self,
                 state: MjState, mode: InteractionMode,
                 overlay: FetchOverlay,
                 robot: str, ball: str, human: str,
                 brain: FetcherCPG):

        super().__init__(frequency=1000)

        self.mode = mode
        self.overlay = overlay
        self.viewer = None

        self.robot = state.data.body(robot)
        self.ball = state.data.body(ball)
        self.human = human

        self.brain = brain

        self._key_processing_step_period, self._next_key_processing_step = 0.1, 0

    @classmethod
    def adjust_camera(cls, specs: MjSpec, config: ViewerConfig):
        config.camera_distance = 10
        config.camera_angle = 45
        adjust_side_camera(
            specs, config, config.robot_name_prefix,
            orthographic=False)

    def on_viewer_ready(self, viewer: Handle):
        self.viewer = viewer
        glfw.set_input_mode(self.viewer.glfw_window, glfw.STICKY_KEYS, True)

    def _step(self, state: MjState):
        if self.viewer is not None:
            if self._next_key_processing_step <= state.data.time:
                self._process_keys()
                self._next_key_processing_step += self._key_processing_step_period

    def _is_pressed(self, k: Keys):
        return glfw.get_key(self.viewer.glfw_window, k.value) == glfw.PRESS

    @abstractmethod
    def _process_keys(self): ...
