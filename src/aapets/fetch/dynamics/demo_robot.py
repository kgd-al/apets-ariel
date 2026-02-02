import glfw
from mujoco import MjSpec
from mujoco.viewer import Handle

from .base import GenericFetchDynamics
from aapets.fetch.controllers.fetcher import FetcherCPG
from ..overlay import FetchOverlay
from ..types import InteractionMode, Keys as K
from ...common.config import ViewerConfig
from ...common.mujoco.state import MjState
from ...common.world_builder import adjust_shoulder_camera


class DemoRobotDynamics(GenericFetchDynamics):
    def __init__(self,
                 state: MjState,
                 overlay: FetchOverlay,
                 robot: str, ball: str, human: str,
                 brain: FetcherCPG):

        super().__init__(
            state, InteractionMode.ROBOT, overlay,
            robot, ball, human, brain
        )

        self.__robot_controls = {
            K.RIGHT: (0, -1), K.LEFT: (0, 1), K.UP: (1, 1), K.DOWN: (1, -1)
        }

        self._joystick = None

    @classmethod
    def adjust_camera(cls, specs: MjSpec, config: ViewerConfig):
        config.camera_distance = .5
        config.camera_angle = 20
        adjust_shoulder_camera(
            specs, config, config.robot_name_prefix,
            orthographic=False)

    def on_viewer_ready(self, viewer: Handle):
        super().on_viewer_ready(viewer)

        for i in range(glfw.JOYSTICK_LAST):
            if glfw.joystick_is_gamepad(i):
                self._joystick = i
                print("Connected to gamepad", i, glfw.get_gamepad_name(self._joystick))
                break
            elif glfw.joystick_present(i):
                print("Found raw joystick", i, glfw.get_joystick_name(i))
        if self._joystick is None:
            print("No gamepad found")


    def prepare(self):
        self.brain.overwrite_modulators(0, 0)

    def _process_keys(self):
        changed = False
        ranges = [.1, .05]
        modulators = [self.brain.alpha, self.brain.beta]
        for k, (index, value) in self.__robot_controls.items():
            if not self._is_pressed(k):
                continue
            modulators[index] += value * ranges[index]
            changed = True
        if changed:
            self.brain.overwrite_modulators(*modulators)
