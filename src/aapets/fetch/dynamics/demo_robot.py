import glfw
from mujoco import MjSpec
from mujoco.viewer import Handle

from .base import GenericFetchDynamics
from ..controllers.fetcher import FetcherCPG
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

        self._gamepad = None

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
                self._gamepad = i
                print("Connected to gamepad", i, glfw.get_gamepad_name(self._gamepad))
                break
            elif glfw.joystick_present(i):
                print("Found raw joystick", i, glfw.get_joystick_name(i))
        if self._gamepad is None:
            print("No gamepad found")

    def prepare(self):
        self.brain.overwrite_modulators(0, 0)

    def _process_keys(self):
        changed = False
        ranges = [.1, .05]
        modulators = [self.brain.alpha, self.brain.beta]

        for k, (index, value) in self.__robot_controls.items():
            if not self._key_pressed(k):
                continue
            modulators[index] += value * ranges[index]
            changed = True

        if self._gamepad is not None:
            gamepad_state = glfw.get_gamepad_state(self._gamepad)
            def _axis(a): return round(gamepad_state.axes[a], 2)
            lateral = (
                .5 * (_axis(glfw.GAMEPAD_AXIS_RIGHT_TRIGGER) - _axis(glfw.GAMEPAD_AXIS_LEFT_TRIGGER))
            )

            lhs = _axis(glfw.GAMEPAD_AXIS_LEFT_Y)
            rhs = _axis(glfw.GAMEPAD_AXIS_RIGHT_Y)
            forward = max([-x for x in [lhs, rhs] if x != 0] or [0])

            assert -1 <= lateral <= 1
            assert -1 <= forward <= 1
            modulators = [lateral, forward]

            changed = True

        if changed:
            self.brain.overwrite_modulators(*modulators)
