import itertools
from abc import abstractmethod

import glfw
import numpy as np
from mujoco import MjSpec, mjtGeom, mjtJoint, mjtTexture, mjtBuiltin, mjtRndFlag, mjtMark, mjtTrn, mjtSensor, mjtObj
from mujoco.viewer import Handle

from ..controllers.fetcher import FetcherCPG
from ..overlay import FetchOverlay
from ..types import InteractionMode, Keys, Config, Buttons, NewBodyParts, FetchTaskObjects
from ...common.monitors import Monitor
from ...common.mujoco.state import MjState
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
    def adjust_camera(cls, specs: MjSpec, config: Config):
        config.camera_distance = 10
        config.camera_angle = 45
        adjust_side_camera(
            specs, config, config.robot_name_prefix,
            orthographic=False)

    @classmethod
    def adjust_world(cls, specs: MjSpec, config: Config):
        add_ball(specs, (1, 0, .05))
        add_walls(specs, extent=config.arena_extent)
        specs.add_texture(builtin=mjtBuiltin.mjBUILTIN_FLAT,
                          rgb1=[0, 0, 0], rgb2=[0, 0, 0],
                          width=1024, height=1024,
                          random=.01, mark=mjtMark.mjMARK_RANDOM, markrgb=[1, 1, 1],
                          type=mjtTexture.mjTEXTURE_SKYBOX, name="skybox")

        robot_name = f"{config.robot_name_prefix}1"
        add_mouth(specs, robot_name)
        add_eyes(specs, robot_name)

    def on_viewer_ready(self, viewer: Handle):
        self.viewer = viewer
        glfw.set_input_mode(self.viewer.glfw_window, glfw.STICKY_KEYS, True)

    def _step(self, state: MjState):
        if self.viewer is not None:
            if self._next_key_processing_step <= state.data.time:
                self._process_keys()
                self._next_key_processing_step += self._key_processing_step_period

    def _key_pressed(self, k: Keys):
        return glfw.get_key(self.viewer.glfw_window, k.value) == glfw.PRESS

    def _mouse_down(self, k: Buttons):
        return glfw.get_mouse_button(self.viewer.glfw_window, k.value) == glfw.PRESS

    def _mouse_pos(self):
        return np.array(glfw.get_cursor_pos(self.viewer.glfw_window))

    @abstractmethod
    def _process_keys(self): ...


def add_ball(specs: MjSpec, pos):
    ball = specs.worldbody.add_body(
        name=FetchTaskObjects.BALL,
        pos=pos,
        mass=.2,
    )
    ball.add_geom(
        name=FetchTaskObjects.BALL,
        type=mjtGeom.mjGEOM_SPHERE,
        size=(.05, 0, 0),
        rgba=(1, 1, 1, 1),
    )
    ball.add_joint(type=mjtJoint.mjJNT_FREE, stiffness=0, damping=0, frictionloss=.01, armature=0)


def add_walls(specs: MjSpec, extent: float):
    wall = specs.worldbody.add_body(
        name="walls"
    )

    depth, wall_height, slope_height = .1, 2.5, .25
    color = (1, 1, 1)

    pi = np.pi
    wall_angles = [.5 * pi, .5 * pi, 0, 0]
    slopes_angles = [
        (0, .25 * pi, .5 * pi), (0, -.25 * pi, -.5 * pi),
        (-.25 * pi, 0, 0), (.25 * pi, 0, 0),
    ]
    slopes_xy_delta = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0)
    ]

    for i, (x, y) in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
        wall.add_geom(
            name=f"wall_{i}",
            pos=(extent * x, extent * y, wall_height),
            type=mjtGeom.mjGEOM_BOX,
            axisangle=[0, 0, 1, wall_angles[i]],
            size=(extent, depth, wall_height),
            rgba=[*color, .5],
        )

        wall.add_geom(
            name=f"slope_{i}",
            pos=np.array([extent * x, extent * y, .5 * slope_height]) + 2 * depth * np.array(slopes_xy_delta[i]),
            type=mjtGeom.mjGEOM_BOX,
            euler=slopes_angles[i],
            size=(extent, .1, slope_height),
            rgba=[*color, 1],
        )


def add_mouth(specs: MjSpec, robot_name: str):
    cs = core_size(specs, robot_name)

    depth = .01
    mouth = specs.body(f"{robot_name}_world").add_body(
        name=NewBodyParts.MOUTH_BODY,
        pos=(np.sqrt(2) * cs - .5 * depth, 0, -.5 * cs)
    )
    mouth.add_geom(
        type=mjtGeom.mjGEOM_BOX,
        mass=.001,
        size=(depth, .01, .01)
    )

    mouth_actuator = specs.add_actuator(
        name=NewBodyParts.MOUTH_SUCKER,
        target=NewBodyParts.MOUTH_BODY,
        trntype=mjtTrn.mjTRN_BODY,
        ctrlrange=[0, 1],
    )
    mouth_actuator.set_to_adhesion(gain=5)

    specs.add_sensor(
        name=NewBodyParts.MOUTH_SENSOR,
        type=mjtSensor.mjSENS_CONTACT,
        objtype=mjtObj.mjOBJ_BODY,
        objname=mouth.name,
        intprm=[1, 0, 1]
    )


def add_eyes(specs: MjSpec, robot_name: str):
    s = core_size(specs, robot_name)

    robot = specs.body(f"{robot_name}_core")
    for dx, dz, lr in itertools.product([0, 1], [0, 1], [0, 1]):
        x = .0 + dx * .5
        z = .5 + dz * .2
        robot.add_geom(
            name=f"{NewBodyParts.SPIDER_EYES}_{dx}{dz}{lr}",
            type=mjtGeom.mjGEOM_ELLIPSOID,
            size=(.02, .01, .01),
            pos=[s, -x * s, z * s] if lr > 0 else [x * s, -s, z * s],
            rgba=[1, 0, 0, 1],
            quat=[np.cos(lr * np.pi / 4), 0, 0, np.sin(lr * np.pi / 4)],
        )


def core_size(specs: MjSpec, robot_name: str):
    core_sizes = specs.geom(f"{robot_name}_core").size
    assert core_sizes[0] == core_sizes[1] == core_sizes[2]
    return core_sizes[0]
