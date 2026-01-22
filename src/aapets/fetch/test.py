from dataclasses import dataclass
from enum import Enum, auto, StrEnum
from pathlib import Path
from typing import Annotated

import numpy as np
from mujoco import mj_forward, mjtGeom, mjv_initGeom, mjv_connector, MjSpec, mj_step, mjtJoint, mjr_overlay
from mujoco.viewer import Handle

from .ABCpg import ABCpg
from ..common.config import ViewerConfig, BaseConfig
from ..common.monitors import Monitor
from ..common.monitors.plotters.record import MovieRecorder
from ..common.mujoco.callback import MjcbCallbacks
from ..common.mujoco.state import MjState
from ..common.mujoco.viewer import passive_viewer
from ..common.robot_storage import RerunnableRobot
from ..common.world_builder import adjust_side_camera, adjust_shoulder_camera

if __name__ == "__main__":
    from ..zoo.evolve import Arguments


class FetchOverlay:
    def __init__(self, brain: ABCpg):
        self.brain = brain
        self.geom_id = []

    def start(self, viewer: Handle, state: MjState):
        scene = viewer.user_scn

        n = 4
        self.geom_id = [scene.ngeom+i for i in range(n)]
        scene.ngeom += n

        mjv_initGeom(scene.geoms[self.geom_id[0]],
                     mjtGeom.mjGEOM_ARROW,
                     np.zeros(3), np.zeros(3), np.zeros(9),
                     [1, 1, 0, 1])

        mjv_initGeom(scene.geoms[self.geom_id[1]],
                     mjtGeom.mjGEOM_ARROW,
                     np.zeros(3), np.zeros(3), np.zeros(9),
                     [0, 1, 1, 1])

        mjv_initGeom(scene.geoms[self.geom_id[2]],
                     mjtGeom.mjGEOM_ARROW2,
                     np.zeros(3), np.zeros(3), np.zeros(9),
                     [1, 1, 1, 1])

        mjv_initGeom(scene.geoms[self.geom_id[3]],
                     mjtGeom.mjGEOM_LABEL,
                     np.zeros(3), np.zeros(3), np.zeros(9),
                     [0, 0, 0, 1])

    def render(self, viewer: Handle, state: MjState):
        scene = viewer.user_scn
        mjv_connector(scene.geoms[self.geom_id[0]],
                      mjtGeom.mjGEOM_ARROW, .005,
                      self.brain.body.xpos, self.brain.body.xpos + self.brain.forward)

        mjv_connector(scene.geoms[self.geom_id[1]],
                      mjtGeom.mjGEOM_ARROW, .005,
                      self.brain.body.xpos, self.brain.body.xpos + self.brain.goal)

        scene.geoms[self.geom_id[2]].label = f"a: {self.brain.angle*180/np.pi:+.2f}"
        if self.brain.is_target_visible:
            scene.geoms[self.geom_id[2]].rgba = [1, 1, 1, 1]
        else:
            scene.geoms[self.geom_id[2]].rgba = [1, 0, 0, 1]

        fwd_pos = self.brain.body.xpos + .5 * self.brain.forward
        tgt_pos = self.brain.body.xpos + .5 * self.brain.goal
        offset = .5 * (tgt_pos - fwd_pos)
        mjv_connector(scene.geoms[self.geom_id[2]],
                      mjtGeom.mjGEOM_ARROW2, .0025,
                      fwd_pos + offset, tgt_pos + offset)

        # scene.geoms[self.geom_id[2]].text
        # viewer.set_texts((
        #     None, None,
        #     "Alpha\nBeta",
        #     f"{self.brain.alpha}\n{self.brain.beta}"
        # ))

    def stop(self, viewer: Handle, state: MjState):
        pass


class InteractionMode(StrEnum):
    BALL = auto()
    ROBOT = auto()
    HUMAN = auto()


class Keys(Enum):
    RIGHT = 262
    UP = 265
    LEFT = 263
    DOWN = 264


class FetchDynamics:
    def __init__(self,
                 state: MjState, mode: InteractionMode,
                 robot: str, ball: str, human: str,
                 brain: ABCpg):

        self.mode = mode

        self.robot = state.data.body(robot)
        self.ball = state.data.body(ball)
        self.human = None

        k = Keys
        match mode:
            case InteractionMode.BALL:
                self.forces = {k.RIGHT: (1, 0), k.UP: (0, 1), k.LEFT: (-1, 0), k.DOWN: (0, -1)}
                self.keys = list(self.forces.keys())

            case InteractionMode.ROBOT:
                self.brain = brain
                self.forces = {k.RIGHT: (0, 1), k.LEFT: (0, -1), k.UP: (1, 1), k.DOWN: (1, -1)}
                self.keys = list(self.forces.keys())
                self.ranges = [np.pi / 16, .5]
                self.modulators = [0, 0]

    def ball_mode_process_key(self, key: Keys):
        self.ball.xfrc_applied[:2] = 50 * np.array(self.forces[key])
        # print("[kgd-debug] ball.xfrc:", self.ball.xfrc_applied)

    def robot_mode_process_key(self, key: Keys):
        index, value = self.forces[key]
        self.modulators[index] += value * self.ranges[index]
        self.brain.overwrite_modulators(*self.modulators)

    # def key_press(self, key: int, scancode: int, action: int, mods: int):
    def key_press(self, key: int):
        for k in self.keys:
            if key == k.value:
                match self.mode:
                    case InteractionMode.BALL:
                        self.ball_mode_process_key(k)
                    case InteractionMode.ROBOT:
                        self.robot_mode_process_key(k)
                    case InteractionMode.HUMAN:
                        self.human_mode_process_key(k)

                return True
        return False


def add_ball(specs: MjSpec, pos):
    ball = specs.worldbody.add_body(
        name="ball",
        pos=pos,
        mass=.2,
    )
    ball.add_geom(
        name="ball",
        type=mjtGeom.mjGEOM_SPHERE,
        size=(.05, 0, 0),
        rgba=(1, 1, 1, 1),
    )
    ball.add_joint(type=mjtJoint.mjJNT_FREE, stiffness=0, damping=0, frictionloss=.01, armature=0)


def add_walls(specs: MjSpec, extent: int):
    wall = specs.worldbody.add_body(
        name="walls"
    )

    height = .25
    color = (1, 1, 1, .1)
    for i, (x, y, r) in enumerate([(1, 0, .5), (-1, 0, .5), (0, 1, 0), (0, -1, 0)]):
        wall.add_geom(
            name=f"wall_{i}",
            pos=(extent * x, extent * y, .5 * height),
            type=mjtGeom.mjGEOM_BOX,
            axisangle=[0, 0, 1, r * np.pi],
            size=(extent, .1, height),
            rgba=color,
        )


@dataclass
class Arguments(BaseConfig, ViewerConfig):
    robot_archive: Annotated[
        Path, "Where to look for a pre-trained robot",
        dict(required=True)
    ] = None

    test_folder: Annotated[Path, "Where to store the results"] = Path("tmp/fetch")

    mode: Annotated[
        InteractionMode, "What type of interactions to do",
        dict(choices=list(InteractionMode))
    ] = InteractionMode.BALL


def main():
    args = Arguments.parse_command_line_arguments(description="Fetch task software prototype")
    record = RerunnableRobot.load(args.robot_archive)
    record.config.override_with(args, verbose=True)

    folder = args.test_folder
    folder.mkdir(exist_ok=True, parents=True)

    args.mode = InteractionMode.BALL
    # args.mode = InteractionMode.ROBOT
    # args.mode = InteractionMode.HUMAN

    args.camera = "apet1_tracking-cam"
    if args.camera is not None:  # Adjust camera *before* compilation
        match args.mode:
            case InteractionMode.BALL:
                args.camera_distance = 10
                args.camera_angle = 45
                adjust_side_camera(
                    record.mj_spec, args, args.robot_name_prefix,
                    orthographic=False)

            case InteractionMode.ROBOT:
                args.camera_distance = .5
                args.camera_angle = 20
                adjust_shoulder_camera(
                    record.mj_spec, args, args.robot_name_prefix,
                    orthographic=False)

    elif not args.movie:
        args.camera = "tracking"

    add_ball(record.mj_spec, (1, 0, .05))
    add_walls(record.mj_spec, extent=5)

    state = MjState.from_spec(record.mj_spec)
    model, data = state.model, state.data
    mj_forward(model, data)

    assert record.brain[0] == "RevolveCPG"
    brain = ABCpg(record.brain[1], state=state, body="apet1_world", targets=["ball"])

    monitors: dict[str, Monitor] = {}
    if args.movie:
        monitors["movie_recorder"] = MovieRecorder(
            args.movie_framerate, args.movie_width, args.movie_height,
            folder.joinpath("video.mp4"),
            camera=args.camera, shadows=True,
            speed_up=8
        )

    dynamics = FetchDynamics(
        state, mode=args.mode,
        robot="apet1_world", ball="ball", human=None,
        brain=brain
    )
    with MjcbCallbacks(state, [brain], monitors, args) as callback:
        if args.movie:
            mj_step(model, data, nstep=int(args.duration / model.opt.timestep))

        else:
            passive_viewer(state, args, [FetchOverlay(brain)], dynamics.key_press)


if __name__ == "__main__":
    main()
