from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
from mujoco import mj_forward, mjtGeom, mjv_initGeom, mjv_connector, MjSpec, mj_step
from mujoco.viewer import Handle

from aapets.common.world_builder import adjust_camera
from .ABCpg import ABCpg
from ..common.config import ViewerConfig, BaseConfig
from ..common.monitors import Monitor
from ..common.monitors.plotters.record import MovieRecorder
from ..common.mujoco.callback import MjcbCallbacks
from ..common.mujoco.state import MjState
from ..common.mujoco.viewer import passive_viewer
from ..common.robot_storage import RerunnableRobot

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

    def stop(self, viewer: Handle, state: MjState):
        pass


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
    ball.add_freejoint()


@dataclass
class Arguments(BaseConfig, ViewerConfig):
    robot_archive: Annotated[
        Path, "Where to look for a pre-trained robot",
        dict(required=True)
    ] = None

    test_folder: Annotated[Path, "Where to store the results"] = Path("tmp/fetch")


def main():
    args = Arguments.parse_command_line_arguments(description="Fetch task software prototype")
    record = RerunnableRobot.load(args.robot_archive)
    record.config.override_with(args, verbose=True)

    folder = args.test_folder
    folder.mkdir(exist_ok=True, parents=True)

    args.camera = "apet1_tracking-cam"
    if args.camera is not None:  # Adjust camera *before* compilation
        args.camera_distance = 4
        args.camera_angle = 45
        adjust_camera(record.mj_spec, args, args.robot_name_prefix)

    elif not args.movie:
        args.camera = "tracking"

    add_ball(record.mj_spec, (-1, 0, .05))

    state = MjState.from_spec(record.mj_spec)
    model, data = state.model, state.data
    mj_forward(model, data)

    assert record.brain[0] == "RevolveCPG"
    brain = ABCpg(record.brain[1], state=state, body="apet1_world", target="ball")

    monitors: dict[str, Monitor] = {}
    if args.movie:
        monitors["movie_recorder"] = MovieRecorder(
            args.movie_framerate, args.movie_width, args.movie_height,
            folder.joinpath("video.mp4"),
            camera=args.camera, shadows=True,
            speed_up=8
        )

    with MjcbCallbacks(state, [brain], monitors, args) as callback:
        if args.movie:
            mj_step(model, data, nstep=int(args.duration / model.opt.timestep))

        else:
            passive_viewer(state, args, [FetchOverlay(brain)])

    print(state.data.time)

if __name__ == "__main__":
    main()
