from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
from mujoco import mj_forward, mjtGeom, MjSpec, mj_step, mjtJoint

from .dynamics.demo import DemoDynamics
from .dynamics.demo_ball import DemoBallDynamics
from .dynamics.demo_robot import DemoRobotDynamics
from .dynamics.fetch import FetchDynamics
from aapets.fetch.controllers.fetcher import FetcherCPG
from .overlay import FetchOverlay
from .types import InteractionMode
from ..common.config import ViewerConfig, BaseConfig, ViewerModes
from ..common.monitors import Monitor
from ..common.monitors.plotters.brain_activity import BrainActivityPlotter
from ..common.monitors.plotters.record import MovieRecorder
from ..common.monitors.plotters.trajectory import TrajectoryPlotter
from ..common.mujoco.callback import MjcbCallbacks
from ..common.mujoco.state import MjState
from ..common.mujoco.viewer import passive_viewer
from ..common.robot_storage import RerunnableRobot

if __name__ == "__main__":
    from ..zoo.evolve import Arguments


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

    dynamics_class = {
        InteractionMode.DEMO: DemoDynamics,
        InteractionMode.BALL: DemoBallDynamics,
        InteractionMode.ROBOT: DemoRobotDynamics,
        InteractionMode.HUMAN: FetchDynamics,
    }[args.mode]

    args.robot_name_prefix = "apet"
    args.camera = f"{args.robot_name_prefix}1_tracking-cam"
    if args.camera is not None:  # Adjust camera *before* compilation
        dynamics_class.adjust_camera(record.mj_spec, args)

    elif not args.movie:
        args.camera = "tracking"

    add_ball(record.mj_spec, (1, 0, .05))
    add_walls(record.mj_spec, extent=5)

    state = MjState.from_spec(record.mj_spec)
    model, data = state.model, state.data
    mj_forward(model, data)

    robot_name = f"{args.robot_name_prefix}1"

    assert record.brain[0] == "RevolveCPG"
    brain = FetcherCPG(record.brain[1], state=state, body=f"{robot_name}_world", targets=["ball"])

    overlay = FetchOverlay(brain, mode=args.mode)

    monitors: dict[str, Monitor] = {}

    monitors["fetch-dynamics"] = dynamics = dynamics_class(
        state,
        overlay=overlay,
        robot="apet1_world", ball="ball", human="None",
        brain=brain,
    )

    match args.mode:
        case InteractionMode.DEMO:
            args.duration = dynamics_class.duration()
        case InteractionMode.ROBOT:
            dynamics.prepare()

    if args.movie:
        monitors["movie_recorder"] = MovieRecorder(
            args.movie_framerate, args.movie_width, args.movie_height,
            folder.joinpath("video.mp4"),
            camera=args.camera, shadows=True,
            speed_up=8
        )

    if True:
        monitors["brain_activity"] = brain_plotter = BrainActivityPlotter(
            args.sample_frequency, robot_name,
            folder.joinpath(f"brain_activity.pdf"),
            rename={
                f"apet1_{lhs}-servo": rhs
                for lhs, rhs in [
                    ("C-H", "FL"), ("C-H-B-H", "FLK"),
                    ("C-rH", "FR"), ("C-rH-B-H", "FRK"),
                    ("C-lH", "BL"), ("C-lH-B-H", "BLK"),
                    ("C-bH", "BR"), ("C-bH-B-H", "BRK"),
                ]
            }
        )

        monitors["trajectory"] = traj_plotter = TrajectoryPlotter(
            args.sample_frequency, robot_name,
            folder.joinpath(f"trajectory.pdf")
        )

    with MjcbCallbacks(state, [brain], monitors, args) as callback:
        if args.movie or args.viewer is ViewerModes.NONE:
            mj_step(model, data, nstep=int(args.duration / model.opt.timestep))

        else:
            passive_viewer(state, args, [overlay], viewer_ready_callback=dynamics.on_viewer_ready)

    if args.mode is InteractionMode.DEMO:
        dynamics.postprocess(brain_plotter, traj_plotter)


if __name__ == "__main__":
    main()
