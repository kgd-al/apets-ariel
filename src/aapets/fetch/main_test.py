from mujoco import mj_forward, mj_step

from .controllers.fetcher import FetcherCPG
from .dynamics.demo import DemoDynamics
from .dynamics.demo_ball import DemoBallDynamics
from .dynamics.demo_robot import DemoRobotDynamics
from .dynamics.fetch import FetchDynamics
from .overlay import FetchOverlay
from .types import InteractionMode
from ..common.config import ViewerModes
from ..common.monitors import Monitor
from ..common.monitors.plotters.brain_activity import BrainActivityPlotter
from ..common.monitors.plotters.record import MovieRecorder
from ..common.monitors.plotters.trajectory import TrajectoryPlotter
from ..common.mujoco.callback import MjcbCallbacks
from ..common.mujoco.state import MjState
from ..common.mujoco.viewer import passive_viewer, interactive_viewer
from ..common.robot_storage import RerunnableRobot
from .types import Config as Arguments

if __name__ == "__main__":
    from ..zoo.evolve import Arguments as ZooArguments


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

    dynamics_class.adjust_world(record.mj_spec, args)

    args.robot_name_prefix = "apet"
    if args.mode is InteractionMode.HUMAN:
        args.camera = f"ortho-cam"
    else:
        args.camera = f"{args.robot_name_prefix}1_tracking-cam"
    if args.camera is not None:  # Adjust camera *before* compilation
        dynamics_class.adjust_camera(record.mj_spec, args)

    elif not args.movie:
        args.camera = "tracking"

    state = MjState.from_spec(record.mj_spec)
    model, data = state.model, state.data
    mj_forward(model, data)

    robot_name = f"{args.robot_name_prefix}1"

    targets = ["ball"]
    if args.mode is InteractionMode.HUMAN:
        targets.append("ball")

    assert record.brain[0] == "RevolveCPG"
    brain = FetcherCPG(
        record.brain[1], robot=robot_name,
        state=state, body=f"{robot_name}_world", targets=targets)

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
        match args.viewer:
            case ViewerModes.NONE:
                mj_step(model, data, nstep=int(args.duration / model.opt.timestep))

            case ViewerModes.PASSIVE:
                passive_viewer(state, args, [overlay], viewer_ready_callback=dynamics.on_viewer_ready)

            case ViewerModes.INTERACTIVE:
                interactive_viewer(state.model, state.data, args)

    if args.mode is InteractionMode.DEMO:
        dynamics.postprocess(brain_plotter, traj_plotter)


if __name__ == "__main__":
    main()
