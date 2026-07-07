import pprint
import traceback
from dataclasses import dataclass
from typing import Annotated

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from mujoco import mj_forward, mj_step

from .config import Symmetry
from .evaluation import Evaluator
from .types import Individual
from ..bin.rerun import Arguments as RerunArguments, main as rerun
from ..common import controllers
from ..common.config import ViewerModes
from ..common.controllers.ABCpg import SymmetricalABCPG
from ..common.controllers.abstract import Controller
from ..common.monitors.metrics_storage import RESET, BAD, GOOD, WARN
from ..common.monitors.plotters.brain_activity import BrainActivityPlotter
from ..common.mujoco.callback import MjcbCallbacks
from ..common.mujoco.state import MjState
from ..common.robot_storage import RerunnableRobot


@dataclass
class Arguments(RerunArguments):
    fix: Annotated[bool, "Try to create a copy of provided robot archive with errors fixed"] = False
    interactive: Annotated[bool, "Are interactions allowed"] = False


def check(args: Arguments, record: RerunnableRobot):
    err = 0

    robot_name = args.robot_name_prefix

    state, model, data = MjState.from_spec(record.mj_spec).unpacked
    mj_forward(model, data)

    if record.config.symmetry is not Symmetry.NONE:
        sym_joints = SymmetricalABCPG.symmetrical_joints(state, robot_name)
        if not sym_joints.valid():
            print(f"{BAD}Non symmetrical hinges:{RESET}")
            pprint.pprint(sym_joints)
            err += 1
        else:
            print(f"{GOOD}All hinges symmetrical{RESET}")

    if record.config.symmetry is Symmetry.BOTH:
        brain: SymmetricalABCPG = controllers.get(record.brain[0])(
            weights=record.brain[2], state=state, name=robot_name, **record.brain[1])
        assert isinstance(brain, SymmetricalABCPG)

        cpgs = len(brain.actuators)
        np.set_printoptions(linewidth=1000)
        print(brain._weight_matrix[:cpgs, :cpgs])
        print(brain._weight_matrix)

        brain_activity = BrainActivityPlotter(
            args.sample_frequency, robot_name,
            args.robot_archive.with_suffix(f".brain_activity.{args.plot_format}"),
            verbose=True
        )

        with MjcbCallbacks(state, [brain], dict(brain_activity=brain_activity), args) as callback:
            mj_step(model, data, nstep=int(args.duration / model.opt.timestep))

        w, h = matplotlib.rcParams["figure.figsize"]
        n = len(brain_activity.actuators) // 2

        y_lim = np.ceil(1.1 * brain_activity.max_range)
        fig, axes = plt.subplots(n, 2,
                                 sharex=True, sharey=True,
                                 figsize=(3 * w, .25 * n * h))

        data = brain_activity.data
        x = np.array(data[0])
        actuators = {n: i for i, n in enumerate(brain_activity.actuators.keys())}
        print(actuators)
        for i, (pos, names) in enumerate(sym_joints.items()):
            for j, label in enumerate(["Position", "Control"]):
                ax = axes[i][j]
                for name in names:
                    ix = 2 * actuators[name] + j + 1
                    print(ix, i, j, name, actuators[name])
                    ax.plot(x, data[ix], zorder=1)
                ax.set_ylim(-y_lim, y_lim)

                title = f"{pos}: {names}"
                if i == 0:
                    title = f"{label}\n\n" + title

                ax.set_title(title)

        fig.tight_layout()
        fig.savefig(args.robot_archive.with_suffix(f".brain_activity.symmetrical.{args.plot_format}"), bbox_inches="tight")
        plt.close(fig)

    return err


def main(args: Arguments):
    assert args.robot_archive is not None
    record = RerunnableRobot.load(args.robot_archive)

    try:
        err = check(args, record)
    except:
        err = 10

    assert (genome := record.misc.get("genotype")) is not None
    assert (s_data := record.misc.get("genotype_rendering").get("data")) is not None

    if err > 0 and args.fix:
        ind = Individual(genome)
        ind._develop(s_data)
        path = Evaluator.save_robot(ind, record.metrics,
                                    record.config.where(data_folder=args.robot_archive.parent),
                                    s_data, args.robot_archive.stem + ".fixed")
        args.robot_archive = path

        record = RerunnableRobot.load(path)
        try:
            status = f"{GOOD}fixed" if check(args, record) == 0 else f"{WARN}half-backed"
            expt = None
        except Exception as e:
            status = f"{BAD}buggy"
            expt = e
        print(f"Saved {status}{RESET} champion archive in", path)
        if expt is not None:
            print("Error was", expt)
            traceback.print_exception(expt)

    if err > 0 and args.interactive:
        args.viewer = ViewerModes.PASSIVE
        args.auto_start = False
        rerun(args)

    return err


if __name__ == "__main__":
    exit(main(Arguments.parse_command_line_arguments("Rerun evolved champions")))
