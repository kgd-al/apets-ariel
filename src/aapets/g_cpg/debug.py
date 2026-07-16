import pprint
import traceback
from dataclasses import dataclass
from typing import Annotated

import matplotlib
import numpy as np
from matplotlib import pyplot as plt, figure
from mujoco import mj_forward, mj_step

from ariel.utils.mujoco_ops import has_self_collision
from .config import Symmetry
from .evaluation import Evaluator
from .types import Individual, morphological_symmetry, self_collision
from ..bin.rerun import Arguments as RerunArguments, main as rerun
from ..common import controllers
from ..common.config import ViewerModes
from ..common.controllers.ABCpg import SymmetricalABCPG
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

    if not (collisions := self_collision(state)):
        if args.verbosity > 1:
            print(f"{BAD}Self-colliding morphology:{RESET}")
            pprint.pprint(collisions)
        err += 1

    if record.config.symmetry is not Symmetry.NONE and err == 0:
        symmetry = morphological_symmetry(state, robot_name, "body")
        if not symmetry.valid():
            if args.verbosity > 1:
                print(f"{BAD}Morphology is not symmetrical:{RESET}")
                symmetry.pretty_print()
            err += 1
        elif args.verbosity > 1:
            print(f"{GOOD}Morphology is symmetrical{RESET}")

    if record.config.symmetry is Symmetry.BOTH and err == 0:
        brain: SymmetricalABCPG = controllers.get(record.brain[0])(
            weights=record.brain[2], state=state, name=robot_name, **record.brain[1])
        assert isinstance(brain, SymmetricalABCPG)
        hinges = morphological_symmetry(state, robot_name, "joint")

        cpgs = len(brain.actuators)
        np.set_printoptions(linewidth=1000, precision=1)
        if args.verbosity >= 10:
            print(brain._weight_matrix[:cpgs, :cpgs])
            print(brain._weight_matrix)

        brain_activity = BrainActivityPlotter(
            args.sample_frequency, robot_name,
            args.robot_archive.with_suffix(f".brain_activity.{args.plot_format}"),
            verbose=True
        )

        with MjcbCallbacks(state, [brain], dict(brain_activity=brain_activity), args):
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
        if not hinges.valid():
            if args.verbosity > 1:
                print(f"{BAD}Non-symmetrical hinges:{RESET}")
                pprint.pprint(hinges)
            err += 1

        mismatches = []
        for i, (pos, names) in enumerate(hinges.items()):
            for j, label in enumerate(["Position", "Control"]):
                ax = axes[i][j]
                ixs = []
                for name in names:
                    ix = 2 * actuators[name] + j + 1
                    ixs.append(ix)
                    ax.plot(x, data[ix], zorder=1)
                ax.set_ylim(-y_lim, y_lim)

                title = f"{pos}: {names}"
                if i == 0:
                    title = f"{label}\n\n" + title

                ax.set_title(title)

                if (j == 1 and
                        any(abs(lhs) != abs(rhs) for lhs, rhs in zip(data[ixs[0]], data[ixs[1]]))):
                    mismatches.append(names)

        fig.tight_layout()
        plot_file = args.robot_archive.with_suffix(f".brain_activity.symmetrical.{args.plot_format}")
        fig.savefig(plot_file, bbox_inches="tight")
        if args.verbosity > 1:
            print("Saved symmetrical brain activity to", plot_file)
        plt.close(fig)

        brain_activity.plot("brain_activity.base.pdf")
        fig.savefig("brain_activity.symmetrical.pdf", bbox_inches="tight")

        if len(mismatches) > 0:
            if args.verbosity > 1:
                print(f"{BAD}Gait is not symmetrical:{RESET}")
                pprint.pprint(mismatches)
            err += 1
        elif args.verbosity > 1:
            print(f"{GOOD}Gait is symmetrical{RESET}")

    return err


def main(args: Arguments):
    assert args.robot_archive is not None
    record = RerunnableRobot.load(args.robot_archive)
    if args.verbosity > 1:
        print("Loaded:", args.robot_archive)

    try:
        err = check(args, record)
    except:
        err = 10

    assert (genome := record.misc.get("genotype")) is not None
    assert (s_data := record.misc.get("genotype_rendering").get("data")) is not None

    initial_err = err
    fixing_err = 0
    if not args.fix:
        fixing_err = err
    elif initial_err > 0:
        ind = Individual(genome)
        ind._develop(s_data)
        path = Evaluator.save_robot(ind, record.metrics,
                                    record.config.where(data_folder=args.robot_archive.parent),
                                    s_data, args.robot_archive.stem + ".fixed")
        args.robot_archive = path

        record = RerunnableRobot.load(path)
        try:
            fixing_err = check(args, record)
            status = f"{GOOD}fixed" if err == 0 else f"{WARN}half-backed"
            expt = None
        except Exception as e:
            fixing_err = 10
            status = f"{BAD}buggy"
            expt = e
        if args.verbosity > 1:
            print(f"Saved {status}{RESET} champion archive in", path)
        if expt is not None:
            if args.verbosity > 1:
                print("Error was", expt)
                traceback.print_exception(expt)

    if args.verbosity == 1:
        print(args.robot_archive, end='')
        if initial_err == 0:
            print(f"\t{GOOD}OK{RESET}")
        elif fixing_err == 0:
            print(f"\t{WARN}FIXED{RESET} ({initial_err} errors)")
        else:
            print(f"\t{BAD}NOK{RESET} ({fixing_err} errors)")

    if fixing_err > 0 and args.interactive:
        args.viewer = ViewerModes.PASSIVE
        args.auto_start = False
        args.verbosity = 0
        rerun(args)

    return err


if __name__ == "__main__":
    exit(main(Arguments.parse_command_line_arguments("Rerun evolved champions")))
