import copy
import pprint
import traceback
from collections import defaultdict
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
from ..common.monitors.metrics_storage import RESET, BAD, GOOD, WARN
from ..common.monitors.plotters.brain_activity import BrainActivityPlotter
from ..common.mujoco.callback import MjcbCallbacks
from ..common.mujoco.state import MjState
from ..common.robot_storage import RerunnableRobot


@dataclass
class Arguments(RerunArguments):
    fix: Annotated[bool, "Try to create a copy of provided robot archive with errors fixed"] = False
    interactive: Annotated[bool, "Are interactions allowed"] = False


def abs_y(p: np.ndarray) -> np.ndarray:
    p = copy.copy(p)
    p[1] = abs(p[1])
    return p


def morphological_symmetry(state: MjState, robot_name):
    class MjSymmetry(defaultdict):
        def __init__(self):
            super().__init__()

            mj_forward(state.model, state.data)
    
            bodies = SymmetricalBodies(list)
            for i in range(state.model.nbody):
                body = state.data.body(i)
                name = body.name
                if not name.startswith(f"{robot_name}") or name.split("_")[-1][0] != "C":
                    continue
                bodies[string_hash(abs_y(body.xpos))].append(name)

        def valid(self):
            return all(len(p) == 2 for p in self.values())

        @staticmethod
        def string_hash(p: np.ndarray):
            return np.array2string(np.round(p, 3)+0)

    return bodies


def check(args: Arguments, record: RerunnableRobot):
    err = 0

    robot_name = args.robot_name_prefix

    state, model, data = MjState.from_spec(record.mj_spec).unpacked

    mj_forward(model, data)

    if record.config.symmetry is not Symmetry.NONE:
        symmetry = morphological_symmetry(state, robot_name)
        if not symmetry.valid():
            print(f"{BAD}Morphology is not symmetrical:{RESET}")
            pprint.pprint(symmetry)
            err += 1
        else:
            print(f"{GOOD}Morphology is symmetrical{RESET}")

    if record.config.symmetry is Symmetry.BOTH and err == 0:
        brain: SymmetricalABCPG = controllers.get(record.brain[0])(
            weights=record.brain[2], state=state, name=robot_name, **record.brain[1])
        assert isinstance(brain, SymmetricalABCPG)

        cpgs = len(brain.actuators)
        np.set_printoptions(linewidth=1000, precision=1)
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
        hinges = {pos: names for pos, names in symmetry.items() if names[0].endswith("servo")}
        print(hinges)
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
                        any(abs_y(lhs) != abs(rhs) for lhs, rhs in zip(data[ixs[0]], data[ixs[1]]))):
                    err += 1

        fig.tight_layout()
        plot_file = args.robot_archive.with_suffix(f".brain_activity.symmetrical.{args.plot_format}")
        fig.savefig(plot_file, bbox_inches="tight")
        print("Saved symmetrical brain activity to", plot_file)
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
