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
from .types import Individual, morphological_symmetry, has_self_collisions, behavioral_symmetry
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
    err, fixable = 0, True

    robot_name = args.robot_name_prefix

    state, model, data = MjState.from_spec(record.mj_spec).unpacked

    mj_forward(model, data)

    if collisions := has_self_collisions(state.spec, collect=True):
        if args.verbosity > 1:
            print(f"{BAD}Self-colliding morphology:{RESET}")
            pprint.pprint(collisions)
        err += 1
        fixable = False

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

        cpgs = len(brain.actuators)

        brain_activity = BrainActivityPlotter(
            args.sample_frequency, robot_name,
            args.robot_archive.with_suffix(f".brain_activity.{args.plot_format}"),
            verbose=True
        )

        with MjcbCallbacks(state, [brain], dict(brain_activity=brain_activity), args):
            mj_step(model, data, nstep=int(args.duration / model.opt.timestep))

        mismatches = behavioral_symmetry(
            state, brain_activity,
            robot_name=args.robot_name_prefix,
            save_plot=args.robot_archive.with_suffix(f".brain_activity.symmetrical.{args.plot_format}"),
            collect=True
        )

        if len(mismatches) > 0:
            if args.verbosity > 1:
                print(f"{BAD}Gait is not symmetrical:{RESET}")
                pprint.pprint(mismatches)
            err += 1
        elif args.verbosity > 1:
            print(f"{GOOD}Gait is symmetrical{RESET}")

        if len(mismatches) > 0 and args.verbosity >= 10:
            with np.printoptions(linewidth=1000, precision=1):
                print(brain._weight_matrix[:cpgs, :cpgs])
                print(brain._weight_matrix)

    return err, fixable


def main(args: Arguments):
    assert args.robot_archive is not None
    record = RerunnableRobot.load(args.robot_archive)
    if args.verbosity > 1:
        print("Loaded:", args.robot_archive)

    try:
        err, fixable = check(args, record)
    except Exception as expt:
        err, fixable = 10, False
        print("Test threw exception:", expt)

    assert (genome := record.misc.get("genotype")) is not None
    assert (s_data := record.misc.get("genotype_rendering").get("data")) is not None

    initial_err = err
    fixing_err = 0
    if not args.fix:
        fixing_err = err
    elif initial_err > 0 and fixable:
        ind = Individual(genome)
        ind._develop(s_data)
        path = Evaluator.save_robot(ind, record.metrics,
                                    record.config.where(data_folder=args.robot_archive.parent),
                                    s_data, args.robot_archive.stem + ".fixed")
        args.robot_archive = path

        record = RerunnableRobot.load(path)
        try:
            fixing_err, _ = check(args, record)
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

    if args.verbosity >= 1:
        print(args.robot_archive, end='')
        if initial_err == 0:
            print(f"\t{GOOD}OK{RESET}")
        elif fixing_err == 0:
            print(f"\t{WARN}FIXED{RESET} ({initial_err} errors)")
        else:
            print(f"\t{BAD}NOK{RESET} ({fixing_err} errors)")

    if args.interactive and ((fixing_err > 0) or (not fixable and initial_err > 0)):
        print("Launching interactive window")
        args.viewer = ViewerModes.PASSIVE
        args.auto_start = False
        args.verbosity = 0
        rerun(args)

    return err


if __name__ == "__main__":
    exit(main(Arguments.parse_command_line_arguments("Rerun evolved champions")))
