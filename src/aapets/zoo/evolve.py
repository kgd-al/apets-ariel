import argparse
import functools
import shutil
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Optional

import cma
import humanize
import matplotlib
import numpy as np
import pandas as pd
from mujoco import mj_step

from aapets.common import canonical_bodies, morphological_measures
from aapets.common.config import EvoConfig, BaseConfig, ViewerModes
from aapets.common.controllers import RevolveCPG
from aapets.common.monitors import XSpeedMonitor
from aapets.common.monitors.metrics_storage import EvaluationMetrics
from aapets.common.mujoco.callback import MjcbCallbacks
from aapets.common.mujoco.state import MjState
from aapets.common.robot_storage import RerunnableRobot
from aapets.common.world_builder import make_world, compile_world
from aapets.bin.rerun import Arguments as RerunArguments, main as _rerun


def rerun(args, champion_archive):
    rr = RerunnableRobot.load(champion_archive)

    if args is None:
        args = rr.config
        assert isinstance(args, Arguments)

    rerun_args = RerunArguments.copy_from(args)

    rerun_args.robot_archive = champion_archive

    rerun_args.movie = True
    rerun_args.viewer = ViewerModes.NONE

    rerun_args.movie = True
    rerun_args.camera = f"{args.robot_name_prefix}1_tracking-cam"
    rerun_args.camera_angle = 45
    rerun_args.camera_distance = 2
    rerun_args.camera_center = "com"

    rerun_args.plot_format = "png"
    rerun_args.plot_trajectory = True
    rerun_args.plot_brain_activity = True
    rerun_args.render_brain_genotype = False
    rerun_args.render_brain_phenotype = False

    _rerun(rerun_args)

    make_summary(args, len(rr.brain[1]), rr.metrics.data["xspeed"])


def make_summary(args, params, fitness):
    folder = args.data_folder

    steps_per_episode = args.duration * args.control_frequency

    summary = {
        "budget": args.budget * steps_per_episode,
        "fitness": fitness,
        "run": args.seed,
        "body": args.body,
        "params": params,
    }

    mm = morphological_measures.measure(canonical_bodies.get(args.body).spec)
    summary.update(mm.major_metrics)
    summary = pd.DataFrame.from_dict({k: [v] for k, v in summary.items()})
    summary.index = [folder]

    summary.to_csv(folder.joinpath("summary.csv"))
    print(summary.to_string())


class Environment:
    def __init__(self, args: "Arguments"):
        self.robot = canonical_bodies.get(args.body)
        self.world = make_world(self.robot.spec, camera_centered=True)

        self.state, _, _ = compile_world(self.world)
        self._params = RevolveCPG.num_parameters(self.state)

        self.args = args

        self.evaluate = functools.partial(
            self._evaluate,
            xml=self.state.to_string(),
            args=self.args
        )

        self.cma_evaluate = functools.partial(
            self.evaluate, return_float=True)

    @property
    def num_parameters(self): return self._params

    def save_champion(self, champion: np.ndarray, metrics: EvaluationMetrics):
        path = self.args.data_folder.joinpath("champion.zip")
        RerunnableRobot(
            mj_spec=self.state.spec,
            brain=(RevolveCPG.name(), champion),
            metrics=metrics,
            misc=dict(),
            config=self.args
        ).save(path)
        return path

    @staticmethod
    def _evaluate(weights, xml, args, return_float=False):
        state, model, data = MjState.from_string(xml).unpacked
        cpg = RevolveCPG(weights, state)

        fitness = XSpeedMonitor("apet1")
        with MjcbCallbacks(state, [cpg], {"fitness": fitness}, args):
            mj_step(model, data, nstep=int(args.duration / model.opt.timestep))

        if return_float:
            return -fitness.value
        else:
            return EvaluationMetrics(dict(xspeed=fitness.value)), -fitness.value


@dataclass
class Arguments(BaseConfig, EvoConfig):
    body: Annotated[str, "Morphology to use",
                    dict(choices=canonical_bodies.get_all())] = None

    budget: Annotated[int, "Number of CMA-ES evaluations to perform"] = 10
    threads: Annotated[Optional[int], ("Number of threads to use. A positive number requests that number of core, zero"
                                       "disables parallelism and -1 requests everything")] = 1

    initial_std: Annotated[float, "Initial standard deviation for CMA-ES"] = .5

    symlink_last: Annotated[bool, "Make a symbolic link to the last run"] = True

    rerun: Annotated[Optional[Path], "Path to the archive to use for re-evaluation"] = None


def main() -> int:
    start, err = time.perf_counter(), 0

    # ==========================================================================
    # Parse command-line arguments

    args = Arguments.parse_command_line_arguments(
        "Evolve a cpg controller via CMA-ES for a robot from the zoo (canonical_bodies)")

    if args.rerun is not None:
        if not args.rerun.exists():
            raise ValueError(f"Cannot rerun from non-existing archive '{args.rerun}'")
        exit(rerun(None, args.rerun))

    if args.body is None:
        raise RuntimeError("No canonical body specified")

    if args.data_folder is None:
        args.data_folder = Path("tmp/cma/").joinpath(f"run-{args.seed}")

    folder = args.data_folder
    folder_str = str(folder) + "/"

    if folder.exists():
        if args.overwrite:
            shutil.rmtree(folder)
        else:
            raise ValueError(f"Output folder '{folder}' already exists and overwriting was not requested")
    folder.mkdir(parents=True, exist_ok=True)

    if args.symlink_last:
        symlink = folder.parent.joinpath("last")
        symlink.unlink(missing_ok=True)
        symlink.symlink_to(args.data_folder.name, target_is_directory=True)

    args.write_yaml(folder.joinpath("config.yaml"))
    args.pretty_print()

    evaluator = Environment(args)

    # Initial parameter values for the brain.
    initial_mean = evaluator.num_parameters * [0.5]

    # We use the CMA-ES optimizer from the cma python package.
    options = cma.CMAOptions()
    options.set("verb_filenameprefix", folder_str)
    # options.set("bounds", [-1.0, 1.0])
    options.set("seed", args.seed)
    options.set("tolfun", 0)
    options.set("tolflatfitness", 10)
    es = cma.CMAEvolutionStrategy(initial_mean, args.initial_std, options)
    # args.threads = 0
    es.optimize(evaluator.cma_evaluate, maxfun=args.budget, n_jobs=args.threads, verb_disp=1)
    with open(folder.joinpath("cma-es.pkl"), "wb") as f:
        f.write(es.pickle_dumps())

    res = es.result_pretty()
    matplotlib.use("agg")
    cma.plot(folder_str, abscissa=1)
    # plt.tight_layout()
    cma.s.figsave(folder.joinpath('plot.png'), bbox_inches='tight')  # save current figure
    cma.s.figsave(folder.joinpath('plot.pdf'), bbox_inches='tight')  # save current figure

    rerun_metrics, rerun_fitness = evaluator.evaluate(res.xbest)
    champion_archive = evaluator.save_champion(res.xbest, rerun_metrics)
    if rerun_fitness != res.fbest:
        print("Different fitness value on rerun:")
        print(res.fbest, res.xbest)
        print("Rerun:", rerun_metrics)

    rerun(args, champion_archive)

    duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start))
    print(f"Completed evolution is {duration} with exit code {err}")

    return err


if __name__ == "__main__":
    exit(main())
