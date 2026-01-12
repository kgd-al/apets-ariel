import argparse
import functools
import json
import pprint
import shutil
import time
import copy
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Optional

import cma
import humanize
import matplotlib
import numpy as np
import pandas as pd
import yaml
from mujoco import mj_step

from aapets.common import canonical_bodies
from aapets.common.config import EvoConfig, BaseConfig
from aapets.common.controllers import RevolveCPG
from aapets.common.monitors import XSpeedMonitor
from aapets.common.monitors.metrics_storage import EvaluationMetrics
from aapets.common.mujoco.callback import MjcbCallbacks
from aapets.common.mujoco.state import MjState
from aapets.common.robot_storage import RerunnableRobot
from aapets.common.world_builder import make_world, compile_world


def rerun(args):
    folder = args.output_folder
    # with open(folder.joinpath("config.json"), "rt") as f:
    #     config = vars(args).copy()
    #     config["output_folder"] = str(args.output_folder)
    #     print("Configuration:", pprint.pformat(config))
    #     f.write(json.dumps(config))

    env_kwargs = environment_arguments(args)
    env_kwargs.update(dict(
        rerun=True,
        render=not args.headless or args.movie, headless=args.headless,
        log_trajectory=True, log_reward=True,
        plot_path=folder,

        introspective=args.introspective,

        return_ff=True,

        start_paused=args.start_paused
    ))
    if args.movie:
        env_kwargs["record_settings"] = RecordSettings(
            video_directory=folder,
            overwrite=True,
            fps=25,
            width=480, height=480,

            camera_id=2
        )

    evaluator = Environment(**env_kwargs)

    # with open(folder.joinpath("cma-es.pkl"), "rb") as f:
    #     es = pickle.loads(f.read())

    start_time = time.time()
    fitness_function = evaluator.evaluate(np.random.default_rng(0).uniform(-1, 1, evaluator._params))
    fitness = -fitness_function.fitness

    steps_per_episode = (getattr(args, "evolution_simulation_time", args.simulation_time)
                         * STANDARD_CONTROL_FREQUENCY)

    summary = {
        "arch": args.arch,
        "budget": args.budget * steps_per_episode,
        "neighborhood": np.nan,
        "width": np.nan,
        "depth": np.nan,
        "reward": args.reward,
        "run": args.seed,
        "body": args.body + ("45" if args.rotated else ""),
        "params": evaluator.num_parameters,
        "tps": steps_per_episode / (time.time() - start_time),
    }
    if args.arch == "mlp":
        summary["width"] = args.width
        summary["depth"] = args.depth
    elif args.arch == "cpg":
        summary["neighborhood"] = args.neighborhood
    summary.update(fitness_function.infos)
    summary = pd.DataFrame.from_dict({k: [v] for k, v in summary.items()})
    summary.index = [folder]

    summary.to_csv(folder.joinpath("summary.csv"))
    print(summary.to_string())

    return fitness


class Environment:
    def __init__(self, args: "Arguments"):
        self.robot = canonical_bodies.get(args.body)
        self.world = make_world(self.robot.spec)

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
        RerunnableRobot(
            mj_spec=self.state.spec,
            brain=("RevolveCPG", champion),
            metrics=metrics,
            misc=dict(),
            config=self.args
        ).save(self.args.data_folder.joinpath("champion.zip"))

        RerunnableRobot.load(self.args.data_folder.joinpath("champion.zip"))

    @staticmethod
    def _evaluate(weights, xml, args, return_float=False):
        state, model, data = MjState.from_string(xml).unpacked
        cpg = RevolveCPG(weights, state)

        fitness = XSpeedMonitor("apet1")
        with MjcbCallbacks(state, [cpg], {"fitness": fitness}, args):
            mj_step(model, data, nstep=int(args.duration / model.opt.timestep))

        if return_float:
            return fitness.value
        else:
            return EvaluationMetrics(dict(xspeed=fitness.value)), fitness.value


@dataclass
class Arguments(BaseConfig, EvoConfig):
    body: Annotated[str, "Morphology to use",
                    dict(choices=canonical_bodies.get_all(), required=True)] = None

    budget: Annotated[int, "Number of CMA-ES evaluations to perform"] = 10
    threads: Annotated[Optional[int], ("Number of threads to use. A positive number requests that number of core, zero"
                                       "disables parallelism and -1 requests everything")] = 1

    initial_std: Annotated[float, "Initial standard deviation for CMA-ES"] = .5

    symlink_last: Annotated[bool, "Make a symbolic link to the last run"] = True


def main() -> int:
    start, err = time.perf_counter(), 0

    # ==========================================================================
    # Parse command-line arguments

    parser = argparse.ArgumentParser(description="Rerun evolved champions")
    Arguments.populate_argparser(parser)
    args = parser.parse_args(namespace=Arguments())

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
    evaluator.save_champion(res.xbest, rerun_metrics)
    if rerun_fitness != res.fbest:
        print("Different fitness value on rerun:")
        print(res.fbest, res.xbest)
        print("Rerun:", rerun_metrics)

    args.evolution_simulation_time = args.simulation_time
    args.simulation_time = 15
    args.movie = True
    rerun(args)

    duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start))
    print(f"Completed evolution is {duration} with exit code {err}")

    return err


if __name__ == "__main__":
    exit(main())
