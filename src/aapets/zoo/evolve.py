import argparse
import json
import pprint
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

from aapets.common import canonical_bodies
from aapets.common.config import EvoConfig, BaseConfig


def environment_arguments(args):
    env_args = dict(
        arch=args.arch,
        reward=args.reward,

        simulation_time=args.simulation_time,
        rotated=args.rotated,
    )

    if args.arch == "mlp":
        # body = modular_robots_v2.get(args.body)
        body = modular_robots_v1.get(args.body)
        assert args.depth is not None
        assert args.width is not None
        env_args.update(dict(
            body=body,
            mlp_depth=args.depth,
            mlp_width=args.width,
        ))

    elif args.arch == "cpg":
        body, cpg_network_structure, output_mapping = bco(args.body, args.neighborhood)
        env_args.update(dict(
            body=body,
            cpg_network_structure=cpg_network_structure,
            output_mapping=output_mapping,
        ))

    else:
        raise ValueError(f"Unknown architecture: {args.arch}")

    return env_args


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
    def __init__(self, body, **kwargs):
        self._body = body

            case "cpg":
                self._arch = arch
                self._cpg_network_structure = kwargs.pop("cpg_network_structure")
                self._output_mapping = kwargs.pop("output_mapping")
                self._params = self._cpg_network_structure.num_connections
                self._brain_factory = self.cpg_brain

        self.rerun = kwargs.get('rerun', False)
        self.rotated = kwargs.pop('rotated', False)

        self.evaluate_one = functools.partial(
            self._evaluate, reward=reward, **kwargs)

    @property
    def num_parameters(self): return self._params

    def cpg_brain(self, weights):
        return BrainCpgNetworkStatic.uniform_from_params(
            params=weights,
            cpg_network_structure=self._cpg_network_structure,
            initial_state_uniform=math.sqrt(2) * 0.5,
            output_mapping=self._output_mapping,
        )

    def mlp_brain(self, weights):
        return TensorBrainFactory(
            body=self._body,
            width=self._width, depth=self._depth, weights=weights
        )

    def evaluate(self, weights: np.ndarray) -> float | StepwiseFitnessFunction:
        robot = ModularRobot(body=self._body, brain=self._brain_factory(weights))

        res = self.evaluate_one(
            Evaluator.scene(robot, rerun=self.rerun, rotation=self.rotated)
        )

        # weights_np_str = np.array2string(
        #     robot.brain.make_instance()._weight_matrix,
        #     precision=2, separator=',', suppress_small=True,
        #     floatmode="maxprec", max_line_width=1000
        # )
        # print(f"-- {res} - {sha1(weights).hexdigest()}:\n{weights_np_str}")

        return res

    @staticmethod
    def _evaluate(scene: ModularRobotScene, reward, **kwargs):
        render = kwargs.pop("render", False)
        fitness_function = MoveForwardFitness(
            reward=reward,
            rerun=kwargs.pop("rerun", False),
            render=render,
            log_trajectory=kwargs.pop("log_trajectory", False),
            backup_trajectory=False,
            log_reward=kwargs.pop("log_reward", False),
            introspective=kwargs.pop("introspective", False),
        )

        return_ff = kwargs.pop("return_ff", False)

        plot_path: Optional[Path] = kwargs.pop("plot_path", None)
        simulator = LocalSimulator(scene=scene, fitness_function=fitness_function, **kwargs)
        simulator.run(render)

        if plot_path is not None and plot_path.exists():
            fitness_function.do_plots(plot_path)

        # Minimize negative value
        if return_ff:
            return fitness_function
        else:
            return -fitness_function.fitness


@dataclass
class Arguments(BaseConfig, EvoConfig):
    body: Annotated[str, "Morphology to use",
                    dict(choices=canonical_bodies.get_all(), required=True)] = None

    budget: Annotated[int, "Number of CMA-ES evaluations to perform"] = 10
    threads: Annotated[Optional[int], ("Number of threads to use. A positive number requests that number of core, zero"
                                       "disables parallelism and -1 requests everything")] = 1

    std_dev: Annotated[float, "Initial standard deviation for CMA-ES"] = .5


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

    args.write_yaml(folder.joinpath("config.yaml"))
    args.pretty_print()

    env_args = environment_arguments(args)

    evaluator = Environment(**env_args)

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
    es.optimize(evaluator.evaluate, maxfun=args.budget, n_jobs=args.threads, verb_disp=1)
    with open(folder.joinpath("cma-es.pkl"), "wb") as f:
        f.write(es.pickle_dumps())

    res = es.result_pretty()
    matplotlib.use("agg")
    cma.plot(folder_str, abscissa=1)
    # plt.tight_layout()
    cma.s.figsave(folder.joinpath('plot.png'), bbox_inches='tight')  # save current figure
    cma.s.figsave(folder.joinpath('plot.pdf'), bbox_inches='tight')  # save current figure

    rerun_fitness = evaluator.evaluate(res.xbest)
    if rerun_fitness != res.fbest:
        print("Different fitness value on rerun:")
        print(res.fbest, res.xbest)
        print("Rerun:", rerun_fitness)

    args.evolution_simulation_time = args.simulation_time
    args.simulation_time = 15
    args.movie = True
    rerun(args)

    duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start))
    print(f"Completed evolution is {duration} with exit code {err}")

    return err

if __name__ == "__main__":
    exit(main())
