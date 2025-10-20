import functools
import json
import logging
import os
import pprint
import random
import shutil
import sys
import time
from copy import copy
from functools import partial
from pathlib import Path
from random import Random
from typing import Iterable, Optional, Sequence, Any, Type, Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qdpy import tools, containers, algorithms
from qdpy.algorithms import Evolution, QDAlgorithmLike
from qdpy.containers import Container
from qdpy.phenotype import Individual as QDPyIndividual, IndividualLike, Fitness as QDPyFitness, \
    Features as QDPyFeatures
from qdpy.plots import plot_evals, plot_iterations

from aapets.config import EvoConfig
from src.aapets.evaluation_result import EvaluationResult
from src.aapets.genotype import Genotype


def normalize_run_parameters(options: EvoConfig):
    if options.seed is None:
        options.seed = round(1000 * time.time())
        logging.info(f"Set seed from time: {options.seed}")

    # Define the run folder
    logging.info(f"Run folder: {options.output_folder}")

    # Check the thread parameter
    max_threads = len(os.sched_getaffinity(0))
    if options.threads is None:
        options.threads = max_threads - 1
    else:
        options.threads = max(1, min(options.threads, max_threads))
    logging.info(f"Parallel: {options.threads}")

    if options.verbosity >= 0:
        raw_dict = {k: v for k, v in options.__dict__.items() if not k.startswith('_')}
        logging.info(f"Post-processed command line arguments:\n{pprint.pformat(raw_dict)}")


class QDIndividual(QDPyIndividual):
    def __init__(self, genotype):
        QDPyIndividual.__init__(self)
        self.genotype = genotype
        try:
            assert self.id is not None
        except:
            raise RuntimeError("Genotype without id")
        self.infos = dict()

    @property
    def id(self): return self.genotype.id()

    @property
    def fitness(self): return super().fitness

    @fitness.setter
    def fitness(self, fitness: Iterable[float] | float) -> None:
        if isinstance(fitness, (int, float)):
            fitness = [fitness]
        self._fitness = QDPyFitness(fitness, [1 for _ in fitness])

    @property
    def features(self): return super().features

    @features.setter
    def features(self, features: Iterable[float]):
        self._features = QDPyFeatures(list(features))

    def update(self, r: EvaluationResult):
        self.fitness = r.fitness
        self.infos.update(r.infos)


class Algorithm(Evolution):
    def __init__(self, container: Container, genome, options, labels, **kwargs):
        # Manage run id, seed, data folder...
        normalize_run_parameters(options)

        self.rng = Random(options.seed)
        random.seed(options.seed)
        np.random.seed(options.seed % (2**32-1))

        self.window = None

        self.genome_data = genome.Data(config=options, seed=options.seed)

        def select(grid):
            # return self.rng.choice(grid)
            k = min(len(grid), options.tournament)

            if self.window is None:
                source = grid.items
            else:
                source = None
            candidates = self.rng.sample(source, k)

            candidate_cells = [grid.index_grid(c.features) for c in candidates]
            curiosity = [grid.curiosity[c] for c in candidate_cells]
            if all([c == 0 for c in curiosity]):
                cell = self.rng.choice(candidate_cells)
            else:
                cell = candidate_cells[np.argmax(curiosity)]
            selection = candidates[candidate_cells.index(cell)]
            return selection

        def init(_):
            genotype = genome.random(self.genome_data)
            for _ in range(options.initial_mutations):
                genotype.mutate(self.genome_data)
            return QDIndividual(genotype)

        def vary(parent):
            return QDIndividual(parent.genotype.mutated(self.genome_data))

        sel_or_init = partial(tools.sel_or_init, init_fn=init, sel_fn=select, sel_pb=1)

        output_folder = Path(options.output_folder)
        if output_folder.exists():
            if options.overwrite:
                shutil.rmtree(output_folder, ignore_errors=True)
                logging.warning(f"Purging contents of {output_folder}, as requested")
            else:
                raise RuntimeError(f"Output folder {output_folder} already exists"
                                   f" and overwriting was NOT requested.")

        output_folder.mkdir(parents=True, exist_ok=False)

        self.labels = labels
        self._curiosity_lut = {}

        self._latest_champion = None
        self._snapshots = output_folder.joinpath("snapshots")
        self._snapshots.mkdir(exist_ok=False)
        logging.info(f"Created folder {self._snapshots}")

        budget = options.population_size * options.generations
        Evolution.__init__(self, container=container, name="MapElite",
                           budget=budget, batch_size=options.batch_size,
                           select_or_initialise=sel_or_init, vary=vary,
                           optimisation_task="maximisation",
                           **kwargs)

    def tell(self, individual: IndividualLike, *args, **kwargs) -> bool:
        grid: Grid = self.container
        added = super().tell(individual, *args, **kwargs)
        parent = self._curiosity_lut.pop(individual.id, None)
        if parent is not None:
            grid.curiosity[parent] += {True: 1, False: -.5}[added]

        if added:
            new_champion = False
            if self._latest_champion is None:
                self._latest_champion = (0, self.nb_evaluations,
                                         individual.fitness)
                new_champion = True
                # print("[kgd-debug] First champion:", self._latest_champion)
            else:
                n, timestamp, fitness = self._latest_champion
                if individual.fitness.dominates(fitness):
                    self._latest_champion = (n+1, self.nb_evaluations,
                                             individual.fitness)
                    new_champion = True
                    # print("[kgd-debug] New champion:", self._latest_champion)

            if new_champion:
                n, t, _ = self._latest_champion
                file = self._snapshots.joinpath(f"better-{n}-{t}.json")
                with open(file, "w") as f:
                    json.dump(self.to_json(individual), f)

        return added

    @classmethod
    def to_json(cls, i: IndividualLike):
        return {
            "id": i.id, #"parents": i.genotype.parents(),
            "fitness": i.fitness.values,
            "descriptors": i.features.values,
            # "stats": i.stats,
            "genotype": str(i.genotype)
        }


class Grid(containers.Grid):
    def __init__(self, **kwargs):
        containers.Grid.__init__(self, **kwargs)
        self.curiosity = np.zeros(self._shape, dtype=float)

    def update(self, iterable: Iterable,
               ignore_exceptions: bool = True, issue_warning: bool = True) -> int:
        added = containers.Grid.update(self, iterable, ignore_exceptions, issue_warning)
        return added

    def add(self, individual: IndividualLike,
            raise_if_not_added_to_depot: bool = False) -> Optional[int]:
        r = containers.Grid.add(self, individual, raise_if_not_added_to_depot)
        return r


class Logger(algorithms.TQDMAlgorithmLogger):

    final_filename = "iteration-final.p"
    iteration_filenames = "iteration-%03i.p"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         final_filename=Logger.final_filename,
                         iteration_filenames=self.iteration_filenames)

    def _started_optimisation(self, algo: QDAlgorithmLike) -> None:
        """Do a mery dance so that tqdm uses stdout instead of stderr"""
        sys.stderr = sys.__stdout__
        super()._started_optimisation(algo)
        self._tqdm_pbar.file = sys.stdout

    def _vals_to_cols_title(self, content: Sequence[Any]) -> str:
        header = algorithms.AlgorithmLogger._vals_to_cols_title(self, content)
        mid_rule = "-" * len(header)
        return header + "\n" + mid_rule

    def summary_plots(self, **kwargs):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        summary_plots(evals=self.evals, iterations=self.iterations,
                      grid=self.algorithms[0].container,
                      labels=self.algorithms[0].labels,
                      output_dir=self.log_base_path, name=Path(self.final_filename).stem,
                      **kwargs)


def plot_grid(data, filename, xy_range, cb_range, labels, fig_size, cmap="inferno",
              fontsize=12, nb_ticks=5):
    fig, ax = plt.subplots(figsize=fig_size)

    if cb_range in [None, "equal"]:
        cb_range_arg = cb_range
        cb_range = np.quantile(data, [0, 1])

        if isinstance(cb_range_arg, str):
            if cb_range_arg == "equal":
                extrema = max(abs(cb_range[0]), abs(cb_range[1]))
                cb_range = (-extrema, extrema)
            else:
                raise ValueError(f"Unknown cb_range type '{cb_range}'")

    g_shape = data.shape
    cax = ax.imshow(data.T, interpolation="none", cmap=plt.get_cmap(cmap),
                    vmin=cb_range[0], vmax=cb_range[1],
                    aspect="equal",
                    origin='lower', extent=(-.5, g_shape[0]+.5, -.5, g_shape[1]+.5))

    # Set labels
    def ticks(i):
        return np.linspace(-.5, g_shape[i]+.5, nb_ticks), [
            f"{(xy_range[i][1] - xy_range[i][0]) * x / g_shape[i] + xy_range[i][0]:3.3g}"
            for x in np.linspace(0, g_shape[i], nb_ticks)
        ]

    ax.set_xlabel(labels[1], fontsize=fontsize)
    ax.set_xticks(*ticks(0))
    ax.set_yticks(*ticks(1))
    ax.set_ylabel(labels[2], fontsize=fontsize)
    ax.autoscale_view()

    ax.xaxis.set_tick_params(which='minor', direction="in", left=False, bottom=False, top=False, right=False)
    ax.yaxis.set_tick_params(which='minor', direction="in", left=False, bottom=False, top=False, right=False)
    ax.set_xticks(np.arange(-.5, data.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-.5, data.shape[1], 1), minor=True)

    # Place the colorbar with same size as the image
    divider = make_axes_locatable(ax)
    cax2 = divider.append_axes("right", size=0.5, pad=0.15)
    cbar = fig.colorbar(cax, cax=cax2, format="%g")
    cbar.ax.tick_params(labelsize=fontsize-2)
    cbar.ax.set_ylabel(labels[0], fontsize=fontsize)

    # Write
    plt.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    if filename.exists():
        logging.info(f"Generated {filename}")
    else:
        logging.warning(f"Failed to generate {filename}")
    plt.close()


def summary_plots(evals: pd.DataFrame, iterations: pd.DataFrame, grid: Grid,
                  output_dir: Path, name: str,
                  labels, ext="png", fig_size=(4, 4), ticks=5):

    output_path = Path(output_dir).joinpath("plots")
    def path(filename): return output_path.joinpath(f"{name}_{filename}.{ext}")
    output_path.mkdir(exist_ok=True)
    assert len(str(name)) > 0

    if name.endswith("final"):
        plot_evals(evals["max0"], path("fitness_max"), ylabel="Fitness", figsize=fig_size)
        ylim_contsize = (0, len(grid)) if np.isinf(grid.capacity) else (0, grid.capacity)
        plot_evals(evals["cont_size"], path("container_size"), ylim=ylim_contsize, ylabel="Container size",
                   figsize=fig_size)
        plot_iterations(iterations["nb_updated"], path("container_updates"), ylabel="Number of updated bins",
                        figsize=fig_size)

    for filename, cb_label, data, bounds in [
        ("grid_fitness", labels[0], grid.quality_array[..., 0], grid.fitness_domain[0]),
        ("grid_activity", "activity", grid.activity_per_bin, (0, np.max(grid.activity_per_bin))),
        ("grid_curiosity", "curiosity", grid.curiosity, "equal")
    ]:
        plot_path = path(filename)
        plot_grid(data=data, filename=plot_path,
                  xy_range=grid.features_domain, cb_range=bounds, labels=[cb_label, *labels[1:]],
                  fig_size=fig_size, nb_ticks=ticks)


class MapEliteEvolver:
    def __init__(self,
                 genome: Type,
                 grid_shape, fitness_domain, features_domain, labels,
                 evaluator: Callable[[Genotype], EvaluationResult],
                 options, **kwargs):
        super().__init__(**kwargs)

        self.grid = Grid(shape=grid_shape,
                         max_items_per_bin=1,
                         fitness_domain=fitness_domain,
                         features_domain=features_domain)
        self.algo = Algorithm(self.grid,
                              genome=genome,
                              options=options, labels=labels,
                              budget=0, batch_size=0)
        self.evaluator = functools.partial(self.__evaluate, evaluator=evaluator)

        self.window = 0.1
        self.window_sizes = [round(self.window * s) for s in grid_shape]
        assert all(1 < x < s for x, s in zip(self.window_sizes, grid_shape)), \
            f"Window size '{self.window_sizes}' does not make sense"

    @staticmethod
    def __evaluate(individual, evaluator):
        res = evaluator(individual.genotype)
        individual.fitness = res["fitness"]
        individual.features = res["features"]
        return individual

    def run(self, n):
        logger = Logger(self.algo,
                        save_period=0,
                        log_base_path="tmp")

        self.algo.budget = n * self.grid.capacity
        self.algo._batch_size = self.grid.capacity
        self.algo.optimise(evaluate=self.evaluator)

    # def select(self):
    #     for item in self.
    #     return [], {}
    #
    # def feedback(self, individuals: Iterable[Individual[Genotype, Phenotype]]):
    #     pass
