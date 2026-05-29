import os

import multiprocessing

from dataclasses import dataclass
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from deap import creator, base, tools
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from mypy.types import Any

from .config import Config
from .evaluation import forward_locomotion, save_robot
from .novelty import NoveltyArchive
from .types import Genome, StaticData, Individual


def _shaded_plots(df: pd.DataFrame, out: Path):
    chapters = df.columns.get_level_values(0).unique()
    chapters = [c for c in chapters if c != ""]

    fig, axes = plt.subplots(len(chapters), 1,
                             figsize=(10, 4 * len(chapters)),
                             sharex=True)

    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

    gens = df[("", "gen")]
    for ax, chapter in zip(axes, chapters):
        df_c = df[chapter]
        cols = df_c.columns
        if (n := len(cols)) % 2 != 1:
            continue
        alpha = 1/(2*len(cols))
        mid = (n - 1)//2

        handles = []

        ax.plot(gens, df_c[cols[mid]], label=cols[mid])
        handles.append(ax.plot([], [], color=color, linewidth=2, label=cols[mid])[0])

        for i in range(mid):
            a, b = mid-i-1, mid+i+1
            ax.fill_between(gens, df_c[cols[a]], df_c[cols[b]], alpha=alpha, color=color)
            handles.append(Patch(facecolor=color, alpha=(mid-i) * alpha, label=f"{cols[a]}-{cols[b]}"))
        ax.set_title(chapter)
        ax.legend(handles=handles)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("generation")
    plt.tight_layout()
    plt.savefig(out, dpi=150)


def _min_max_plots(df: pd.DataFrame, out: Path):
    chapters = df.columns.get_level_values(0).unique()
    chapters = [c for c in chapters if c != ""]

    fig, axes = plt.subplots(len(chapters), 1,
                             figsize=(10, 4 * len(chapters)),
                             sharex=True)

    for ax, chapter in zip(axes, chapters):
        for col in df[chapter].columns:
            ax.plot(df[("", "gen")], df[chapter][col], label=col)
        ax.set_title(chapter)
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("generation")
    plt.tight_layout()
    plt.savefig(out, dpi=150)


class DEAPWrap:
    def __init__(self, config: Config):
        self.config = config
        self.data = StaticData.create_for_eshn_cppn(
            dimension=3, seed=config.seed,
            with_input_bias=True, with_input_length=True,
            with_output_bias=False, with_leo=True,
            with_innovations=True, with_lineage=True
        )
        self.data.set_config(config)
        self.rng = self.data.rng
        random.seed(config.seed)

        fitness = self.create("RobotFitness", base.Fitness, weights=[1, 1])
        ind = self.create("Individual", Individual, fitness=fitness)

        self.toolbox = base.Toolbox()
        individual = self.register("individual", ind.random, data=self.data)
        self.population = self.register("population", tools.initRepeat,
                                        list, individual)
        self.mate = self.register("mate", ind.crossover, data=self.data)
        self.mutate = self.register("mutate", ind.mutate, data=self.data)
        self.evaluate = self.register("evaluate", self._evaluate, config=self.config)
        self.save = self.register("save", self._save_robot, config=self.config)
        self.select = self.register("select", tools.selNSGA2)

        self.pool = multiprocessing.Pool(config.threads or 1)
        self.register("map", self.pool.map)

        self.archive = NoveltyArchive(config)

        fitnesses = ["Speed", "Novelty"]
        stats, detailed_stats = {}, {}
        for i, f in enumerate(fitnesses):
            f_stats = tools.Statistics(key=lambda _ind, _i=i: _ind.fitness.values[_i])
            f_stats.register("min", np.min)
            f_stats.register("avg", np.mean)
            f_stats.register("max", np.max)
            stats[f] = f_stats

            d_stats = tools.Statistics(key=lambda _ind, _i=i: _ind.fitness.values[_i])
            for q in [0, 10, 25, 50, 75, 90, 100]:
                d_stats.register(f"q{q}", lambda x, _q=q: np.percentile(x, _q))
            detailed_stats[f] = d_stats

        nsga_stats = tools.Statistics()
        nsga_stats.register("pareto_size", lambda inds: len(
            tools.sortNondominated(inds, len(inds), first_front_only=True)[0]
        ))
        stats["nsga"] = nsga_stats

        # Front-facing stats (min, max, ...)
        self.stats = tools.MultiStatistics(**stats)

        self.logbook = tools.Logbook()
        self.logbook.header = ["gen", "evals"] + self.stats.fields

        self.logbook.header = ["gen", "evals", "nsga"] + fitnesses
        for f, stats in stats.items():
            self.logbook.chapters[f].header = stats.fields

        # Detailed stats (quantiles)
        self.detailed_stats = tools.MultiStatistics(**detailed_stats)
        self.detailed_logbook = tools.Logbook()
        self.detailed_logbook.header = ["gen"] + fitnesses
        for f, stats in detailed_stats.items():
            self.detailed_logbook.chapters[f].header = stats.fields

    @staticmethod
    def create(name, value, **kwargs):
        creator.create(name, value, **kwargs)
        return getattr(creator, name)

    def register(self, name, value, *args, **kwargs) -> Any:
        self.toolbox.register(name, value, *args, **kwargs)
        return getattr(self.toolbox, name)

    @staticmethod
    def _evaluate(ind, config: Config):
        return forward_locomotion(ind, config)

    @staticmethod
    def _save_robot(ind, config: Config):
        return save_robot(ind, config)

    def run(self, generations: Optional[int] = None):
        generations = generations or self.config.generations

        cxpb, mutpb = 1, 1

        def eval(_pop):
            for ind, (fitness, descriptors) in zip(_pop, self.toolbox.map(self.evaluate, _pop)):
                novelty = self.archive.novelty(descriptors)
                ind.fitness.values = (*fitness, novelty)
                ind.descriptors = descriptors

        pop = self.population(n=self.config.population_size)
        eval(pop)
        pop = tools.selNSGA2(pop, k=len(pop))

        for gen in range(generations):
            # parent selection — NSGA-II style tournament
            parents = tools.selTournamentDCD(pop, k=len(pop))
            offspring = [self.toolbox.clone(ind) for ind in parents]

            # vary
            for c1, c2 in zip(offspring[::2], offspring[1::2]):
                if self.rng.random() < cxpb:
                    self.toolbox.mate(c1, c2)
                    del c1.fitness.values
                    del c2.fitness.values

            for mutant in offspring:
                if self.rng.random() < mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # evaluate invalids
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            eval(invalid)

            # survivor selection — this is where selNSGA2 belongs
            pop = tools.selNSGA2(pop + offspring, k=len(pop))

            self.logbook.record(gen=gen, evals=len(invalid), **self.stats.compile(pop))
            print(self.logbook.stream)

            self.detailed_logbook.record(gen=gen, evals=len(invalid), **self.detailed_stats.compile(pop))

        pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)
        champion = max(pareto_front[0], key=lambda _ind: _ind.fitness.values[0])

        return champion

    def do_plots(self):
        out = self.config.data_folder

        # Save logbooks
        df = self._to_file(self.logbook, out.joinpath("log"))
        ddf = self._to_file(self.detailed_logbook, out.joinpath("detailed_log"))

        # Generate plots
        _min_max_plots(df, out.joinpath("log.png"))
        _shaded_plots(ddf, out.joinpath("detailed.png"))

    @staticmethod
    def _to_file(logbook: tools.Logbook, out: Path):
        # Save the csv
        top_level = [f for f in logbook.header if f not in logbook.chapters]
        data = {("", f): logbook.select(f) for f in top_level}

        for chapter_name, chapter in logbook.chapters.items():
            for field in chapter.header:
                data[(chapter_name, field)] = chapter.select(field)

        df = pd.DataFrame(data)
        if out.suffix == "":
            out = out.with_suffix(".parquet")
        df.to_parquet(out, index=False)

        return df
