import multiprocessing
import random
import time
from multiprocessing import Queue, Process
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from deap import creator, base, tools
from mypy.types import Any

from . import evaluation
from .config import Config
from .evaluation import Evaluator
from .novelty import NoveltyArchive
from .plotting import min_max_plots, shaded_plots, LearningLog, Genealogy
from .revdeknn import RevDEKNN
from .types import StaticData, Individual
from ..common.monitors.metrics_storage import EvaluationMetrics


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
        np.random.seed(config.seed)

        fitness = self.create("RobotFitness", base.Fitness, weights=[1, 1])
        self.individual = self.create("Individual", Individual, fitness=fitness, descriptors=list)

        self.toolbox = base.Toolbox()
        make_individual = self.register("individual", self.individual.random, data=self.data)
        self.population = self.register("population", tools.initRepeat,
                                        list, make_individual)
        # self.mate = self.register("mate", ind.crossover, data=self.data)
        # self.mutate = self.register("mutate", ind.mutate, data=self.data)
        evaluator = evaluation.evaluator(config.task)
        self.evaluate = self.register(
            "evaluate", self._evaluate,
            evaluator=evaluator, config=self.config, return_metrics=False)
        self.evaluate_and_learn = self.register(
            "evaluate_and_learn", self._evaluate_and_learn,
            evaluator=evaluator, config=self.config)
        # self.select = self.register("select", tools.selNSGA2)

        self.individual.init(config)

        LearningLog.init_queue(Queue())
        threads = config.threads or 1
        if threads > 1:
            self.pool = multiprocessing.Pool(
                processes=threads,
                initializer=LearningLog.init_queue, initargs=(LearningLog.queue(),),
            )
            self.register("map", self.pool.map)

        self.archive = NoveltyArchive(config, evaluator.descriptor_names())

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

        self.genealogy = Genealogy(config.data_folder, enabled=False)

    @staticmethod
    def create(name, value, **kwargs):
        creator.create(name, value, **kwargs)
        return getattr(creator, name)

    def register(self, name, value, *args, **kwargs) -> Any:
        self.toolbox.register(name, value, *args, **kwargs)
        return getattr(self.toolbox, name)

    @staticmethod
    def _evaluate(ind: Individual, evaluator: Evaluator, config: Config, return_metrics):
        state = evaluator.prepare(ind, config)
        evaluator.reset(state)
        return evaluator.evaluate(state, ind.weights, config, return_metrics)

    @classmethod
    def _evaluate_and_learn(cls, ind: Individual, evaluator: Evaluator, config: Config):
        assert config.learning > 0
        if ind.invalid:
            print("Skipping learning/evaluation for invalid individual", ind.id)
            return evaluator.evaluate_invalid(ind, config), ind.weights

        state = evaluator.prepare(ind, config)
        np.random.seed(config.seed + ind.id)

        def _eval(_population):
            _fitnesses, _data = [], []
            for _i in _population:
                evaluator.reset(state)
                r = evaluator.evaluate(state, _i, config, return_metrics=False)
                _fitnesses.append([-r.fitness])  # Invert because revdeknn wants to minimize
                _data.append(r)
            return _fitnesses, _data

        theta_0 = ind.weights
        population = np.array(
            [theta_0] + [
                theta_0 + np.random.normal(0, config.rev_de_knn_deviation)
                for _ in range(config.rev_de_knn_sample_size - 1)
            ]
        )

        fitnesses, data = _eval(population)
        LearningLog.log_pop(ind.id, 0, fitnesses)

        teacher = RevDEKNN(eval_fn=_eval, config=config)

        learning_iterations = config.learning // config.rev_de_knn_sample_size
        for i in range(learning_iterations-1):
            population, fitnesses, data = teacher.step(population, fitnesses, data)
            LearningLog.log_pop(ind.id, i, fitnesses)

        assert all(lhs <= rhs for lhs, rhs in zip(fitnesses[:-1], fitnesses[1:])), f"Non monotonic fitnesses: {fitnesses}"

        return data[0], population[0]

    def run(self, generations: Optional[int] = None):
        generations = generations or self.config.generations

        def _eval(_pop):
            for ind, (result, weights) in zip(_pop, self.toolbox.map(self.evaluate_and_learn, _pop)):
                ind.weights = weights
                ind.fitness.values = (result.fitness, -np.inf)
                ind.descriptors = result.descriptors

        def _novelty(_pop):
            for ind, n in zip(_pop, self.archive.process_generation([i.descriptors for i in _pop])):
                ind.fitness.values = (*ind.fitness.values[:-1], n)

        def _genealogy(_gen, _pop):
            for ind in _pop:
                self.genealogy.write(_gen, ind)

        def _log(_pop, _gen, evals):
            self.logbook.record(gen=_gen, evals=evals, **self.stats.compile(_pop))
            print(self.logbook.stream)

            self.detailed_logbook.record(gen=_gen, **self.detailed_stats.compile(_pop))

        def _selection(_pop, _k):
            _filtered_pop = [p for p in _pop if all(np.isfinite(p.fitness.values))]
            return tools.selNSGA2(_filtered_pop, k=min(_k, len(_filtered_pop)))

        # Learning process
        if self.config.learning > 0:
            logger_proc = Process(target=LearningLog.writer,
                                  args=(LearningLog.queue(),
                                        self.config.data_folder))
            logger_proc.start()

        pop = self.population(n=self.config.population_size)
        _eval(pop)
        _novelty(pop)
        _genealogy(0, pop)
        pop = _selection(pop, len(pop))
        _log(pop, 0, evals=len(pop))

        for gen in range(1, generations):
            offspring = []
            while len(offspring) < len(pop):
                if self.rng.random() < self.config.probability_crossover:
                    parents = _tournament_dcd(pop, self.rng, k=2, n=2)
                    child = self.individual.mated(*parents, self.data, self.config.probability_mutation)

                else:
                    parent = _tournament_dcd(pop, self.rng, k=2, n=1)
                    child = self.individual.mutated(parent, self.data)

                del child.fitness.values
                offspring.append(child)

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            assert len(invalid) == len(offspring)
            _eval(invalid)
            _novelty(pop + offspring)  # Always recompute novelty
            _genealogy(gen, invalid)
            # _novelty(invalid)  # Only compute novelty once (wrong?)
            pop = _selection(pop + offspring, len(pop))
            _log(pop, gen, evals=len(invalid))

        pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)
        champion = max(pareto_front[0], key=lambda _ind: _ind.fitness.values[0])

        self.genealogy.close()

        if self.config.learning > 0:
            LearningLog.close_queue()
            logger_proc.join()

        return champion

    def save(self, champion: Individual, metrics: EvaluationMetrics):
        out = self.config.data_folder
        assert out is not None

        champion_path = Evaluator.save_robot(champion, metrics, self.config, self.data)

        self._to_file(self.logbook, out.joinpath("log"))
        self._to_file(self.detailed_logbook, out.joinpath("detailed_log"))
        self.archive.save(out)

        return champion_path

    @classmethod
    def plot(cls, folder: Path):
        assert folder is not None

        min_max_plots(cls._from_file(folder.joinpath("log")), folder.joinpath("log.png"))
        shaded_plots(cls._from_file(folder.joinpath("detailed_log")), folder.joinpath("detailed.png"))

        NoveltyArchive.plot_from(folder)
        LearningLog.plot(folder)
        Genealogy.plot(folder)

    @staticmethod
    def _to_file(logbook: tools.Logbook, out: Path):
        top_level = [f for f in logbook.header if f not in logbook.chapters]
        data = {("_", f): logbook.select(f) for f in top_level}

        for chapter_name, chapter in logbook.chapters.items():
            for field in chapter.header:
                data[(chapter_name, field)] = chapter.select(field)

        df = pd.DataFrame(data)

        df.to_csv(_with_suffix(out), index=False)

        return df

    @staticmethod
    def _from_file(path: Path):
        df = pd.read_csv(_with_suffix(path), header=[0, 1])
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df


def _tournament_dcd(population, rng: random.Random, k: int = 2, n: int = 1):
    """
    Selects n unique individuals from the population using tournaments of size k.
    :param population: The population to choose from
    :param rng: Source of randomness
    :param k: Tournament size (larger equals more pressure)
    :param n: Number of individuals to select
    :return: n unique individuals from the population (not clones, direct references)
    """

    def _tournament(lhs, rhs):
        if lhs.fitness.dominates(rhs.fitness):
            return lhs
        elif rhs.fitness.dominates(lhs.fitness):
            return rhs

        if lhs.fitness.crowding_dist < rhs.fitness.crowding_dist:
            return lhs
        elif rhs.fitness.crowding_dist < lhs.fitness.crowding_dist:
            return rhs

        return lhs if rng.random() < 0.5 else rhs

    p = len(population)
    chosen: list[int] = []

    for i in range(n):
        idx = rng.sample(range(p - i), k)

        for excl in sorted(chosen):
            idx = [r + 1 if r >= excl else r for r in idx]

        # Actually not optimal for k>2 as ties are broken randomly k-1 times instead of once
        best = idx[0]
        for j in idx[1:]:
            if _tournament(population[j], population[best]):
                best = j

        chosen.append(best)

    return population[chosen[0]] if n == 1 else [population[i] for i in chosen]


def _with_suffix(path: Path):
    if path.suffix == "":
        path = path.with_suffix(".csv")
    return path
