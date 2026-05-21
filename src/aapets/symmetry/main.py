import functools
from dataclasses import dataclass, field
from random import Random, random
from typing import Annotated, Optional

import abrain
from deap import creator, base, tools
from mypy.types import Any

from aapets.common.config import EvoConfig, BaseConfig
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule


@dataclass
class Config(BaseConfig, EvoConfig):
    @classmethod
    def yaml_tag(cls): return "MainConfig"

    population_size: Annotated[int, "Population size (duh)"] = 8  # Must be a multiple of 4
    generations: Annotated[int, "Number of generations (double duh)"] = 10


@dataclass
class Genome:
    body: CoreModule
    brain: abrain.Genome


@dataclass
class Individual:
    genome: Genome

    # rank: int | None = None
    # crowding: float = 0.0
    #
    # def dominates(self, other: "Individual") -> bool:
    #     return (all(lhs >= rhs for lhs, rhs in zip(self.fitness, other.fitness))
    #             and any(a < b for a, b in zip(self.fitness, other.fitness)))

    @classmethod
    def random(cls, rng: Random):
        print(f"random({cls})")
        return cls(Genome(rng.randint(-10, 0), rng.randint(0, 10)))

    @staticmethod
    def mutate(ind, rng: Random):
        print(f"mutate({ind}")
        return ind

    @staticmethod
    def crossover(lhs, rhs, rng: Random):
        print(f"crossover({lhs}, {rhs})")
        return rng.choice([lhs, rhs])


class DEAPWrap:
    def __init__(self, config: Config):
        self.config = config
        self.rng = Random(config.seed)

        fitness = self.create("RobotFitness", base.Fitness, weights=(1, 1))
        ind = self.create("Individual", Individual, fitness=fitness)

        self.toolbox = base.Toolbox()
        individual = self.register("individual", ind.random, rng=self.rng)
        self.population = self.register("population", tools.initRepeat,
                                        list, individual)
        self.mate = self.register("mate", ind.crossover, rng=self.rng)
        self.mutate = self.register("mutate", ind.mutate, rng=self.rng)
        self.evaluate = self.register("evaluate", self._evaluate, config=self.config)
        self.select = self.register("select", tools.selNSGA2)

    @staticmethod
    def create(name, value, **kwargs):
        creator.create(name, value, **kwargs)
        return getattr(creator, name)

    def register(self, name, value, *args, **kwargs) -> Any:
        self.toolbox.register(name, value, *args, **kwargs)
        return getattr(self.toolbox, name)

    @staticmethod
    def _evaluate(ind, config: Config):
        print(f"evaluate({ind})")
        return random(), random()

    def run(self, generations: Optional[int] = None):
        generations = generations or self.config.generations

        cxpb, mutpb = 1, 1

        pop = self.population(n=self.config.population_size)
        for ind, fit in zip(pop, self.toolbox.map(self.evaluate, pop)):
            ind.fitness.values = fit
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
            for ind, fit in zip(invalid, map(self.toolbox.evaluate, invalid)):
                ind.fitness.values = fit

            # survivor selection — this is where selNSGA2 belongs
            pop = tools.selNSGA2(pop + offspring, k=len(pop))


def main(args: Config):
    print(args)

    algo = DEAPWrap(args)
    algo.run(args.generations)

    # for g in range(10):
    #     # Select the next generation individuals
    #     offspring = toolbox.select(pop, len(pop))
    #     # Clone the selected individuals
    #     offspring = map(toolbox.clone, offspring)
    #
    #     # Apply crossover on the offspring
    #     for child1, child2 in zip(offspring[::2], offspring[1::2]):
    #         if random.random() < CXPB:
    #             toolbox.mate(child1, child2)
    #             del child1.fitness.values
    #             del child2.fitness.values
    #
    #     # Apply mutation on the offspring
    #     for mutant in offspring:
    #         if random.random() < MUTPB:
    #             toolbox.mutate(mutant)
    #             del mutant.fitness.values
    #
    #     # Evaluate the individuals with an invalid fitness
    #     invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    #     fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    #     for ind, fit in zip(invalid_ind, fitnesses):
    #         ind.fitness.values = fit
    #
    #     # The population is entirely replaced by the offspring
    #     pop[:] = offspring


if __name__ == "__main__":
    main(Config.parse_command_line_arguments("NSGA-II test"))
