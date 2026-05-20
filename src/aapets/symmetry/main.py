import functools
from dataclasses import dataclass
from random import Random

from deap import creator, base, tools
from mypy.types import Any

from aapets.common.config import EvoConfig, BaseConfig


@dataclass
class Config(BaseConfig, EvoConfig):
    @classmethod
    def yaml_tag(cls): return "MainConfig"


@dataclass
class Genotype:
    body: int
    brain: int

    @staticmethod
    def random(rng: Random):
        print("random()")
        return creator.Genotype(rng.randint(-10, 0), rng.randint(0, 10))

    @staticmethod
    def mutate(rng: Random, ind):
        print(f"mutate({ind}")
        return ind

    @staticmethod
    def crossover(rng: Random, lhs, rhs):
        print(f"crossover({lhs}, {rhs})")
        return rng.choice([lhs, rhs])


class DEAPWrap:
    def __init__(self, config: Config):
        self.config = config
        self.rng = Random(config.seed)

        creator.create("RobotFitness", base.Fitness, weight=(1, 1))
        creator.create("RobotGenome", Genotype, fitness=creator.RobotFitness)

        tb = base.Toolbox()
        tb.register("individual", functools.partial(Genotype.random, rng=self.rng))
        tb.register("population", tools.initRepeat,
                    list, tb.individual)
        tb.register("mate", Genotype.crossover, rng=self.rng)
        tb.register("mutate", Genotype.mutate, rng=self.rng)
        tb.register("evaluate", functools.partial(self._evaluate, config=self.config))
        tb.register("select", tools.selNSGA2)

    @staticmethod
    def _evaluate(ind, config: Config):
        print(f"evaluate({ind})")


def main(args: Config):
    print(args)
    genotype_class = Genotype

    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=10,
        sampling=genotype_class.Random,
        mutation=genotype_class.Mutation,
        crossover=genotype_class.Crossover,
    )


if __name__ == "__main__":
    main(Config.parse_command_line_arguments("NSGA-II test"))
