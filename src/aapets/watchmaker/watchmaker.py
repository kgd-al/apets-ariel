from dataclasses import dataclass
from random import Random

import numpy as np

from aapets.watchmaker.config import WatchmakerConfig
from aapets.watchmaker.window import MainWindow


class Genotype:
    @dataclass
    class Data:
        size: int
        rng: np.random.Generator

    def __init__(self, data: np.ndarray):
        self.data = data

    @classmethod
    def random(cls, data: Data):
        # return cls(.5 * np.ones(data.size))
        return cls(data.rng.uniform(-1, 1, data.size))


class Individual:
    def __init__(self, genotype: Genotype):
        self.genotype = genotype
        self.fitness = float("nan")
        self.video = None


class Watchmaker:
    def __init__(self, window: MainWindow, config: WatchmakerConfig):
        self.window = window
        self.config = config

        self.genetic_data = Genotype.Data(
            size=8,
            rng=np.random.default_rng(config.seed)
        )
        self.population = [
            Individual(Genotype.random(self.genetic_data))
        ]
