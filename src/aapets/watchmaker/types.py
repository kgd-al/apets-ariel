import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Callable

import numpy as np


class Genotype:
    @dataclass
    class Data:
        size: int
        rng: np.random.Generator
        scale: float
        range: float

    def __init__(self, data: np.ndarray):
        self.data = data.copy()

    @classmethod
    def random(cls, data: Data) -> "Genotype":
        # return cls(.5 * np.ones(data.size))
        m_range = data.range or 1
        return cls(data.rng.uniform(-m_range, m_range, data.size))

    def mutated(self, data: Data) -> "Genotype":
        clone = self.__class__(self.data)
        clone.data += data.rng.normal(0, data.scale, data.size)
        if (m_range := data.range) is not None:
            clone.data.clip(-m_range, m_range, out=clone.data)
        return clone


class Individual:
    __next_id = 0

    def __init__(self, genotype: Genotype, parent: int = -1):
        self.genotype = genotype
        self.fitness = float("nan")
        self.video = None

        self.id = self.__next_id + 1
        self.__class__.__next_id += 1

        self.parent = parent

    def __repr__(self): return f"R{self.id}"

    def mutated(self, data: Genotype.Data):
        return self.__class__(self.genotype.mutated(data), parent=self.id)

    def to_string(self):
        return (f"{self.id} {self.parent} {self.fitness}"
                " " + " ".join([f"{x:g}" for x in self.genotype.data]))


class Selector(ABC):
    @abstractmethod
    def select(self, population: List[Individual]) -> Tuple[int, Individual]: ...


class HillClimberSelector(Selector):
    def select(self, population: List[Individual]) -> Tuple[int, Individual]:
        i_max = max(range(len(population)), key=lambda i: population[i].fitness)
        return i_max, population[i_max]


class RandomSelector(Selector):
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def select(self, population: List[Individual]) -> Tuple[int, Individual]:
        i = self.rng.randint(0, len(population)-1)
        return i, population[i]
