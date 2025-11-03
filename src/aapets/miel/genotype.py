import copy
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from typing import TypeVar, Type, Generic

import numpy as np
import numpy.typing as npt
from abrain import Genome as CPPNGenome

from aapets.common.config import EvoConfig

G = TypeVar("G", bound="GenericGenotype")


class GenericGenotype(ABC):
    class GenericGenotypeData(ABC, Generic[G]):
        pass

    @classmethod
    @abstractmethod
    def random(cls: Type[G], data: GenericGenotypeData[G]) -> G:
        raise NotImplementedError()

    @abstractmethod
    def mutate(self: G, data: GenericGenotypeData[G]) -> None:
        raise NotImplementedError()

    def mutated(self: G, data: GenericGenotypeData[G]) -> G:
        _copy = self.copy()
        _copy.mutate(data)
        return _copy

    @classmethod
    @abstractmethod
    def mate(cls: Type[G], lhs: G, rhs: G, data: GenericGenotypeData[G]) -> G:
        raise NotImplementedError()

    @abstractmethod
    def copy(self: G) -> G:
        raise NotImplementedError()


class Genotype(GenericGenotype):
    __key = object()

    @dataclass
    class Data:
        body: "Genotype.Body.Data"
        brain: CPPNGenome.Data
        config: EvoConfig

        def __init__(self, config: EvoConfig, seed=None):
            size = config.body_genotype_size or 64
            self.body = Genotype.Body.Data(
                rng=np.random.default_rng(seed),
                size=size, fields=3,
            )
            self.brain = CPPNGenome.Data.create_for_eshn_cppn(
                dimension=3, seed=seed,
                with_input_bias=True, with_input_length=True,
                with_leo=True, with_output_bias=False,
                with_innovations=True, with_lineage=True)
            self.config = config

    @dataclass
    class Body:
        @dataclass
        class Data:
            rng: np.random.Generator
            size: int
            fields: int

        data: list[npt.NDArray[np.float32]] = field(default_factory=list)

        @classmethod
        def random(cls, data: "Genotype.Body.Data") -> "Genotype.Body":
            return cls([data.rng.random(data.size, np.float32) for _ in range(data.fields)])

        def mutate(self, data: "Genotype.Body.Data", mutation_rate: float) -> None:
            rate = 1
            while data.rng.random() <= rate:
                self.data[data.rng.integers(data.fields)][data.rng.integers(data.size)] += data.rng.normal(0, 1)
                rate *= mutation_rate

        @classmethod
        def crossover(cls,
                      lhs: "Genotype.Body", rhs: "Genotype.Body",
                      data: "Genotype.Body.Data"):
            child_data = []
            for lhs_field, rhs_field in zip(lhs.data, rhs.data):
                i = data.rng.integers(data.size)
                child_data.append(lhs_field[:i] + rhs_field[i:])
            return cls(child_data)

        def copy(self) -> "Genotype.Body":
            return self.__class__(
                copy.deepcopy(self.data)
            )

    def __init__(self, body, brain, *, _key):
        assert _key is Genotype.__key, "Cannot be constructed directly"
        self.body = body
        self.brain = brain

    @property
    def id(self): return self.brain.id

    @classmethod
    def random(cls, data: Data) -> "Genotype":
        return cls(
            body=cls.Body.random(data.body),
            brain=CPPNGenome.random(data.brain),
            _key=cls.__key
        )

    def mutate(self, data: Data):
        if data.brain.rng.random() < data.config.body_brain_mutation_ratio:
            self.body.mutate(data.body, mutation_rate=data.config.body_mutation_rate)
        else:
            self.brain.mutate(data.brain)

    @classmethod
    def mate(cls, lhs: "Genotype", rhs: "Genotype", data: Data):
        return cls(
            body=cls.Body.crossover(lhs.body, rhs.body, data.body),
            brain=CPPNGenome.crossover(lhs.brain, rhs.brain, data.brain),
            _key=cls.__key
        )

    def copy(self):
        return Genotype(
            body=self.body.copy(),
            brain=self.brain.copy(),
            _key=self.__key
        )

    def to_json(self):
        return dict(
            body=[a.tolist() for a in self.body.data],
            brain=self.brain.to_json(),
        )

    @classmethod
    def from_json(cls, data):
        return cls(
            body=cls.Body([np.array(a) for a in data["body"]]),
            brain=CPPNGenome.from_json(data["brain"]),
            _key=cls.__key
        )
