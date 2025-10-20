import abc
import copy
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import TypeVar, Type, Generic

import numpy as np
import numpy.typing as npt
import torch
from abrain import Genome as CPPNGenome
from torch import nn

from ariel.body_phenotypes.robogen_lite.config import NUM_OF_TYPES_OF_MODULES, NUM_OF_FACES, NUM_OF_ROTATIONS
from config import EvoConfig


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


class NeuralDevelopmentalEncoding(nn.Module):
    def __init__(self, number_of_modules: int, input_size: int) -> None:
        super().__init__()
        # Hidden Layers
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 128)

        # ------------------------------------------------------------------- #
        # OUTPUTS
        self.type_p_shape = (number_of_modules, NUM_OF_TYPES_OF_MODULES)
        self.type_p_out = nn.Linear(
            128,
            number_of_modules * NUM_OF_TYPES_OF_MODULES,
        )

        self.conn_p_shape = (number_of_modules, number_of_modules, NUM_OF_FACES)
        self.conn_p_out = nn.Linear(
            128,
            number_of_modules * number_of_modules * NUM_OF_FACES,
        )

        self.rot_p_shape = (number_of_modules, NUM_OF_ROTATIONS)
        self.rot_p_out = nn.Linear(
            128,
            number_of_modules * NUM_OF_ROTATIONS,
        )

        self.output_layers = [
            self.type_p_out,
            self.conn_p_out,
            self.rot_p_out,
        ]
        self.output_shapes = [
            self.type_p_shape,
            self.conn_p_shape,
            self.rot_p_shape,
        ]
        # ------------------------------------------------------------------- #

        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Disable gradients for all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(
        self,
        genotype: list[npt.NDArray[np.float32]],
    ) -> list[npt.NDArray[np.float32]]:
        outputs: list[npt.NDArray[np.float32]] = []
        for idx, chromosome in enumerate(genotype):
            with torch.no_grad():  # double safety
                x = torch.from_numpy(chromosome).to(torch.float32)

                x = self.fc1(x)
                x = self.relu(x)

                x = self.fc2(x)
                x = self.relu(x)

                x = self.fc3(x)
                x = self.relu(x)

                x = self.fc4(x)
                x = self.relu(x)

                x = self.output_layers[idx](x)
                x = self.sigmoid(x)

                x = x.view(self.output_shapes[idx])
                outputs.append(x.detach().numpy())
        return outputs


class Genotype(GenericGenotype):
    __key = object()

    @dataclass
    class Data:
        body: "Genotype.BodyGenotype.Data"
        brain: CPPNGenome.Data
        config: EvoConfig

        def __init__(self, config: EvoConfig, seed=None):
            size = config.body_genotype_size or 64
            decoder = NeuralDevelopmentalEncoding(
                number_of_modules=config.max_modules,
                input_size=size
            )
            self.body = Genotype.BodyGenotype.Data(
                rng=np.random.default_rng(seed),
                size=size, fields=len(decoder.output_layers),
                decoder=decoder
            )
            self.brain = CPPNGenome.Data.create_for_eshn_cppn(
                dimension=3, seed=seed,
                with_input_bias=True, with_input_length=True,
                with_leo=True,
                with_innovations=True, with_lineage=True)
            self.config = config

    @dataclass
    class BodyGenotype:
        @dataclass
        class Data:
            rng: np.random.Generator
            size: int
            fields: int

            decoder: NeuralDevelopmentalEncoding

        data: list[npt.NDArray[np.float32]]

        @classmethod
        def random(cls, data: "Genotype.BodyGenotype.Data") -> "Genotype.BodyGenotype":
            return cls([data.rng.random(data.size, np.float32) for _ in range(data.fields)])

        def mutate(self, data: "Genotype.BodyGenotype.Data", mutation_rate: float) -> None:
            rate = 1
            while data.rng.random() <= rate:
                self.data[data.rng.integers(data.fields)][data.rng.integers(data.size)] += data.rng.normal(0, 1)
                rate *= mutation_rate

        @classmethod
        def crossover(cls,
                      lhs: "Genotype.BodyGenotype", rhs: "Genotype.BodyGenotype",
                      data: "Genotype.BodyGenotype.Data"):
            child_data = []
            for lhs_field, rhs_field in zip(lhs.data, rhs.data):
                i = data.rng.integers(data.size)
                child_data.append(lhs_field[:i] + rhs_field[i:])
            return cls(child_data)

        def copy(self) -> "Genotype.BodyGenotype":
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
            body=cls.BodyGenotype.random(data.body),
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
            body=cls.BodyGenotype.crossover(lhs.body, rhs.body, data.body),
            brain=CPPNGenome.crossover(lhs.brain, rhs.brain, data.brain),
            _key=cls.__key
        )

    def copy(self):
        return Genotype(
            body=self.body.copy(),
            brain=self.brain.copy(),
            _key=self.__key
        )
