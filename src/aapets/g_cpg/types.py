from typing import List, Tuple

import copy
import numpy as np
from dataclasses import dataclass, field
from mujoco import MjSpec

from abrain import Genome as BrainGenome
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.ec.genotypes.tree import TreeGenome, operators
from .config import Config
from ..common.controllers.ABCpg import ABCpg
from ..common.world_builder import make_world, compile_world


@dataclass
class StaticData(BrainGenome.Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = None

    def set_config(self, config: Config):
        self.config = config


class BodyGenome(TreeGenome):
    @classmethod
    def random(cls, data: StaticData) -> 'BodyGenome':
        tree = operators.random_tree(data.config.max_modules)
        tree.__class__ = cls
        return tree

    def mutate(self, data: StaticData):
        op = data.rng.choice([
            operators.mutate_replace_node,  # change a node's type/rotation
            operators.mutate_subtree_replacement,  # GP subtree mutation (Koza 1992)
            operators.mutate_shrink,  # replace subtree with single leaf
            operators.mutate_hoist,  # promote a child, drop its siblings
        ])
        genome = copy.deepcopy(self)  # don't mutate in-place for DEAP compat
        op(genome)
        return genome

    @staticmethod
    def crossover(lhs: 'BodyGenome', rhs: 'BodyGenome', data: StaticData):
        return operators.crossover_subtree(lhs, rhs)[0]

    def clone(self) -> 'BodyGenome':
        return copy.deepcopy(self)


@dataclass
class Genome:
    body: BodyGenome
    brain: BrainGenome

    @classmethod
    def random(cls, data: StaticData):
        brain = BrainGenome.random(data)
        for i in range(data.config.initial_mutations_brain):
            brain.mutate(data)
        return cls(BodyGenome.random(data), brain)

    def mutate(self, data: StaticData):
        self.body.mutate(data)
        self.brain.mutate(data)

    def cross(self, other: 'Genome', data: StaticData):
        return self.__class__(
            BodyGenome.crossover(self.body, other.body, data),
            BrainGenome.crossover(self.brain, other.brain, data),
        )


class CopyableSpec(MjSpec):
    def __deepcopy__(self, memo):
        new = CopyableSpec.from_string(self.spec.to_xml())
        memo[id(self)] = new
        return new


@dataclass
class Individual:
    # Genotype
    genome: Genome

    # Phenotype
    body: str = None
    weights: np.ndarray = None

    @property
    def id(self): return self.genome.brain.id()

    @property
    def parents(self): return self.genome.brain.parents()

    def __post_init__(self):
        self._develop()

    def __deepcopy__(self, memo):
        new = copy.copy(self)  # shallow copy first
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, copy.deepcopy(v, memo))  # deepcopy all attrs generically
        new._develop()
        return new

    @classmethod
    def random(cls, data: StaticData):
        return cls(Genome.random(data))

    @staticmethod
    def mutated(ind: 'Individual', data: StaticData):
        child = copy.deepcopy(ind)
        child.mutate(data)
        child.genome.brain.update_lineage(data, parents=[ind.genome.brain])
        child._develop()
        return child

    def mutate(self, data: StaticData):
        self.genome.mutate(data)

    @classmethod
    def mated(cls, lhs: 'Individual', rhs: 'Individual', data: StaticData, mutation: float):
        child = cls(Genome.cross(lhs.genome, rhs.genome, data))
        while data.rng.random() < mutation:
            child.mutate(data)
        child._develop()
        return child

    def _develop(self):
        robot_name = "embryo"
        robot = construct_mjspec_from_graph(self.genome.body.to_networkx())
        robot.spec.body("core").quat = (np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8))
        # robot = canonical_bodies.get(CanonicalBodies.SPIDER)
        world = make_world(robot.spec.copy(), robot_name=robot_name)
        state, _, _ = compile_world(world)
        brain = ABCpg.from_cppn(self.genome.brain, state, name=robot_name)

        self.body = robot.spec.to_xml()
        self.weights = brain.extract_weights()
