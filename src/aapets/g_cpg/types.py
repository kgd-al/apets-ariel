import copy
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mujoco
import networkx as nx
import numpy as np
from mujoco import MjSpec, MjData, mj_forward

from abrain import Genome as BrainGenome
from ariel.body_phenotypes.robogen_lite import config as robogen_config
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.ec.genotypes.tree import TreeGenome, operators
from .config import Config, Symmetry
from ..common.controllers.ABCpg import ABCpg, SymmetricalABCPG
from ..common.controllers.abstract import Controller
from ..common.mujoco.state import MjState
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
        symmetry = int(data.config.symmetry is not Symmetry.NONE)
        tree = operators.random_tree(data.config.max_modules // (1 + symmetry))
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

    def render_genotype(self, path: Path, data: StaticData):
        body_path = path.with_suffix(path.suffix + ".body.png")
        _tree_genome_to_dot(self.body, body_path)
        print("Rendered body genotype to", body_path)

        brain_path = path.with_suffix(path.suffix + ".brain.png")
        p = self.brain.to_dot(data, brain_path, ext="png")
        p.rename(brain_path)
        print("Rendered brain genotype to", brain_path)


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
    brain_type: Controller = None

    @property
    def id(self): return self.genome.brain.id()

    @property
    def parents(self): return self.genome.brain.parents()

    def __deepcopy__(self, memo):
        new = copy.copy(self)  # shallow copy first
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            setattr(new, k, copy.deepcopy(v, memo))  # deepcopy all attrs generically
        return new

    @classmethod
    def random(cls, data: StaticData):
        ind = cls(Genome.random(data))
        ind._develop(data)
        return ind

    @staticmethod
    def mutated(ind: 'Individual', data: StaticData):
        child = copy.deepcopy(ind)
        child.mutate(data)
        child.genome.brain.update_lineage(data, parents=[ind.genome.brain])
        child._develop(data)
        return child

    def mutate(self, data: StaticData):
        self.genome.mutate(data)

    @classmethod
    def mated(cls, lhs: 'Individual', rhs: 'Individual', data: StaticData, mutation: float):
        child = cls(Genome.cross(lhs.genome, rhs.genome, data))
        while data.rng.random() < mutation:
            child.mutate(data)
        child._develop(data)
        return child

    def _develop(self, data: StaticData):
        symmetry = data.config.symmetry
        robot = _develop_body(self.genome.body, symmetry)
        # robot = canonical_bodies.get(CanonicalBodies.SPIDER)

        brain = _develop_brain(self.genome.brain, robot, symmetry)

        self.body = robot.spec.to_xml()
        self.weights = brain.extract_weights()
        self.brain_type = brain.__class__

    @classmethod
    def init(cls, config: Config):
        if config.symmetry is not Symmetry.NONE:
            rc = robogen_config
            rc.ALLOWED_FACES[rc.ModuleType.CORE] = [
                rc.ModuleFaces.RIGHT,
                rc.ModuleFaces.BACK,
            ]


def _tree_genome_to_dot(genome: BodyGenome, path: Path):
    styles = {
        "CORE": dict(shape="box3d", style="filled", fillcolor="#4C72B0",
                     fontcolor="white", width=0.7, height=0.7),  # blue cube
        "BRICK": dict(shape="box3d", style="filled", fillcolor="#55A868",
                      fontcolor="white", width=0.45, height=0.45),  # green, smaller cube
        "HINGE": dict(shape="box", style="filled", fillcolor="#C44E52",
                      fontcolor="white", width=0.6, height=0.25),  # red brick
    }
    graph = genome.to_networkx()
    for n, attrs in graph.nodes(data=True):
        graph.nodes[n]["label"] = f"{n}\n{attrs['rotation']}"
        for k, v in styles[attrs["type"]].items():
            graph.nodes[n][k] = v

    for u, v, attrs in graph.edges(data=True):
        graph.edges[u, v]["label"] = attrs["face"]

    dot = nx.nx_pydot.to_pydot(graph)
    dot.set_rankdir("TB")
    dot.write_png(path)


def _develop_body(genome: BodyGenome, symmetry: Symmetry):
    # Flip faces connection
    core_faces_map = {"RIGHT": "FRONT", "BACK": "LEFT"}

    # Flip subtree connections
    local_mirror_map = {"LEFT": "RIGHT", "RIGHT": "LEFT"}

    rc = robogen_config
    graph = genome.to_networkx()

    # draw_graph(graph, save_file="body_graph.base.pdf")
    # print(nx.to_dict_of_dicts(graph))

    if symmetry is not Symmetry.NONE:
        # Apply symmetry by cloning RIGHT -> FRONT and BACK -> LEFT
        next_id = max(graph.nodes) + 1
        core_id = robogen_config.IDX_OF_CORE

        for src_face, dst_face in core_faces_map.items():
            root = next(
                (c for c in graph.successors(core_id)
                 if graph.edges[core_id, c]["face"] == src_face),
                None,
            )
            if root is None:
                continue

            subtree_nodes = nx.descendants(graph, root) | {root}
            depths = nx.shortest_path_length(graph, source=root)
            id_map = {old: next_id + i for i, old in enumerate(subtree_nodes)}
            next_id += len(subtree_nodes)

            for old_id, new_id in id_map.items():
                data = dict(**graph.nodes[old_id])
                if (rotation := data.get("rotation")) is not None:
                    rotation = rc.ModuleRotationsTheta[rotation]
                    rotation = rc.ModuleRotationsTheta((-rotation.value) % 360).name
                    data["rotation"] = rotation
                graph.add_node(new_id, **data)

            for parent, child in graph.subgraph(subtree_nodes).edges:
                face = graph.edges[parent, child]["face"]
                # if False and depths[parent] % 2 == 0:
                #     new_face = face
                # else:
                new_face = local_mirror_map.get(face, face)  # Flip faces, if needed
                graph.add_edge(
                    id_map[parent], id_map[child],
                    face=new_face
                )
            graph.add_edge(core_id, id_map[root], face=dst_face)

    # print(nx.to_dict_of_dicts(graph))
    # draw_graph(graph, save_file="body_graph.symmetrical.pdf")

    robot = construct_mjspec_from_graph(graph)
    robot.spec.body("core").quat = (np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8))
    return robot


def _develop_brain(genome: BrainGenome, robot: CoreModule, symmetry: Symmetry):
    robot_name = "embryo"
    world = make_world(robot.spec.copy(), robot_name=robot_name)
    state, model, data = compile_world(world)
    mujoco.mj_forward(model, data)

    cpg_class = ABCpg if symmetry is not Symmetry.BOTH else SymmetricalABCPG
    brain = cpg_class.from_cppn(genome, state, name=robot_name)

    return brain


def morphological_symmetry(state: MjState, robot_name: str, o_type: Literal["body", "joint"]):
    n, fn, p_attr = {
        "body": (state.model.nbody, MjData.body, "xpos"),
        "joint": (state.model.njnt, MjData.joint, "xanchor"),
    }[o_type]

    class MjSymmetry(defaultdict):
        def __init__(self):
            super().__init__(list)

            mj_forward(state.model, state.data)

            for i in range(n):
                obj = fn(state.data, i)
                name = obj.name
                if not name.startswith(f"{robot_name}") or name.split("_")[-1][0] != "C":
                    continue
                self[self.string_hash(obj)].append(name)

        def valid(self):
            return all(len(p) == 2 for p in self.values())

        @staticmethod
        def string_hash(obj):
            a = getattr(obj, p_attr)
            a[1] = abs(a[1])
            return np.array2string(np.round(a, 3)+0)

    return MjSymmetry()
