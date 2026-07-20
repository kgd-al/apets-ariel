import copy
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import matplotlib
import mujoco
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from mujoco import MjSpec, MjData, mj_forward

from abrain import Genome as BrainGenome
from ariel.body_phenotypes.robogen_lite import config as robogen_config
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.ec.genotypes.tree import TreeGenome, operators
from .config import Config, Symmetry
from ..common.controllers.ABCpg import ABCpg, SymmetricalABCPG
from ..common.controllers.abstract import Controller
from ..common.monitors.metrics_storage import BAD, GOOD, RESET
from ..common.monitors.plotters.brain_activity import BrainActivityPlotter
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

    # Helper
    invalid: bool = False

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

        self.invalid = (has_self_collisions(robot.spec, collect=False)
                        or len(self.weights) == 0)

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


def has_self_collisions(spec: MjSpec,
                        prefix="",
                        margin=0.001,
                        collect=False):
    """
    Tests for self collisions in provided MjSpecs.

    Assumes either a robot without worldbody or a robot whose root body is named <prefix>_core.
    In the former case, only geometries starting with <prefix> are considered for collision.

    :param spec: The MjSpec to test
    :param prefix: The name prefix of the robot's bodies/geoms
    :param margin: How lenient to be with collisions
    :param collect: Whether to collect colliding bodies names or to just return a bool
    :return: Whether collision(s) were detected (if collect is false)
     or the list of colliding body names
    """

    spec = MjSpec.from_string(spec.to_xml())
    prefix = ""
    if (core := spec.body("core")) is None:
        core = [b for b in spec.bodies if b.name.endswith("_core")][0]
        prefix = core.name[:-5]
    if core is None:
        raise ValueError("Provided specs should start with a robot's core "
                         "or have a body named <prefix>_core")
    core.quat = (1, 0, 0, 0)

    state, model, data = MjState.from_spec(spec).unpacked
    mj_forward(model, data)

    collisions = []

    def aabb(_g):
        mg, dg = model.geom(_g), data.geom(_g)
        pos, size = dg.xpos, np.abs(dg.xmat.reshape((3, 3))) @ mg.size
        return np.sort(np.array([pos - size, pos + size]), axis=0), model.body(mg.bodyid)

    for i in range(model.ngeom):
        i_aabb, i_body = aabb(i)
        if prefix and not i_body.name.startswith(prefix):
            continue

        for j in range(i+1, model.ngeom):
            j_aabb, j_body = aabb(j)
            if prefix and not j_body.name.startswith(prefix):
                continue

            if i_body.id == j_body.parentid or j_body.id == i_body.parentid:
                continue

            collision = all(i_aabb[0][k] + margin < j_aabb[1][k] - margin
                            and j_aabb[0][k] + margin < i_aabb[1][k] - margin
                            for k in range(3))
            if collision:
                # print(f"{model.geom(i).name}/{model.geom(j).name}:\n"
                #       f"\t{i_aabb}\n\t{j_aabb}\n"
                #       + "\t\t" +
                #       "\n\t\t".join([f"{i_aabb[0][k]+margin} < {j_aabb[1][k]-margin}"
                #                      f" and {j_aabb[0][k]+margin} < {i_aabb[1][k]-margin}"
                #                      for k in range(3)]))
                # print(f"Collision {i}/{j} ({model.geom(i).name}/{model.geom(j).name}):")
                if collect:
                    collisions.append(f"{i}/{j} ({model.geom(i).name}/{model.geom(j).name})")
                else:
                    return True
    # print("No collisions")
    if collect:
        return collisions
    else:
        return False


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
            return np.array2string(np.round(a, 3)+0, precision=3)

        def pretty_print(self):
            """Not quite happy with that, but does the job"""
            for k, v in self.items():
                print(f"  {k}: {GOOD if len(v) == 2 else BAD}{str(v)}{RESET}")

    return MjSymmetry()


def behavioral_symmetry(state: MjState,
                        brain_activity_monitor: BrainActivityPlotter,
                        robot_name,
                        save_plot: Optional[Path] = None,
                        collect=False):

    bam = brain_activity_monitor
    n = len(bam.actuators) // 2
    data = bam.data

    mismatches = []

    x = np.array(data[0])
    actuators = {n: i for i, n in enumerate(bam.actuators.keys())}
    hinges = morphological_symmetry(state, robot_name, "joint")
    if not hinges.valid():
        return 1, ["Non symmetrical hinges"]

    def _ix(_name, _j): return 2 * actuators[_name] + _j + 1

    for i, (pos, names) in enumerate(hinges.items()):
        ixs = [_ix(name, 0) for name in names]
        if any(abs(lhs) != abs(rhs) for lhs, rhs in zip(data[ixs[0]], data[ixs[1]])):
            mismatches.append(names)

    if save_plot:
        w, h = matplotlib.rcParams["figure.figsize"]

        y_lim = np.ceil(1.1 * bam.max_range)
        fig, axes = plt.subplots(n, 2,
                                 sharex=True, sharey=True,
                                 figsize=(3 * w, .25 * n * h))

        for i, (pos, names) in enumerate(hinges.items()):
            for j, label in enumerate(["Position", "Control"]):
                ax = axes[i][j]
                ixs = []
                for name in names:
                    ix = _ix(name, j)
                    ixs.append(ix)
                    ax.plot(x, data[ix], zorder=1)
                ax.set_ylim(-y_lim, y_lim)

                title = f"{pos}: {names}"
                if i == 0:
                    title = f"{label}\n\n" + title

                ax.set_title(title)

                if (j == 1 and
                        any(abs(lhs) != abs(rhs) for lhs, rhs in zip(data[ixs[0]], data[ixs[1]]))):
                    mismatches.append(names)

        fig.tight_layout()
        fig.savefig(save_plot, bbox_inches="tight")
        print("Saved symmetrical brain activity to", save_plot)
        plt.close(fig)

        # bam.plot("brain_activity.base.pdf")
    if collect:
        return mismatches
    else:
        return len(mismatches) == 0
