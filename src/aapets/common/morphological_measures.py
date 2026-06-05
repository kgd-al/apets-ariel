"""MorphologicalMeasures class."""
import collections
from collections import defaultdict
from itertools import product
from typing import Generic, TypeVar, Tuple, List, Optional, Callable, Type

import numpy as np
from mujoco import MjSpec, MjsBody, MjsGeom, mj_forward
from mujoco._structs import _MjDataBodyViews
from numpy.typing import NDArray

from aapets.common.mujoco.state import MjState
from ariel.body_phenotypes import robogen_lite
from ariel.body_phenotypes.robogen_lite.modules.module import Module
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule as Core
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule as Brick
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule as Hinge
from ariel.simulation.environments import BaseWorld


def _mn(module: Type[Module]): return module.module_type.name.lower()


core = _mn(Core)
brick = _mn(Brick)
hinge = _mn(Hinge)


def measure(specs: MjSpec, robot: str = ""):
    return _MorphologicalMeasures(specs.copy(), robot)


class _Node:
    def __init__(self, body: MjsBody, parent: Optional['_Node'] = None):
        self.body = body
        self.parent = parent

        self.bodies = {b.name: b for b in body.bodies}

        self.geoms = []
        if self.module_type != hinge:
            self.geoms = [body.geoms[0]]
            assert len(body.geoms) == 1, f"{body.name}: {body.geoms}"
            assert self.geoms[0].name == body.name

        else:
            assert len(body.bodies) == 2, f"{body.name}: {body.bodies}"
            for b in body.bodies:
                assert len(b.geoms) == 1
                self.geoms.append(b.geoms[0])
            body = body.bodies[1]

        self.children = [self.__class__(b, self) for b in body.bodies]

    @classmethod
    def make_tree(cls, specs: MjSpec, robot: str):
        root = specs.body(core)
        if root is not None:
            return cls(root)

        root = specs.body(f"{robot}1_world")
        if root is not None:
            return cls(root.bodies[0])

        raise ValueError(f"Could find neither robot 'core' nor {robot}1_world")

    def _to_str(self, depth):
        return "\n".join([
            (">" * depth) + (" " if depth > 0 else "") + self.body.name
        ] + [
            child._to_str(depth=depth+1)
            for child in self.children
        ])

    def __repr__(self):
        return self._to_str(depth=0)

    @property
    def module_type(self):
        return self.body.name.split("-")[-1]

    def accumulate(self, operator: Callable, value):
        value = operator(self, value)
        for c in self.children:
            value = c.accumulate(operator, value)
        return value


class _MorphologicalMeasures:
    """
    Modular robot morphological measures. Rewritten from Revolve's implementation to use MjModel directly

    Only works for robot with only right angle module rotations (90 degrees).
    Some measures only work for 2d robots, which is noted in their docstring.

    The measures are based on the following paper:
    Miras, K., Haasdijk, E., Glette, K., Eiben, A.E. (2018).
    Search Space Analysis of Evolvable Robot Morphologies.
    In: Sim, K., Kaufmann, P. (eds) Applications of Evolutionary Computation.
    EvoApplications 2018. Lecture Notes in Computer Science(), vol 10784. Springer, Cham.
    https://doi.org/10.1007/978-3-319-77538-8_47
    """

    def __init__(self, specs: MjSpec, robot: str):
        mjs_tree = _Node.make_tree(specs, robot)
        # print(robot, specs.to_xml())
        # print(mjs_tree)

        state, model, data = MjState.from_spec(specs).unpacked
        mj_forward(model, data)

        # ----
        def get_aabb(n: _Node, value):
            for geom in n.geoms:
                # Global position of the geometry (x, y, z)
                pos = data.geom_xpos[geom.id]

                # World rotation matrix (flat 9 values in row-major)
                r_mat = np.array(data.geom_xmat[geom.id]).reshape(3, 3)

                # AABB
                local_size = model.geom_aabb[geom.id][3:]
                rotated_size = local_size @ r_mat
                corners = np.array([
                    pos + np.array([dx, dy, dz]) * rotated_size
                    for dx in [-1, 1] for dy in [-1, 1] for dz in [-1, 1]
                ])

                # Return the lowest Z value
                value[0, :] = np.minimum(value[0, :], corners.min(axis=0))
                value[1, :] = np.maximum(value[1, :], corners.max(axis=0))
            return value

        self._aabb = mjs_tree.accumulate(
            operator=get_aabb,
            value=np.array([np.full(3, np.inf), np.full(3, -np.inf)])
        )

        # Return the lowest position rounded to avoid floating point issues
        assert np.isfinite(self._aabb).all(), f"Non-finite values in AABB: {self._aabb}"
        self._aabb = np.round(self._aabb, 6)

        # ----
        def count_modules(n: _Node, value):
            value[n.module_type] += 1
            return value
        self._module_counts = mjs_tree.accumulate(operator=count_modules, value=defaultdict(int))

        # ----
        attachment_counts = {_mn(m): len(m().sites) for m in [Core, Brick, Hinge]}

        def test_filled(n: _Node, value):
            if (_n := attachment_counts.get(n.module_type)) is not None:
                if len(n.children) == _n:
                    value[n.module_type] += 1
            return value
        self._filled_modules = mjs_tree.accumulate(operator=test_filled, value=defaultdict(int))

        # ----
        def find_leaves(n: _Node, value):
            if len(n.children) == 0:
                value[n.module_type] += 1
            return value
        self._leaves = defaultdict(int, mjs_tree.accumulate(operator=find_leaves, value=defaultdict(int)))

        # ----
        def find_double_connected(n: _Node, value):
            if n.module_type != core and len(n.children) == 1:
                value[n.module_type] += 1
            return value
        self._double_connected = defaultdict(int, mjs_tree.accumulate(
            operator=find_double_connected, value=defaultdict(int)))

        # ----
        def compute_volume(n: _Node, value):
            return value + sum(model.geom_aabb[g.id][3:].prod() for g in n.geoms)
        self._total_volume = float(mjs_tree.accumulate(operator=compute_volume, value=0))
        self._aabb_volume = float((self._aabb[1] - self._aabb[0]).prod())

        # ---
        all_geoms = {
            tuple(data.geom(g.id).xpos.round(3).tolist()): g.name.split("-")[-1]
            for g in mjs_tree.body.find_all("geom")
        }

        def _measure_symmetry(dim: int):
            potential_axes = {abs(p[dim]) for p in all_geoms.keys()}
            scores = {}

            for axis in potential_axes:
                ags = all_geoms.copy()
                matches, misses, centered = 0, 0, 0
                while len(ags) > 0:
                    item_pos = next(iter(ags.keys()))
                    item_name = ags.pop(item_pos)
                    if item_pos[dim] == axis:
                        centered += 1
                        continue

                    sym_item = list(item_pos)
                    sym_item[dim] -= 2 * (item_pos[dim] - axis)
                    if (sym_item_name := ags.pop(tuple(sym_item), None)) is None:
                        misses += 1
                    else:
                        if item_name == sym_item_name:
                            matches += 2
                        else:
                            misses += 2
                score = (matches + centered) / (misses + matches + centered)
                # print(f"[kgd-debug] Symmetry[{'xyz'[dim]};{axis=}]: {100 * score}"
                #       f" ({matches=}, {misses=} {centered=})")
                assert misses + matches + centered == len(all_geoms), \
                    f"{misses=} + {matches=} + {centered=} != {len(all_geoms)=}"
                scores[axis] = score
            return max(scores.values())

        self._symmetry = dict(
            xy=_measure_symmetry(dim=2),
            xz=_measure_symmetry(dim=1),
            yz=_measure_symmetry(dim=0),
        )

        # ---

    @property
    def all_metrics(self):
        return dict(
            aabb=self.aabb,
            sizes=dict(width=self.width, depth=self.depth, height=self.height),
            modules=dict(total=self.modules, hinges=self.hinges, bricks=self.bricks),
            filled=dict(
                total=self.filled_core+self.filled_hinges+self.filled_bricks,
                core=self.filled_core, hinges=self.filled_hinges, bricks=self.filled_bricks
            ),
            branching=self.branching, limbs=self.limbs,
            leaves=self.leaves, double_connected=self.double_connected,
            length_of_limbs=self.length_of_limbs,
            volume=dict(
                aabb=self._aabb_volume, total=self._total_volume,
                ratio=self.coverage
            ),
            proportion_2d=self.proportion_2d,
            symmetry=dict(
                min=min(self.symmetries.values()),
                max=max(self.symmetries.values()),
                **self.symmetries
            )
        )

    @property
    def major_metrics(self):
        return dict(
            branching=self.branching,
            limbs=self.limbs,
            length_of_limbs=self.length_of_limbs,
            coverage=self.coverage,
            joints=self.joints,
            proportion=self.proportion_2d,
            symmetry=self.symmetry,
        )

    @property
    def aabb(self): return self._aabb

    @property
    def width(self): return self._size(0)

    @property
    def depth(self): return self._size(1)

    @property
    def height(self): return self._size(2)

    def _size(self, dim: int): return float(self._aabb[1, dim] - self._aabb[0, dim])

    @property
    def hinges(self): return self._module_counts[hinge]

    @property
    def bricks(self): return self._module_counts[brick]

    @property
    def modules(self): return 1 + self.hinges + self.bricks

    @property
    def filled_core(self): return self._filled_modules[core]

    @property
    def filled_bricks(self): return self._filled_modules[brick]

    @property
    def filled_hinges(self): return self._filled_modules[hinge]

    @property
    def max_fillable_core_and_bricks(self):
        if not robogen_lite.config.printable:
            return 0

        return min(max(0, (self.modules - 2) // 3), 1 + self.bricks)

    @property
    def branching(self):
        if self.max_fillable_core_and_bricks == 0:
            return 0
        return (self.filled_bricks + self.filled_core) / self.max_fillable_core_and_bricks

    @property
    def leaves(self): return len(self._leaves)

    @property
    def max_leaves(self):
        return self.modules - 1 - max(0, (self.modules - 3) // 3)

    @property
    def limbs(self):
        if self.max_leaves == 0:
            return 0
        return self.leaves / self.max_leaves

    @property
    def double_connected(self): return self._double_connected

    @property
    def max_double_connected(self):
        return max(0, self.bricks + self.hinges - 1)

    @property
    def length_of_limbs(self):
        if self.max_double_connected == 0:
            return 0

        return (self.double_connected[brick] + self.double_connected[hinge]) / self.max_double_connected

    @property
    def joints(self):
        # return self.double_connected[hinge] / max(0, (self.modules - 1) // 2)
        return self.double_connected[hinge] / max(0, (self.modules - 1))

    @property
    def coverage(self): return self._total_volume / self._aabb_volume

    @property
    def proportion_2d(self):
        return min(self.depth, self.width) / max(self.depth, self.width)

    @property
    def symmetries(self): return self._symmetry

    @property
    def symmetry(self): return max(self.symmetries.values())
