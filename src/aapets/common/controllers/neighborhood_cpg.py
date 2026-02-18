import itertools
from dataclasses import dataclass, field
from typing import Sequence, Optional

import numpy as np
from mujoco._structs import _MjModelBodyViews

from . import RevolveCPG
from ..mujoco.state import MjState


class NeighborhoodCPG(RevolveCPG):
    """ Allow providing a maximal distance to connect CPGs """

    def __init__(
            self,
            weights: Sequence[float],
            state: MjState,
            name: str,
            neighborhood: int,
    ):
        self.neighborhood = neighborhood

        super().__init__(weights=weights, state=state, name=name)

    @classmethod
    def name(cls): return "knn_cpg"

    @classmethod
    def num_parameters(cls, state: MjState, name_prefix: str, neighborhood: int, *args, **kwargs) -> int:
        matrix = cls.distance_matrix(state, name_prefix)
        return np.tril(matrix <= neighborhood).sum()

    def make_weights_matrix(self, weights: Sequence[float], state: MjState, name: str):
        # assert len(weights) == RevolveCPG.compute_dimensionality(n), \
        #     f"Need {RevolveCPG.compute_dimensionality(n)} values, got {len(weights)}"

        connectivity_matrix = (self.distance_matrix(state, name) <= self.neighborhood)
        n = connectivity_matrix.shape[0]

        state_size = 2 * n
        _weight_matrix = np.zeros((state_size, state_size))

        for i, w in enumerate(weights[:n]):
            _weight_matrix[i][n + i] = +w
            _weight_matrix[n + i][i] = -w

        for (i, j), w in zip(
            [(i, j) for i, j in itertools.product(range(n), range(n))
             if i < j and connectivity_matrix[i, j]],
            weights[n:],
            strict=True
        ):
            _weight_matrix[i][j] = w
            _weight_matrix[j][i] = w

        # with np.printoptions(precision=1, linewidth=400):
        #     print(_weight_matrix)

        return _weight_matrix

    @classmethod
    def distance_matrix(cls, state: MjState, name_prefix: str) -> np.ndarray:
        bodies = [
            state.model.body(name) for i in range(state.model.nbody)
            if len((name := state.data.body(i).name)) > 0 and name_prefix in name
        ]
        bodies = {b.id: b for b in bodies}

        nodes = {bid: _Node(bid, body) for bid, body in bodies.items()}
        root = None

        # First collect all links
        for bid, node in nodes.items():
            if (parent := nodes.get(int(bodies[bid].parentid), None)) is not None:
                node.parent = parent
                parent.children.append(node)
            else:
                assert root is None
                root = node

        # Then simplify
        assert root.body.name[-5:] == "world"
        root = root.children[0]  # Use core and not world as the tree top
        root.parent = None
        assert root.body.name[-4:] == "core"

        # Merge hinge, stator and rotors
        hinges = [(bid, n) for bid, n in nodes.items() if n.body.name[-5:] == "hinge"]
        for hinge_id, hinge in hinges:
            hinge.children = hinge.children[1].children
            for c in hinge.children:
                c.parent = hinge

        # print(root)

        # Compute distance matrix
        hi = {i: h for i, (_, h) in enumerate(hinges)}
        matrix = np.zeros((len(hinges), len(hinges)))
        for i in range(len(hi)):
            for j in range(i + 1, len(hi)):
                matrix[i, j] = matrix[j, i] = hi[i].distance(hi[j])

        return matrix


@dataclass
class _Node:
    id: int
    body: _MjModelBodyViews
    parent: 'Optional[_Node]' = None
    children: 'list[_Node]' = field(default_factory=list)

    def distance(self, other: '_Node'):
        if self.id == other.id:
            return 0

        def _next(_n, _d):
            r = [(_d+1, c) for c in _n.children]
            if _n.parent is not None:
                r.append((_d+1, _n.parent))
            return r

        seen = {self.id}
        queue = _next(self, 0)
        while len(queue) > 0:
            d, n = queue.pop()
            if n.id not in seen:
                seen.add(n.id)
                if n.id == other.id:
                    return d
                queue.extend(_next(n, d))

        raise KeyError(f"Could not find distance between {self.body.name} and {other.body.name}")

    def __repr__(self, _depth=0):
        pid = self.parent.id if self.parent is not None else "/"
        _repr = f"{_depth * '  '} [{pid} > {self.id}] {self.body.name}\n"
        for _child in self.children:
            _repr += f"{_child.__repr__(_depth + 1)}"
        return _repr
