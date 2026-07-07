import copy
import pprint
from collections import defaultdict

import itertools
from typing import Sequence, Iterable, Tuple

import numpy as np

from abrain import Genome as CPPNGenome
from abrain import Point3D, CPPN3D
from ...common.controllers import RevolveCPG
from ...common.mujoco.state import MjState


_DEBUG = True
if _DEBUG:
    print(__name__, "is in debug mode")


class ABCpg(RevolveCPG):
    def __init__(
            self,
            *args,
            state: MjState,
            scaling_power: float = 1,
            **kwargs
    ):
        super().__init__(*args, state=state, **kwargs)

        self._alpha, self._beta = 0, 1  # Default to no impact
        self.scaling_power = scaling_power

        self._sides = [
            np.sign(self._joints_pos[actuator.name][1]) for actuator in self._actuators
        ]

        self._verticals = [
            np.allclose(state.data.joint(name).xaxis, [0, 0, 1])
            for name in self._mapping.keys()
        ]

    @classmethod
    def name(cls): return "abcpg"

    @property
    def alpha(self): return self._alpha

    @property
    def beta(self): return self._beta

    def set(self, *, alpha: float, beta: float):
        self._alpha = alpha
        self._beta = beta

    def _set_actuators_states(self):
        lateral_scaling = (1 - abs(self._alpha)) ** self.scaling_power
        global_scaling = abs(self._beta) ** self.scaling_power

        forward = np.sign(self._beta)

        # print(f"{lateral_scaling=}, {forward_scaling=}, {forward=}")

        for i, (actuator, ctrl, side, vertical) in enumerate(zip(
                self._actuators, self._state, self._sides, self._verticals)):

            scaling = global_scaling

            # print(actuator.name, "initial scaling", scaling)
            if self._alpha * side * forward > 0:  # Same side
                scaling *= lateral_scaling
                # print(">>", scaling, f" * {lateral_scaling=}")

            if vertical and self._beta < 0:
                scaling *= -1
                # print(">>", scaling, f" * {forward_scaling=}")

            # scaling = 1 if vertical else -1
            # print(">",  scaling)

            actuator.ctrl[:] = scaling * ctrl * self._ranges[i]


class SymmetricalABCPG(ABCpg):
    """
    This class assumes perfect morphological symmetry along the x axis
    """

    @classmethod
    def name(cls): return "sym_abcpg"

    @classmethod
    def symmetrical_joints(cls, state: MjState, name: str):
        class SymmetricalJoints(defaultdict):
            def valid(self):
                return all(len(p) == 2 for p in self.values())
        positions = SymmetricalJoints(list)
        for name, pos in cls.joints_positions(state, name).items():
            pos[1] = abs(pos[1])
            positions[np.array2string(pos, precision=3)].append(name)
        return positions

    @classmethod
    def num_parameters(cls, state: MjState, name: str, *args, **kwargs) -> int:
        n = cls.num_joints(state, name)
        i = n // 2
        assert 2*i == n, f"{cls.__name__} expects an even number of parameters whereas {n} is odd"
        return (
            i  # C_i internal weight
            + i  # C_i <-> C_j weight where y_i == -y_j and x_i == x_j and z_i == z_j
            + i * (i-1)  # C_i <-> C_j where x_i != x_j or z_i != z_j
        )

    @classmethod
    def from_cppn(cls, genotype: CPPNGenome, state: MjState, name: str):
        joints = cls.joints_positions(state, name)
        n = len(joints)
        n_ = n // 2

        state_size = 2 * n
        _weight_matrix = np.zeros((state_size, state_size))

        cppn = CPPN3D(genotype)
        assert cppn.n_inputs() == 7  # 2*3D + length
        assert cppn.n_outputs() == 2

        weights = []

        joints_pts = {name: Point3D(*pos) for name, pos in joints.items()}

        def sorter(_k):
            _p = joints[_k]
            return _p[1], _p[0], _p[2:]

        names = sorted(list(joints_pts.keys()), key=sorter)

        # if _DEBUG:
        #     for joint_name in names:
        #         print(joint_name, joints_pts[joint_name])

        output = cppn.outputs()

        if _DEBUG and True:
            _debug_value = 0
            def weight():
                nonlocal _debug_value
                _debug_value += 1
                return _debug_value
        else:
            def weight(): return output[0]

        for joint_name in names[:n_]:
            pos = joints_pts[joint_name]
            cppn(pos, pos, output)
            # print(f"CPPN({pos}): {output}")
            # weights.append(debug_value())
            weights.append(weight())

        for i, j in cls.network_indices(n):
            lhs_pos, rhs_pos = joints_pts[names[i]], joints_pts[names[j]]
            cppn(lhs_pos, rhs_pos, output)
            # print(f"CPPN({lhs_pos} -> {rhs_pos}): {output}")
            # weights.append(debug_value())
            weights.append(weight() * bool(output[1]))

        # assert len(weights) == cls.num_parameters(state, name), \
        #     f"Mismatch ({len(joints)} cpgs): {len(weights)} != {cls.num_parameters(state, name)}"

        return cls(weights, state=state, name=name)

    def extract_weights(self) -> np.ndarray:
        n = self.cpgs
        n_ = n // 2
        weights = []
        for i in range(n_):
            weights.append(self._weight_matrix[i][n + i])
        for i, j in self.network_indices(n):
            weights.append(self._weight_matrix[i][j])
        assert len(weights) == self.dimensionality
        return np.array(weights)

    def set_weights(self, weights: Sequence[float]):
        n, m, used = self.cpgs, self._weight_matrix, 0
        n_ = n // 2

        for i, w in enumerate(weights[:n_]):
            m[i][n + i] = +w
            m[n - i - 1][2 * n - i - 1] = +w
            m[n + i][i] = -w
            m[2 * n - i - 1][n - i - 1] = -w
            used += 1

        for (i, j), w in zip(self.network_indices(n), weights[n_:]):
            m[i][j] = +w
            m[j][i] = -w
            if j < n - i - 1:
                m[n - j - 1][n - i - 1] = -w
                m[n - i - 1][n - j - 1] = +w
            used += 1

        if _DEBUG and False:
            np.set_printoptions(linewidth=1000)
            print(used, weights)
            print(m[:n,:n])
            print(m)
            exit(42)

        if used != len(weights):
            raise RuntimeError(f"Unused weights in cpg assignment:"
                               f" {used} used, {len(weights)} provided")

    @staticmethod
    def network_indices(n: int) -> Iterable[Tuple[int, int]]:
        for i in range(n):
            for j in range(i+1, n-i):
                yield i, j

