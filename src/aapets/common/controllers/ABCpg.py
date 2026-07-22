import copy
import pprint
from collections import defaultdict

import itertools
from typing import Sequence, Iterable, Tuple

import numpy as np

from abrain import Genome as CPPNGenome
from abrain import Point3D, CPPN3D
from ..misc.debug import kgd_debug
from ...common.controllers import RevolveCPG
from ...common.mujoco.state import MjState


_DEBUG = False
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Just need to sort actuators
        keys = list(self._joints_pos.keys())
        indices = sorted(range(len(keys)),
                         key=lambda _k: self.sort_by_pos(self._joints_pos[keys[_k]]))
        self._actuators = [self._actuators[i] for i in indices]
        self._verticals = [self._verticals[i] for i in indices]
        # if _DEBUG or True:
        #     kgd_debug("Actuators details:")
        #     pprint.pprint([(a.name, np.round(self._joints_pos[a.name], 3), v)
        #                    for a, v in zip(self._actuators, self._verticals)])

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

        # kgd_debug("Zeroing out secondary diagonal")
        # for i in range(n):
        #     m[n-i-1, i] *= 0

        # kgd_debug("Disabling weight length check")
        if used != len(weights):
            raise RuntimeError(f"Unused weights in cpg assignment:"
                               f" {used} used, {len(weights)} provided")

    def _set_actuators_states(self):
        super()._set_actuators_states()
        n_ = self.cpgs // 2
        for a, v in zip(self._actuators[n_:], self._verticals[n_:]):
            if v:
                a.ctrl[:] *= -1

    @classmethod
    def name(cls): return "sym_abcpg"

    @classmethod
    def num_parameters(cls, state: MjState, name: str, *args, **kwargs) -> int:
        n = cls.num_joints(state, name)
        i = n // 2
        assert 2*i == n, f"{cls.__name__} expects an even number of parameters whereas {n} is odd"
        return (
            i  # C_i internal weight
            # + i  # C_i <-> C_j weight where y_i == -y_j and x_i == x_j and z_i == z_j
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

        names = sorted(list(joints_pts.keys()), key=lambda _k: cls.sort_by_pos(joints[_k]))

        output = cppn.outputs()

        if _DEBUG and False:
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
            weights.append(weight() * bool(output[1]) + 0)

        assert len(weights) == cls.num_parameters(state, name), \
            f"Mismatch ({len(joints)} cpgs): {len(weights)} != {cls.num_parameters(state, name)}"

        return cls(weights, state=state, name=name)

    @staticmethod
    def sort_by_pos(p):
        if p[1] < 0:
            a = p[1], p[0], p[2]
        else:
            a = p[1], -p[0], -p[2]
        # TODO: Remember to hate python typelessness
        return np.round(a, 3).tolist()

    @staticmethod
    def network_indices(n: int) -> Iterable[Tuple[int, int]]:
        for i in range(n-1):
            for j in range(i+1, n-i-1):
                yield i, j
