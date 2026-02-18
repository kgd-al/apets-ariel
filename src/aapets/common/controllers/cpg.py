import itertools
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import graphviz
import numpy as np
import numpy.typing as npt
from abrain import Genome as CPPNGenome
from abrain import Point3D, CPPN3D
from mujoco import mjtIntegrator

from .abstract import Controller
from ..mujoco.state import MjState


class RevolveCPG(Controller):
    """Copied from revolve but fully connected"""

    @classmethod
    def name(cls): return "cpg"

    def __init__(
            self,
            weights: Sequence[float],
            state: MjState,
            name: str):

        super().__init__(weights, state, name)

        if state.model.opt.integrator is mjtIntegrator.mjINT_RK4:
            raise NotImplementedError(
                f"Controller {__name__} does not work with RK4 integrator"
            )

        self.cpgs = len(self._joints_pos)

        self._weight_matrix = self.make_weights_matrix(weights, state, name)

        self._initial_state = (
                np.hstack([np.full(self.cpgs, 1), np.full(self.cpgs, -1)])
                * 0.5 * np.sqrt(2)
        )
        self._state = self._initial_state.copy()

        self._time = state.data.time  # To measure dt

    @classmethod
    def num_parameters(cls, state: MjState, name: str, *args, **kwargs) -> int:
        return cls.num_joints(state, name) ** 2

    @classmethod
    def from_weights(cls, weights: Sequence[float], state: MjState, name: str):
        return cls(weights, state, name)

    @classmethod
    def random(cls, state: MjState, name: str, seed: int = None):
        n = cls.compute_dimensionality(state, name)
        return cls(
            np.random.default_rng(seed).uniform(-1, 1, n),
            state, name
        )

    @classmethod
    def from_cppn(cls, genotype: CPPNGenome, state: MjState, name: str):
        joints = cls.joints_positions(state, name)

        state_size = 2 * len(joints)
        _weight_matrix = np.zeros((state_size, state_size))

        cppn = CPPN3D(genotype)
        assert cppn.n_inputs() == 7  # 2*3D + length
        assert cppn.n_outputs() == 2

        weights = []

        joints_pts = {name: Point3D(*pos) for name, pos in joints.items()}
        output = cppn.outputs()
        for name, pos in joints_pts.items():
            cppn(pos, pos, output)
            weights.append(output[0])

        for lhs_name, lhs_pos in joints_pts.items():
            for rhs_name, rhs_pos in joints_pts.items():
                if lhs_name == rhs_name:
                    continue
                cppn(lhs_pos, rhs_pos, output)
                weights.append(output[0] * bool(output[1]))

        return RevolveCPG(weights, state, name)

    @property
    def actuators(self): return self._actuators

    @property
    def ranges(self): return self._ranges

    def extract_weights(self) -> np.ndarray:
        n = self.cpgs
        weights = []
        for i in range(n):
            weights.append(self._weight_matrix[i][n + i])
        for i, j in itertools.product(range(n), range(n)):
            if i != j:
                weights.append(self._weight_matrix[i][j])
        assert len(weights) == self.dimensionality
        return np.array(weights)

    def make_weights_matrix(self, weights: Sequence[float], state: MjState, name: str):
        # assert len(weights) == RevolveCPG.compute_dimensionality(n), \
        #     f"Need {RevolveCPG.compute_dimensionality(n)} values, got {len(weights)}"

        n = self.cpgs
        state_size = 2 * n
        _weight_matrix = np.zeros((state_size, state_size))

        for i, w in enumerate(weights[:n]):
            _weight_matrix[i][n + i] = +w
            _weight_matrix[n + i][i] = -w

        for (i, j), w in zip(
            [(i, j) for i, j in itertools.product(range(n), range(n)) if i != j],
            weights[n:],
            strict=True
        ):
            _weight_matrix[i][j] = w

        # with np.printoptions(precision=1, linewidth=400):
        #     print(_weight_matrix)

        return _weight_matrix

    @staticmethod
    def _rk45(state, a, dt: float):
        """
        Calculate the next state using the RK45 method.

        This implementation of the Runge–Kutta–Fehlberg method allows us to improve accuracy of state calculations by comparing solutions at different step sizes.
        For more info see: See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method.
        RK45 is a method of order 4 with an error estimator of order 5 (Fehlberg, E. (1969). Low-order classical Runge-Kutta formulas with stepsize control. NASA Technical Report R-315.).

        :param state: The current state of the network.
        :param A: The weights matrix of the network.
        :param dt: The step size (elapsed simulation time).
        :return: The new state.
        """
        a1 = np.matmul(a, state)
        a2 = np.matmul(a, (state + dt / 2 * a1))
        a3 = np.matmul(a, (state + dt / 2 * a2))
        a4 = np.matmul(a, (state + dt * a3))
        state = state + dt / 6 * (a1 + 2 * (a2 + a3) + a4)
        return np.clip(state, a_min=-1, a_max=1)

    def __call__(self, state: MjState) -> None:
        dt = state.data.time - self._time

        self._state = self._rk45(self._state, self._weight_matrix, dt)
        self._set_actuators_states()

        self._time = state.data.time

    def _set_actuators_states(self):
        for i, (actuator, ctrl) in enumerate(zip(self._actuators, self._state)):
            actuator.ctrl[:] = ctrl * self._ranges[i]

    def render_phenotype(self, path: Path, scale=1000, *args, **kwargs):
        weights = self._weight_matrix.copy()

        dot = graphviz.Digraph(
            "CPG", "connectivity pattern",
            engine="neato",
            graph_attr=dict(overlap="scale", outputorder="edgesfirst", splines="true"),
            node_attr=dict(style="filled", fillcolor="white"),
            # edge_attr=dict(dir="both")
        )

        w_max = abs(weights).max()
        weights /= w_max

        overlaps = defaultdict(lambda: 0)
        o_scale = .1 * scale

        for name, index in self._mapping.items():
            style = dict(shape="box")

            x, y, _ = self._joints_pos[name]
            x, y = x * scale, y * scale

            o_key = (round(x), round(y))
            overlap = overlaps[o_key]
            if overlap > 0:
                x, y = x + overlap * o_scale, y + overlap * o_scale
                style["color"] = "red"

            pos = f"{x},{y}"

            label = "".join(name.split("_")[-1].split("-")[:-1])
            dot.node(name=str(index), label=label, pos=pos, **style)

            overlaps[o_key] += 1

        n = len(self._joints_pos)
        for i, j in itertools.product(range(n), repeat=2):
            w = weights[i, j]
            if w != 0:
                style = dict(
                    color="black" if w > 0 else "red",
                    penwidth=str(2 + np.log(abs(w)))
                )
                dot.edge(str(i), str(j), **style)

        # print(path)
        # print(dot.source)
        dot.render(outfile=path, neato_no_op=2, cleanup=True)
