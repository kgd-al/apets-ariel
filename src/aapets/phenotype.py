import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any

import graphviz
import numpy as np
import numpy.typing as npt
from abrain import Genome as CPPNGenome, CPPN3D, Point3D
from mujoco import MjModel, MjData, mjtIntegrator
from networkx import DiGraph

from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from .config import CommonConfig
from .genotype import Genotype
from .misc.debug import kgd_debug

JointsDict = dict[str, tuple[float, float, float]]


def decode_body(genotype: Genotype.Body, config: CommonConfig) -> DiGraph:
    # System parameters
    num_modules = config.max_modules

    kgd_debug(f"nde decoder: {config.nde_decoder}")
    (type_probability_space,
     conn_probability_space,
     rotation_probability_space) = config.nde_decoder.forward(genotype.data)
    kgd_debug("ping")

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(num_modules)
    kgd_debug("ping")
    return hpd.probability_matrices_to_graph(
        type_probability_space,
        conn_probability_space,
        rotation_probability_space,
    )


def decode_brain(genotype: CPPNGenome,
                 model: MjModel, data: MjData, config: CommonConfig) -> Any:
    return RevolveCPG(genotype, model, data, config)


class RevolveCPG:
    """Copied from revolve but fully connected"""

    _initial_state: npt.NDArray[float]
    _weight_matrix: npt.NDArray[float]  # nxn matrix matching number of neurons
    # _output_mapping: list[tuple[int, ActiveHinge]]

    def __init__(self, genotype: CPPNGenome,
                 model: MjModel, data: MjData, config: CommonConfig) -> None:

        if model.opt.integrator is mjtIntegrator.mjINT_RK4:
            raise NotImplementedError(
                f"Controller {__name__} does not work with RK4 integrator"
            )

        self._joints_pos = {
            name: data.joint(i).xanchor for i in range(model.njnt)
            if len((name := data.joint(i).name)) > 0
        }

        self._mapping = {
            name: i for i, name in enumerate(self._joints_pos.keys())
        }

        self.cpgs = len(self._joints_pos)
        (self._state,
         self._initial_state,
         self._weight_matrix) = self.make_state_and_weights(genotype, self._joints_pos)

        self._actuators = [
            data.actuator(name) for name in self._mapping.keys()
        ]
        self._ranges = [
            model.actuator(act.name).ctrlrange[1] for act in self._actuators
        ]

        self._time = data.time  # To measure dt

    @property
    def actuators(self): return self._actuators

    @property
    def ranges(self): return self._ranges

    def make_state_and_weights(self, genotype: CPPNGenome, joints: JointsDict):
        state_size = 2 * self.cpgs
        _weight_matrix = np.zeros((state_size, state_size))

        cppn = CPPN3D(genotype)
        assert cppn.n_inputs() == 7  # 2*3D + length
        assert cppn.n_outputs() == 2

        joints_pts = {name: Point3D(*pos) for name, pos in joints.items()}
        output = cppn.outputs()
        for name, i in self._mapping.items():
            p = joints_pts[name]
            cppn(p, p, output)
            w = output[0]
            _weight_matrix[i][self.cpgs + i] = +w
            _weight_matrix[self.cpgs + i][i] = -w

        for lhs_name, lhs_i in self._mapping.items():
            for rhs_name, rhs_i in self._mapping.items():
                if lhs_i == rhs_i:
                    continue
                cppn(joints_pts[lhs_name], joints_pts[rhs_name], output)
                w, l = output
                if bool(l):
                    _weight_matrix[lhs_i][rhs_i] = w

        _initial_state = (
                np.hstack([np.full(self.cpgs, 1), np.full(self.cpgs, -1)])
                * 0.5 * np.sqrt(2)
        )
        _state = _initial_state.copy()

        return _initial_state, _state, _weight_matrix

    @staticmethod
    def _rk45(
        state: npt.NDArray[float], A: npt.NDArray[float], dt: float
    ) -> npt.NDArray[float]:
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
        A1: npt.NDArray[float] = np.matmul(A, state)
        A2: npt.NDArray[float] = np.matmul(A, (state + dt / 2 * A1))
        A3: npt.NDArray[float] = np.matmul(A, (state + dt / 2 * A2))
        A4: npt.NDArray[float] = np.matmul(A, (state + dt * A3))
        state = state + dt / 6 * (A1 + 2 * (A2 + A3) + A4)
        return np.clip(state, a_min=-1, a_max=1)

    def control(self, model: MjModel, data: MjData) -> None:
        dt = data.time - self._time

        self._state = self._rk45(self._state, self._weight_matrix, dt)

        # self._state = np.sin(.5 * 2 * np.pi * data.time + np.arange(2*self.cpgs))

        # Set active hinge targets to match newly calculated state.
        for i, (actuator, ctrl) in enumerate(zip(self._actuators, self._state)):
            actuator.ctrl[:] = ctrl * self._ranges[i]

        self._time = data.time

    def plot_as_network(self, path: Path, scale=1000):
        weights = self._weight_matrix

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
            style = dict(shape="circle")

            x, y, _ = self._joints_pos[name]
            x, y = x * scale, y * scale

            o_key = (round(x), round(y))
            overlap = overlaps[o_key]
            if overlap > 0:
                x, y = x + overlap * o_scale, y + overlap * o_scale
                style["color"] = "red"

            pos = f"{x},{y}"

            dot.node(name=str(index), pos=pos, **style)

            overlaps[o_key] += 1

        n = len(self._joints_pos)
        for i, j in itertools.product(range(n), repeat=2):
            w = weights[i, j]
            if w != 0:
                style = dict(
                    color="black" if w > 0 else "red",
                    penwidth=str(2 + np.log(abs(w)))
                )
                print(i, j, style, w, np.log(abs(w)))
                dot.edge(str(i), str(j), **style)

        # print(path)
        # print(dot.source)
        dot.render(outfile=path, neato_no_op=2, cleanup=True)
