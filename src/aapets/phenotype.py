import pprint
from typing import Any

import numpy as np
import numpy.typing as npt
from abrain import Genome as CPPNGenome, CPPN3D, Point3D
from mujoco import MjModel, MjData
from networkx import DiGraph

from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from .config import CommonConfig
from .genotype import Genotype

JointsDict = dict[str, tuple[float, float, float]]


def decode_body(genotype: Genotype.Body, config: CommonConfig) -> DiGraph:
    # System parameters
    num_modules = config.max_modules

    (type_probability_space,
     conn_probability_space,
     rotation_probability_space) = config.nde_decoder.forward(genotype.data)

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(num_modules)
    return hpd.probability_matrices_to_graph(
        type_probability_space,
        conn_probability_space,
        rotation_probability_space,
    )


def decode_brain(genotype: CPPNGenome, joints: JointsDict,
                 data: MjData, config: CommonConfig) -> Any:
    print("[kgd-debug] decode_brain")
    pprint.pprint(joints)

    return RevolveCPG(genotype, joints, data, config)


class RevolveCPG:
    """Copied from revolve but fully connected"""

    _initial_state: npt.NDArray[float]
    _weight_matrix: npt.NDArray[float]  # nxn matrix matching number of neurons
    # _output_mapping: list[tuple[int, ActiveHinge]]

    def __init__(self, genotype: CPPNGenome, joints: JointsDict,
                 data: MjData, config: CommonConfig) -> None:
        self._mapping = {
            name: i for i, name in enumerate(joints.keys())
        }
        # (
        #     cpg_network_structure,
        #     self._output_mapping,
        # ) = active_hinges_to_cpg_network_structure_neighbor(active_hinges)

        cpgs = len(joints)
        state_size = 2 * cpgs
        self._weight_matrix = np.zeros((state_size, state_size))

        cppn = CPPN3D(genotype)
        assert cppn.n_inputs() == 7  # 2*3D + length
        assert cppn.n_outputs() == 2

        joints = {name: Point3D(*pos) for name, pos in joints.items()}
        output = cppn.outputs()
        for name, i in self._mapping.items():
            p = joints[name]
            cppn(p, p, output)
            w = output[0]
            self._weight_matrix[i][cpgs + i] = w
            self._weight_matrix[cpgs + i][i] = -w

        for lhs_name, lhs_i in self._mapping.items():
            for rhs_name, rhs_i in self._mapping.items():
                if lhs_i == rhs_i:
                    continue
                cppn(joints[lhs_name], joints[rhs_name], output)
                w, l = output
                if bool(l):
                    self._weight_matrix[lhs_i][rhs_i] = w
                    self._weight_matrix[rhs_i][lhs_i] = -w

        self._initial_state = (
            np.hstack([np.full(cpgs, 1), np.full(cpgs, -1)])
            * 0.5 * np.sqrt(2)
        )
        self._state = self._initial_state.copy()

        self._actuators = [
            data.actuator(name) for name in self._mapping.keys()
        ]

        self._time = data.time

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
        if dt > 0:
            self._state = self._rk45(self._state, self._weight_matrix, dt)

            # Set active hinge targets to match newly calculated state.
            for i, (actuator, ctrl) in enumerate(zip(self._actuators, self._state)):
                actuator.ctrl[:] = ctrl #* active_hinge.range


# class RevolveCpg:
#     """
#     CPG network brain.
#
#     A state array that is integrated over time following the differential equation `X'=WX`.
#     W is a weight matrix that is multiplied by the state array.
#     The outputs of the controller are defined by the `outputs`, a list of indices for the state array.
#
#     X <-> Y
#        W
#
#     """
#
#     _initial_state: npt.NDArray[np.float_]
#     _weight_matrix: npt.NDArray[np.float_]  # nxn matrix matching number of neurons
#     _output_mapping: list[tuple[int, ActiveHinge]]
#
#     def __init__(
#         self,
#         initial_state: npt.NDArray[np.float_],
#         weight_matrix: npt.NDArray[np.float_],
#         output_mapping: list[tuple[int, ActiveHinge]],
#     ) -> None:
#         """
#         Initialize this CPG Brain Instance.
#
#         :param initial_state: The initial state of the neural network.
#         :param weight_matrix: The weight matrix used during integration.
#         :param output_mapping: Marks neurons as controller outputs and map them to the correct active hinge.
#         """
#         assert initial_state.ndim == 1
#         assert weight_matrix.ndim == 2
#         assert weight_matrix.shape[0] == weight_matrix.shape[1]
#         assert initial_state.shape[0] == weight_matrix.shape[0]
#         assert all([i >= 0 and i < len(initial_state) for i, _ in output_mapping])
#
#         self._state = initial_state
#         self._weight_matrix = weight_matrix
#         self._output_mapping = output_mapping
