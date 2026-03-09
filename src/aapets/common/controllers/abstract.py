from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, List, Literal

import numpy as np
from mujoco import MjModel, MjData

from ..mujoco.state import MjState


JointsDict = dict[str, tuple[float, float, float]]


class Controller(ABC):
    def __init__(self, weights: Sequence[float], state: MjState, name: str, *args, **kwargs):
        self._joints_pos, self._mapping, self._actuators, self._joints, self._ranges = (
            self.control_data(state, name))

    @classmethod
    def control_data(cls, state: MjState, robot_name: str):
        joints_pos = cls.joints_positions(state, robot_name)
        mapping = {name: i for i, name in enumerate(joints_pos.keys())}
        actuators = [state.data.actuator(name) for name in mapping.keys()]
        joints = [state.data.joint(a.name) for a in actuators]
        ranges = [state.model.actuator(act.name).ctrlrange[1] for act in actuators]

        # print("[kgd-debug|Controller:control_data] Generated control data for", robot_name)
        # print("[kgd-debug|Controller:control_data]", f"{joints_pos=}")
        # print("[kgd-debug|Controller:control_data]", f"{mapping=}")
        # print("[kgd-debug|Controller:control_data]", f"{actuators=}")
        # print("[kgd-debug|Controller:control_data]", f"{joints=}")
        # with np.printoptions(precision=50):
        #     print("[kgd-debug|Controller:control_data]", f"ranges={np.array(ranges)}")

        return joints_pos, mapping, actuators, joints, ranges

    @abstractmethod
    def name(self): ...

    @classmethod
    @abstractmethod
    def num_parameters(cls, state: MjState, name: str, *args, **kwargs) -> int: ...

    @abstractmethod
    def extract_weights(self) -> np.ndarray: ...

    @abstractmethod
    def __call__(self, state: MjState) -> None: ...

    def render_phenotype(self, path: Path, *args, **kwargs):
        pass

    @classmethod
    def joints_positions(cls, state: MjState, name_prefix: str) -> JointsDict:
        return {
            name: state.data.joint(name).xanchor for name in cls.named_joints(state, name_prefix)
        }

    @classmethod
    def named_joints(cls, state: MjState, name_prefix: str) -> List[str]:
        return state.get_names(name_prefix, "joint")

    @classmethod
    def joints(cls, state: MjState, name_prefix: str, struct: Literal["model", "data"]):
        return state.get(cls.named_joints(state, name_prefix), "joint", struct)

    @classmethod
    def actuators(cls, state: MjState, name_prefix: str, struct: Literal["model", "data"]):
        return state.get(cls.named_joints(state, name_prefix), "actuator", struct)

    @classmethod
    def num_joints(cls, state: MjState, name_prefix: str) -> int:
        return len(cls.named_joints(state, name_prefix))
