from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, List

import numpy as np

from ..mujoco.state import MjState


JointsDict = dict[str, tuple[float, float, float]]


class Controller(ABC):
    def __init__(self, weights: Sequence[float], state: MjState, name: str, *args, **kwargs):
        self._joints_pos = self.joints_positions(state, name)

        self._mapping = {
            name: i for i, name in enumerate(self._joints_pos.keys())
        }

        self._actuators = [
            state.data.actuator(name) for name in self._mapping.keys()
        ]
        self._ranges = [
            state.model.actuator(act.name).ctrlrange[1] for act in self._actuators
        ]

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
            name: state.data.joint(name).xanchor for name in cls.joints(state, name_prefix)
        }

    @classmethod
    def joints(cls, state: MjState, name_prefix: str) -> List[str]:
        return [
            name for i in range(state.model.njnt)
            if len((name := state.data.joint(i).name)) > 0 and name_prefix in name
        ]

    @classmethod
    def num_joints(cls, state: MjState, name_prefix: str) -> int:
        return len(cls.joints(state, name_prefix))
