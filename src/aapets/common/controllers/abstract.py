from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np

from ..mujoco.state import MjState


class Controller(ABC):
    def __init__(self, weights: Sequence[float], state: MjState):
        pass

    @property
    @abstractmethod
    def name(self): ...

    @staticmethod
    @abstractmethod
    def compute_dimensionality(joints: int): ...

    @abstractmethod
    def extract_weights(self) -> np.ndarray: ...

    @abstractmethod
    def __call__(self, state: MjState) -> None: ...

    def render_phenotype(self, path: Path, *args, **kwargs):
        pass
