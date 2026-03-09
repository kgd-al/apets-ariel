import logging
from dataclasses import dataclass
from typing import List, Literal

from mujoco import MjModel, MjData, MjSpec


@dataclass
class MjState:
    spec: MjSpec
    model: MjModel
    data: MjData

    @staticmethod
    def from_spec(spec: MjSpec) -> 'MjState':
        model = spec.compile()
        data = MjData(model)
        return MjState(spec=spec, model=model, data=data)

    @staticmethod
    def from_string(xml: str) -> 'MjState':
        return MjState.from_spec(MjSpec.from_string(xml))

    def to_string(self) -> str:
        return self.spec.to_xml()

    def get_names(self, names: str | List[str], dtype: str = "geom") -> List[str]:
        if isinstance(names, str):
            names = [names]
        objects = [
            obj.name
            for obj in self.spec.worldbody.find_all(dtype)
            if any(name in obj.name for name in names)
        ]
        return objects

    def get(self, names: str | List[str], dtype: str, struct: Literal["model", "data"]):
        if isinstance(names, str):
            names = [names]
        getter = getattr(getattr(self, struct), dtype)
        return [getter(name) for name in names]

    def get_model(self, names: List[str], dtype: str):
        return self.get(names, dtype, "model")

    def get_data(self, names: List[str], dtype: str):
        return self.get(names, dtype, "data")

    @property
    def time(self) -> float: return self.data.time

    @property
    def unpacked(self): return self, self.model, self.data
