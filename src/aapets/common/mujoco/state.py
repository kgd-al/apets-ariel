import logging
from dataclasses import dataclass
from typing import List

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

    def get(self, names: List[str], dtype: str = "geom"):
        objects = [
            obj.name
            for obj in self.spec.worldbody.find_all(dtype)
            if any(name in obj.name for name in names)
        ]
        if len(objects) != len(names):
            logging.warning(f"Requested {len(names)} but only found {len(objects)}")

        if len(objects) == 0:
            return objects[0]
        else:
            return objects

    @property
    def time(self) -> float: return self.data.time
