import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import numpy as np
import yaml
from mujoco import MjSpec

from aapets.common.misc.config_base import ConfigBase


@dataclass
class RerunnableRobot:
    mj_spec: MjSpec  # Environment + morphology (single robot)
    brain: np.ndarray  # Morphology is given above, just need the cpg weights

    config: ConfigBase

    fitness: Optional[dict[str, float]]
    infos: dict

    def save(self, path: Path):
        with ZipFile(path, "w") as zip_file:
            zip_file.writestr("mj_spec", self.mj_spec.to_xml())
            zip_file.writestr("brain", self.brain.dumps())
            zip_file.writestr("fitness", yaml.safe_dump(self.fitness))
            zip_file.writestr("info.yaml", yaml.safe_dump(self.infos))
            with zip_file.open("config.yaml", "w") as config_file:
                self.config.write_yaml(config_file)

    @classmethod
    def load(cls, path: Path):
        with ZipFile(path, "r") as zip_file:
            with zip_file.open("config.yaml", "r") as config_file:
                config = ConfigBase.read_yaml(config_file)
            return cls(
                mj_spec=MjSpec.from_string(zip_file.read("mj_spec").decode("utf-8")),
                brain=pickle.loads(zip_file.read("brain")),
                fitness=yaml.safe_load(zip_file.read("fitness").decode("utf-8")),
                infos=yaml.safe_load(zip_file.read("infos")),
                config=config
            )
