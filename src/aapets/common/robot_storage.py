from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, Dict
from zipfile import ZipFile

import numpy as np
import yaml
from mujoco import MjSpec

from .config import BaseConfig
from .misc.config_base import IntrospectiveAbstractConfig
from .monitors.metrics_storage import EvaluationMetrics


@dataclass
class RerunnableRobot:
    mj_spec: MjSpec  # Environment + morphology (single robot)
    brain: Tuple[str, np.ndarray]  # Morphology is given above, just need the cpg weights

    metrics: EvaluationMetrics

    config: IntrospectiveAbstractConfig

    misc: Dict[str, Any]

    def save(self, path: Path):
        with ZipFile(path, "w") as zip_file:
            zip_file.writestr("mj_spec.xml", self.mj_spec.to_xml())
            zip_file.writestr("brain.yaml",
                              yaml.safe_dump({self.brain[0]: self.brain[1].tolist()})),
            zip_file.writestr("monitors.yaml", self.metrics.to_yaml())
            zip_file.writestr("misc.yaml", yaml.dump(self.misc))
            with zip_file.open("config.yaml", "w") as config_file:
                self.config.write_yaml(config_file)

    @classmethod
    def load(cls, path: Path):
        with ZipFile(path, "r") as zip_file:
            with zip_file.open("config.yaml", "r") as config_file:
                config = IntrospectiveAbstractConfig.read_yaml(config_file)
            brain = next(iter(yaml.safe_load(zip_file.read("brain.yaml")).items()))
            return cls(
                mj_spec=MjSpec.from_string(zip_file.read("mj_spec.xml").decode("utf-8")),
                brain=(brain[0], np.array(brain[1])),
                metrics=EvaluationMetrics.from_yaml(zip_file.read("monitors.yaml").decode("utf-8")),
                misc=yaml.unsafe_load(zip_file.read("misc.yaml")),
                config=config
            )
