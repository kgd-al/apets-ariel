import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple, Dict
from zipfile import ZipFile

import numpy as np
import yaml
from mujoco import MjSpec

from .misc.config_base import IntrospectiveAbstractConfig
from .monitors.metrics_storage import EvaluationMetrics

CURRENT_FORMAT = 0


@dataclass
class RerunnableRobot:
    mj_spec: MjSpec  # Environment + morphology (single robot)

    # First is the name of the controller (as given by Controller.name)
    # Last are the weights
    # Dictionary is kwargs arguments (if needed)
    brain: Tuple[str, dict[str, Any], np.ndarray]

    metrics: EvaluationMetrics

    config: IntrospectiveAbstractConfig

    misc: Dict[str, Any]

    def save(self, path: Path):
        with ZipFile(path, "w") as zip_file:
            zip_file.writestr(".format", str(CURRENT_FORMAT))
            zip_file.writestr("mj_spec.xml", self.mj_spec.to_xml())

            brain = self.brain
            zip_file.writestr("brain.yaml",
                              yaml.safe_dump({
                                  "robot": (brain[0], brain[1], brain[2].tolist())
                              })),

            zip_file.writestr("monitors.yaml", self.metrics.to_yaml())
            zip_file.writestr("misc.yaml", yaml.dump(self.misc))
            with zip_file.open("config.yaml", "w") as config_file:
                self.config.write_yaml(config_file)

    @classmethod
    def load(cls, path: Path):
        with ZipFile(path, "r") as zip_file:
            try:
                obsolete = (int(zip_file.read(".format")) != CURRENT_FORMAT)
            except KeyError:
                obsolete = True

            if obsolete:
                logging.warning("Provided archive is in an old format. Applying patches and crossing fingers")

            with zip_file.open("config.yaml", "r") as config_file:
                config = IntrospectiveAbstractConfig.read_yaml(config_file)

            brain = yaml.safe_load(zip_file.read("brain.yaml"))
            if len(brain) > 1:
                logging.warning("> More than one brain in the archive. Ignoring?")
            brain_contents = next(iter(brain.values()))
            if isinstance(brain_contents, list) and len(brain_contents) != 3:
                logging.warning("> Wrong number of brain arguments. Trying to deal with it")
                brain = next(iter(brain.items()))
                brain = (brain[0], {}, brain[-1])
            else:
                brain = brain_contents

            print(f"Restoring brain from:\n{brain}")

            return cls(
                mj_spec=MjSpec.from_string(zip_file.read("mj_spec.xml").decode("utf-8")),
                brain=(brain[0], brain[1], np.array(brain[-1])),
                metrics=EvaluationMetrics.from_yaml(zip_file.read("monitors.yaml").decode("utf-8")),
                misc=yaml.unsafe_load(zip_file.read("misc.yaml")),
                config=config
            )
