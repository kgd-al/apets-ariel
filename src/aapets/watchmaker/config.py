from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

from mujoco import MjSpec

from aapets.common import canonical_bodies
from aapets.common.config import GenericConfig
from aapets.common.misc.config_base import ConfigBase


@dataclass
class WatchmakerConfig(GenericConfig):
    grid_size: Annotated[int, "Number of individuals per dimension"] = 3
    duration: Annotated[Optional[int], "Number of seconds per simulation"] = 5

    body: Annotated[Optional[str], "Morphology to use (or None for GUI selection"] = None

    overwrite: Annotated[bool, "Whether to clear existing data before starting"] = False
    data_folder: Annotated[Path, "Data storage for current experiment"] = \
        "tmp/watchmaker/test-run"

    cache_folder: Annotated[Path, "Persistent storage folder (for config and pre-rendered assets)"] = \
        "tmp/watchmaker/cache"

    video_size: Annotated[int, "Video size (width and height)"] = 200

    population_size: int = 0
    body_spec: MjSpec = None

    def __post_init__(self):
        self.population_size = self.grid_size**2

        if self.body is not None:
            self.body_spec = canonical_bodies.get(self.body).spec

        self.cache_folder = Path(self.cache_folder)

    def update(self):
        self.__post_init__()
