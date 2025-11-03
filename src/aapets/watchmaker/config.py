from dataclasses import dataclass
from typing import Annotated, Optional

from aapets.common.misc.config_base import ConfigBase


@dataclass
class WatchmakerConfig(ConfigBase):
    grid_size: Annotated[int, "Number of individuals per dimension"] = 3
    duration: Annotated[Optional[int], "Number of seconds per simulation"] = 10

    seed: Annotated[Optional[int], "RNG seed (set from time if none)"] = None

    body: Annotated[Optional[str], "Morphology to use (or None for GUI selection"] = None
