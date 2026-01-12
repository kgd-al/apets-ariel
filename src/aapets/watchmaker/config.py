from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Optional

from mujoco import MjSpec

from ..common import canonical_bodies
from ..common.config import EvoConfig, BaseConfig


class RunTypes(StrEnum):
    HUMAN = "human"
    HILL = "hill-climber"
    RANDOM = "random"


@dataclass
class WatchmakerConfig(BaseConfig, EvoConfig):
    speed_up: Annotated[Optional[float], "Speed-up ratio for the videos"] = 6
    duration: Annotated[Optional[int], "Number of seconds per individual"] = 30

    data_folder: Annotated[Path, "Data storage for current experiment"] = \
        Path("tmp/watchmaker")

    population_size: Annotated[Optional[int], "Number of concurrent individuals (parent+offsprings)"] = 9
    max_evaluations: Annotated[Optional[int], "Maximum number of evaluations"] = None

    mutation_scale: Annotated[float, "Standard deviation of the normal law applied to genomes when mutating"] = .5
    mutation_range: Annotated[float, "Maximum value (and opposite of minimum) possible for the genome's fields"] = 2

    body: Annotated[Optional[str], "Morphology to use (or None for GUI selection)"] = None

    video_size: Annotated[int, "Video size (width and height)"] = 200

    camera_angle: Annotated[int, "Camera angle from side (0) to top (90)"] = 90

    layout: Annotated[int, "Binary mask enabling/disabling specific population slots",
                      dict(type=lambda v: int(v, 16))] = 0xFF

    show_trajectory: Annotated[bool, "Whether to show the trajectory as white line"] = False
    show_start: Annotated[bool, "Whether to show the starting point as a 'painted' circle"] = False
    show_xspeed: Annotated[bool, "Whether to show the x velocity"] = False

    run_type: Annotated[RunTypes, "What selection mechanism is used",
                        dict(choices=([t.value for t in RunTypes]),
                             type=lambda s: RunTypes[s.upper()])] = RunTypes.HUMAN

    plot_from: Annotated[Path, "If set to a path, existing data is used for plotting and nothing runs"] = None
    plot_extension: Annotated[str, "Extension for the plots", dict(choices=("png", "pdf"))] = "pdf"

    plots: Annotated[bool, "Graph generation (e.g. where plot libraries are not installed)"] = True
    symlink: Annotated[bool, "Generate a symlink to the last run"] = True

    parallelism: Annotated[bool, "Whether to distribute evaluation on multiple processes/cores"] = True

    camera = "apet1_tracking-cam"

    body_spec: MjSpec = None
    run_id: str = None

    debug_fast: Annotated[bool, "Replaces GIFs with images for faster evaluations"] = False
    debug_show_id: Annotated[bool, "Displays genetic ID for easier debugging"] = False
    timing: Annotated[bool, "Whether to log timing information (for debugging purposes)"] = False
    skip_consent: Annotated[bool, "Go straight to evolution with asking for consent/username/..."] = False

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["body_spec"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__post_init__()

    def __post_init__(self):
        if self.body is not None:
            self.body_spec = canonical_bodies.get(self.body).spec

        self.cache_folder = Path(self.cache_folder)

    def update(self):
        self.__post_init__()
