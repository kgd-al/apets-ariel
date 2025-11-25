import itertools
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Optional, Iterator, Tuple, List

from mujoco import MjSpec

from aapets.common import canonical_bodies
from aapets.common import GenericConfig, EvoConfig, BaseConfig


class PopulationGrid:
    def __init__(self, layout: int):
        self.grid_size = 3
        self.layout = layout & 255
        self.str_layout = f"{layout:08b}"
        self.str_layout = self.str_layout[:4] + "1" + self.str_layout[4:]
        self.population_size = self.str_layout.count("1")

    def __repr__(self):
        return "\n".join([
            f"Layout: {self.layout}"
        ] + [
            " " + self.str_layout[i * self.grid_size:(i + 1) * self.grid_size]
            for i in range(self.grid_size)
        ])

    def ix(self, i: int, j: int) -> int:
        assert 0 <= i < self.grid_size
        assert 0 <= j < self.grid_size
        return j * self.grid_size + i

    def empty_cell(self, i: int, j: int) -> bool:
        return self.str_layout[self.ix(i, j)] != "1"

    def all_cells(self) -> List[Tuple[int, int]]:
        return [
            (i, j) for j, i in
            itertools.product(range(self.grid_size), range(self.grid_size))
        ]

    def valid_cells(self) -> Iterator[Tuple[int, int]]:
        for i, j in self.all_cells():
            if not self.empty_cell(i, j):
                yield i, j

    @property
    @lru_cache(maxsize=1)
    def parent_ix(self): return self.ix(1, 1)


@dataclass
class WatchmakerConfig(BaseConfig, EvoConfig):
    speed_up: Annotated[Optional[float], "Speed-up ratio for the videos"] = 4

    max_evaluations: Annotated[Optional[int], "Maximum number of evaluations"] = None

    body: Annotated[Optional[str], "Morphology to use (or None for GUI selection)"] = None

    video_size: Annotated[int, "Video size (width and height)"] = 200

    camera_angle: Annotated[int, "Camera angle from side (0) to top (90)"] = 90

    layout: Annotated[int, "Binary mask enabling/disabling specific population slots",
                      dict(type=lambda v: int(v, 16))] = 0xFF

    show_trajectory: Annotated[bool, "Whether to show the trajectory as white line"] = False
    show_start: Annotated[bool, "Whether to show the starting point as a 'painted' circle"] = False
    show_xspeed: Annotated[bool, "Whether to show the x velocity"] = False

    grid_spec: PopulationGrid = None
    body_spec: MjSpec = None

    debug_fast: Annotated[bool, "Replaces GIFs with images for faster evaluations"] = False
    debug_show_id: Annotated[bool, "Displays genetic ID for easier debugging"] = False
    timing: Annotated[bool, "Whether to log timing information (for debugging purposes)"] = False

    def __post_init__(self):
        self.grid_spec = PopulationGrid(self.layout)

        if self.body is not None:
            self.body_spec = canonical_bodies.get(self.body).spec

        self.cache_folder = Path(self.cache_folder)

    def update(self):
        self.__post_init__()
