
from dataclasses import dataclass
from typing import Annotated

from ...common.config import BaseConfig


@dataclass
class TestingConfig(BaseConfig):
    base_duration: Annotated[
        float, "Maximum base duration of a trial." 
        " May be adjusted by an appropriate ratio for shorter/longer tasks"
    ] = 600
    base_length: Annotated[
        float, "Base unit length for the paths: circle's radius, shuttlerun's half length, slalom length"
    ] = 2

    movie: Annotated[bool, "Whether to generate videos of the various performances"] = True
    movie_speed: Annotated[float, "Speed factor for the recorded movie"] = 20
    movie_size: Annotated[int, "Size of the generated video"] = 480
    movie_camera: Annotated[str, "Camera used to generate videos"] = "pretty-cam"

    debug_viewer: Annotated[bool, "Whether to use a viewer for debugging purposes"] = False
    debug_draw: Annotated[bool, "Whether to draw additional debugging information"] = False
