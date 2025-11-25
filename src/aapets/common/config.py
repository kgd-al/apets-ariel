from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated, Optional

from .misc.config_base import IntrospectiveAbstractConfig


@dataclass
class BaseConfig(IntrospectiveAbstractConfig):
    seed: Annotated[Optional[int], "RNG seed (set from time if none)"] = None
    verbosity: Annotated[int, "How talkative should I be?"] = 1

    duration: Annotated[Optional[int], "Number of seconds per simulation"] = 10

    control_frequency: Annotated[int, "How often to query the control for new outputs (Hz)"] = 20
    sample_frequency: Annotated[int, "How often to compute intermediates for step-wise functions"] = 20

    robot_name_prefix: Annotated[str, "Name prefix for the robots"] = "apet"


@dataclass
class EvoConfig(IntrospectiveAbstractConfig):
    overwrite: Annotated[bool, "Whether to clear existing data before starting"] = False
    data_folder: Annotated[Path, "Data storage for current experiment"] = \
        "tmp/watchmaker/test-run"

    cache_folder: Annotated[Path, "Persistent storage folder (for config and pre-rendered assets)"] = \
        "tmp/watchmaker/cache"


class ViewerModes(StrEnum):
    NONE = auto()
    INTERACTIVE = auto()
    PASSIVE = auto()


@dataclass
class ViewerConfig(IntrospectiveAbstractConfig):
    viewer: Annotated[ViewerModes, "Whether to render in an interactive viewer"] = ViewerModes.NONE

    speed: Annotated[float, "Rendered simulation speed factor"] = 1.0

    auto_start: Annotated[bool, "Whether to start immediately or wait for an input"] = True
    auto_quit: Annotated[bool, "Whether to wait for confirmation at the end of the simulation"] = True
    camera: Annotated[Optional[str], "Specifies the camera named/id to use"] = None

    settings_save: Annotated[bool, "Whether to save viewer-specific settings before quitting"] = True
    settings_restore: Annotated[bool, "Whether to restore viewer-specific settings"] = True

    movie: Annotated[bool, "Whether to generate a movie"] = False
    movie_framerate: Annotated[int, "How any images per second"] = 25
    movie_width: Annotated[int, "Width of the generated movie"] = 500
    movie_height: Annotated[int, "Height of the generated movie"] = 500


@dataclass
class AnalysisConfig(IntrospectiveAbstractConfig):
    render_brain_genotype: Annotated[
        bool, "Plot brain genotype to png (if possible)"] = False

    render_brain_phenotype: Annotated[
        bool, "Plot brain phenotype (controller) to png (if possible)"] = False

    render_brain_activity: Annotated[
        bool, "Plot controller input/output activity"] = False

    render_trajectory: Annotated[
        bool, "Plot robot trajectory in 2D"] = False
