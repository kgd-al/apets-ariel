from dataclasses import dataclass
from enum import StrEnum, auto, IntEnum
from pathlib import Path
from typing import Annotated

import glfw

from ..common.config import BaseConfig, ViewerConfig, ViewerModes


class InteractionMode(StrEnum):
    DEMO = auto()
    BALL = auto()
    ROBOT = auto()
    HUMAN = auto()


class Keys(IntEnum):
    RIGHT = glfw.KEY_RIGHT
    UP = glfw.KEY_UP
    LEFT = glfw.KEY_LEFT
    DOWN = glfw.KEY_DOWN

    PAGE_DOWN = glfw.KEY_PAGE_DOWN
    LOCK = glfw.KEY_CAPS_LOCK

    CTRL = glfw.KEY_LEFT_CONTROL


class Buttons(IntEnum):
    LEFT = glfw.MOUSE_BUTTON_LEFT
    RIGHT = glfw.MOUSE_BUTTON_RIGHT


class Constraints(StrEnum):
    HAND_BALL = "constraint-hand-ball"
    # ROBOT_BALL = "constraint-robot-ball"


@dataclass
class Config(BaseConfig, ViewerConfig):
    robot_archive: Annotated[
        Path, "Where to look for a pre-trained robot",
        dict(required=True)
    ] = None

    test_folder: Annotated[Path, "Where to store the results"] = Path("tmp/fetch")

    mode: Annotated[
        InteractionMode, "What type of interactions to do",
        dict(choices=list(InteractionMode))
    ] = InteractionMode.BALL

    arena_extent: Annotated[float, "Arena size in both directions"] = 5
    human_height: Annotated[float, "Human agent height"] = 1.11

    debug: Annotated[bool, "Whether to enable debugging tools"] = True

    def __post_init__(self):
        self.viewer = ViewerModes.PASSIVE
