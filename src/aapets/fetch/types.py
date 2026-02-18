from dataclasses import dataclass
from enum import StrEnum, auto, IntEnum
from pathlib import Path
from typing import Annotated

import glfw
from robot_descriptions import allegro_hand_mj_description

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

    R = glfw.KEY_R

    PAGE_DOWN = glfw.KEY_PAGE_DOWN
    LOCK = glfw.KEY_CAPS_LOCK

    CTRL = glfw.KEY_LEFT_CONTROL


class Buttons(IntEnum):
    LEFT = glfw.MOUSE_BUTTON_LEFT
    RIGHT = glfw.MOUSE_BUTTON_RIGHT


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

    debug: Annotated[bool, "Whether to enable debugging tools"] = False

    def __post_init__(self):
        self.viewer = ViewerModes.PASSIVE


class NewBodyParts(StrEnum):
    MOUTH_BODY = auto()
    MOUTH_SUCKER = auto()
    MOUTH_SENSOR = auto()

    SPIDER_EYES = auto()


class FetchTaskObjects(StrEnum):
    BALL = auto()
    HAND = "mocap_hand"


HUMAN_BODY = allegro_hand_mj_description.MJCF_PATH_RIGHT
HUMAN_HAND = "palm"
