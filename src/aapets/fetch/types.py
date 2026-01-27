from enum import StrEnum, auto, IntEnum

import glfw


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

