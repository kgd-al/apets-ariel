"""
Implements the canonical bodies from revolve2

Warning: while the joints have been respected, their orientation is not guaranteed. These robots
should have the same range of movements as their original counterparts but controllers are **not**
expected to transfer (which should be obvious).
"""

import math
import sys
from contextlib import contextmanager
from enum import Enum
from typing import Callable, Type, Tuple, Generator

import numpy as np

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule

current_module = sys.modules[__name__]

Hinge, Brick = HingeModule, BrickModule

SUBMODULES = Hinge | Brick
MODULES = CoreModule | SUBMODULES

faces = [
    ModuleFaces.FRONT, ModuleFaces.LEFT, ModuleFaces.BACK, ModuleFaces.RIGHT,
]
F, L, B, R = faces


def get(name: str):
    body_fn = getattr(current_module, f"body_{name}", None)
    if body_fn is None:
        raise RuntimeError(f"'{name}' is not a valid canonical body name")

    return body_fn()


def get_all() -> dict[str, Callable]:
    return {
        name[5:]: getattr(current_module, name)
        for name in current_module.__dir__()
        if name.startswith("body_")
    }


_DEFAULT_COLORS = {
    CoreModule: (["core"], (.1, .1, .9)),
    Hinge: (["stator", "rotor"], (.9, .1, .1)),
    Brick: (["brick"], (.1, .7, .1)),
}


def apply_color(colors, module):
    geom_names, geom_color = colors[module.__class__]
    for name in geom_names:
        module.spec.geom(name).rgba = (*geom_color, 1)


def make_core():
    core = CoreModule(index=0)
    core.name = "C"
    apply_color(_DEFAULT_COLORS, core)
    return core


class Attacher:
    def __init__(
            self,
            start_index=1,
            colors=None):
        self.colors = colors or _DEFAULT_COLORS
        self.index = start_index

    def __call__(self,
                 parent: MODULES,
                 face: ModuleFaces,
                 module_t: Type[SUBMODULES],
                 name: str,
                 rotation: float = 0):

        module = module_t(self.index)
        apply_color(self.colors, module)

        self.index += 1

        name = f"{parent.name}-{name}"
        parent.sites[face].attach_body(body=module.body, prefix=name + "-")

        module.name = name

        if rotation != 0:
            module.rotate(rotation)

        return module

    @contextmanager
    def branch(self, *args, **kwargs):
        yield self.__call__(*args, **kwargs)


def body_spider() -> CoreModule:
    core, attacher = make_core(), Attacher()

    for f in faces:
        h0 = attacher(core, f, Hinge, f"{f.name[0]}H")
        b0 = attacher(h0, F, Brick, "B")
        h1 = attacher(b0, F, Hinge, "H", rotation=90)
        b1 = attacher(h1, F, Brick, "B")

    return core


def body_spider45() -> CoreModule:
    core = body_spider()
    core.spec.body("core").quat = (math.cos(math.pi / 8), 0, 0, math.sin(math.pi / 8))
    return core


def body_gecko() -> CoreModule:
    core, attacher = make_core(), Attacher()

    al = attacher(core, L, Hinge, "LH", rotation=90)
    attacher(al, F, Brick, "LB")

    ar = attacher(core, R, Hinge, "RH", rotation=90)
    attacher(ar, F, Brick, "RB")

    sh0 = attacher(core, B, Hinge, "S")
    sb0 = attacher(sh0, F, Brick, "S")
    sh1 = attacher(sb0, F, Hinge, "S")
    sb1 = attacher(sh1, F, Brick, "S")

    ll = attacher(sb1, L, Hinge, "LH", rotation=90)
    attacher(ll, F, Brick, "LB")

    lr = attacher(sb1, R, Hinge, "RH", rotation=90)
    attacher(lr, F, Brick, "RB")

    return core


def body_babya() -> CoreModule:
    core, attacher = make_core(), Attacher()

    al = attacher(core, L, Hinge, "LH", rotation=90)
    attacher(al, F, Brick, "LB")

    h0 = attacher(core, R, Hinge, f"RH", rotation=90)
    h1 = attacher(h0, F, Hinge, "H", rotation=-90)
    b0 = attacher(h1, F, Brick, "B")
    h2 = attacher(b0, F, Hinge, "H", rotation=90)
    b1 = attacher(h2, F, Brick, "B")

    sh0 = attacher(core, B, Hinge, "S")
    sb0 = attacher(sh0, F, Brick, "S")
    sh1 = attacher(sb0, F, Hinge, "S")
    sb1 = attacher(sh1, F, Brick, "S")

    ll = attacher(sb1, L, Hinge, "LH", rotation=90)
    attacher(ll, F, Brick, "LB")

    lr = attacher(sb1, R, Hinge, "RH", rotation=90)
    attacher(lr, F, Brick, "RB")

    return core


def body_ant() -> CoreModule:
    """
    Get the ant modular robot.

    :returns: the robot.
    """
    core, attacher = make_core(), Attacher()

    def limb(src, face):
        _h = attacher(src, face, Hinge, f"{face.name[0]}H", rotation=90)
        attacher(_h, F, Brick, "B")

    def limbs(src):
        limb(src, L)
        limb(src, R)

    sh0 = attacher(core, B, Hinge, "S")
    sb0 = attacher(sh0, F, Brick, "S")
    sh1 = attacher(sb0, F, Hinge, "S")
    sb1 = attacher(sh1, F, Brick, "S")

    limbs(core)
    limbs(sb0)
    limbs(sb1)

    return core


def body_salamander() -> CoreModule:
    """
    Get the salamander modular robot.

    :returns: the robot.
    """
    core, attacher = make_core(), Attacher()

    # left arm
    l_h = attacher(core, L, Hinge, "lH")
    attacher(l_h, F, Hinge, "lHH", rotation=90)

    # right arm
    attacher(core, R, Hinge, "rH", rotation=90)

    # first block (two bricks, two hinges)
    b_h = attacher(core, B, Hinge, "bH")
    with attacher.branch(b_h, F, Brick, "bHB") as b_hb:
        attacher(b_hb, R, Hinge, "bHBrH", rotation=90)
    with attacher.branch(b_hb, F, Brick, "bHBB") as b_hbb:
        attacher(b_hbb, R, Hinge, "LH", rotation=90)
    b_hbbh = attacher(b_hbb, F, Hinge, "bHBBH")

    # second block
    b_hbbh_b = attacher(b_hbbh, F, Brick, "bHBBBH_B")
    b_hbbh_bb = attacher(b_hbbh_b, F, Brick, "bHBBBH_BB")
    b_hbbh_bbb = attacher(b_hbbh_bb, F, Brick, "bHBBBH_BBB")
    b_hbbh_bbbb = attacher(b_hbbh_bbb, F, Brick, "bHBBBH_BBBB")

    # body.core_v1.left = Hinge(np.pi / 2.0)
    # body.core_v1.left.attachment = Hinge(-np.pi / 2.0)

    # body.core_v1.right = Hinge(0.0)

    body.core_v1.back = Hinge(np.pi / 2.0)
    body.core_v1.back.attachment = Brick(-np.pi / 2.0)
    body.core_v1.back.attachment.left = Hinge(0.0)
    body.core_v1.back.attachment.front = Brick(0.0)
    body.core_v1.back.attachment.front.left = Hinge(0.0)
    body.core_v1.back.attachment.front.front = Hinge(np.pi / 2.0)
    body.core_v1.back.attachment.front.front.attachment = Brick(-np.pi / 2.0)

    body.core_v1.back.attachment.front.front.attachment.left = Hinge(0.0)
    body.core_v1.back.attachment.front.front.attachment.left.attachment = Brick(0.0)
    body.core_v1.back.attachment.front.front.attachment.left.attachment.left = Brick(
        0.0
    )
    body.core_v1.back.attachment.front.front.attachment.left.attachment.front = (
        Hinge(np.pi / 2.0)
    )
    body.core_v1.back.attachment.front.front.attachment.left.attachment.front.attachment = Hinge(
        -np.pi / 2.0
    )

    body.core_v1.back.attachment.front.front.attachment.front = Brick(0.0)
    body.core_v1.back.attachment.front.front.attachment.front.left = Hinge(0.0)
    body.core_v1.back.attachment.front.front.attachment.front.front = Brick(0.0)
    body.core_v1.back.attachment.front.front.attachment.front.front.left = (
        Hinge(0.0)
    )
    body.core_v1.back.attachment.front.front.attachment.front.front.front = Brick(0.0)
    body.core_v1.back.attachment.front.front.attachment.front.front.front.front = (
        Hinge(np.pi / 2.0)
    )
    body.core_v1.back.attachment.front.front.attachment.front.front.front.front.attachment = Brick(
        -np.pi / 2.0
    )
    body.core_v1.back.attachment.front.front.attachment.front.front.front.front.attachment.left = Brick(
        0.0
    )
    body.core_v1.back.attachment.front.front.attachment.front.front.front.front.attachment.front = Hinge(
        np.pi / 2.0
    )

    return body


def blokky_v1() -> CoreModule:
    """
    Get the blokky modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.left = Hinge(np.pi / 2.0)
    body.core_v1.back = Brick(0.0)
    body.core_v1.back.right = Hinge(np.pi / 2.0)
    body.core_v1.back.front = Hinge(np.pi / 2.0)
    body.core_v1.back.front.attachment = Hinge(-np.pi / 2.0)
    body.core_v1.back.front.attachment.attachment = Brick(0.0)
    body.core_v1.back.front.attachment.attachment.front = Brick(0.0)
    body.core_v1.back.front.attachment.attachment.front.right = Brick(0.0)
    body.core_v1.back.front.attachment.attachment.front.right.left = Brick(0.0)
    body.core_v1.back.front.attachment.attachment.front.right.front = Brick(0.0)
    body.core_v1.back.front.attachment.attachment.right = Brick(0.0)
    body.core_v1.back.front.attachment.attachment.right.front = Brick(0.0)
    body.core_v1.back.front.attachment.attachment.right.front.right = Brick(0.0)
    body.core_v1.back.front.attachment.attachment.right.front.front = Hinge(0.0)

    return body


def park_v1() -> CoreModule:
    """
    Get the park modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.back = Hinge(np.pi / 2.0)
    body.core_v1.back.attachment = Hinge(-np.pi / 2.0)
    body.core_v1.back.attachment.attachment = Brick(0.0)
    body.core_v1.back.attachment.attachment.right = Brick(0.0)
    body.core_v1.back.attachment.attachment.left = Hinge(0.0)
    body.core_v1.back.attachment.attachment.front = Brick(0.0)
    body.core_v1.back.attachment.attachment.front.right = Hinge(-np.pi / 2.0)
    body.core_v1.back.attachment.attachment.front.front = Hinge(-np.pi / 2.0)
    body.core_v1.back.attachment.attachment.front.left = Hinge(0.0)
    body.core_v1.back.attachment.attachment.front.left.attachment = Brick(0.0)
    body.core_v1.back.attachment.attachment.front.left.attachment.right = Hinge(
        -np.pi / 2.0
    )
    body.core_v1.back.attachment.attachment.front.left.attachment.left = Brick(0.0)
    body.core_v1.back.attachment.attachment.front.left.attachment.front = Hinge(
        0.0
    )
    body.core_v1.back.attachment.attachment.front.left.attachment.front.attachment = (
        Brick(0.0)
    )

    return body


def babyb_v1() -> CoreModule:
    """
    Get the babyb modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.left = Hinge(np.pi / 2.0)
    body.core_v1.left.attachment = Brick(-np.pi / 2.0)
    body.core_v1.left.attachment.front = Hinge(0.0)
    body.core_v1.left.attachment.front.attachment = Brick(0.0)
    body.core_v1.left.attachment.front.attachment.front = Hinge(np.pi / 2.0)
    body.core_v1.left.attachment.front.attachment.front.attachment = Brick(0.0)

    body.core_v1.right = Hinge(np.pi / 2.0)
    body.core_v1.right.attachment = Brick(-np.pi / 2.0)
    body.core_v1.right.attachment.front = Hinge(0.0)
    body.core_v1.right.attachment.front.attachment = Brick(0.0)
    body.core_v1.right.attachment.front.attachment.front = Hinge(np.pi / 2.0)
    body.core_v1.right.attachment.front.attachment.front.attachment = Brick(0.0)

    body.core_v1.front = Hinge(np.pi / 2.0)
    body.core_v1.front.attachment = Brick(-np.pi / 2.0)
    body.core_v1.front.attachment.front = Hinge(0.0)
    body.core_v1.front.attachment.front.attachment = Brick(0.0)
    body.core_v1.front.attachment.front.attachment.front = Hinge(np.pi / 2.0)
    body.core_v1.front.attachment.front.attachment.front.attachment = Brick(0.0)

    body.core_v1.back = Hinge(np.pi / 2.0)
    body.core_v1.back.attachment = Brick(-np.pi / 2.0)

    return body


def garrix_v1() -> CoreModule:
    """
    Get the garrix modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.front = Hinge(np.pi / 2.0)

    body.core_v1.left = Hinge(np.pi / 2.0)
    body.core_v1.left.attachment = Hinge(0.0)
    body.core_v1.left.attachment.attachment = Hinge(-np.pi / 2.0)
    body.core_v1.left.attachment.attachment.attachment = Brick(0.0)
    body.core_v1.left.attachment.attachment.attachment.front = Brick(0.0)
    body.core_v1.left.attachment.attachment.attachment.left = Hinge(0.0)

    part2 = Brick(0.0)
    part2.right = Hinge(np.pi / 2.0)
    part2.front = Hinge(np.pi / 2.0)
    part2.left = Hinge(0.0)
    part2.left.attachment = Hinge(np.pi / 2.0)
    part2.left.attachment.attachment = Hinge(-np.pi / 2.0)
    part2.left.attachment.attachment.attachment = Brick(0.0)

    body.core_v1.left.attachment.attachment.attachment.left.attachment = part2

    return body


def insect_v1() -> CoreModule:
    """
    Get the insect modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.right = Hinge(np.pi / 2.0)
    body.core_v1.right.attachment = Hinge(-np.pi / 2.0)
    body.core_v1.right.attachment.attachment = Brick(0.0)
    body.core_v1.right.attachment.attachment.right = Hinge(0.0)
    body.core_v1.right.attachment.attachment.front = Hinge(np.pi / 2.0)
    body.core_v1.right.attachment.attachment.left = Hinge(np.pi / 2.0)
    body.core_v1.right.attachment.attachment.left.attachment = Brick(-np.pi / 2.0)
    body.core_v1.right.attachment.attachment.left.attachment.front = Hinge(
        np.pi / 2.0
    )
    body.core_v1.right.attachment.attachment.left.attachment.right = Hinge(0.0)
    body.core_v1.right.attachment.attachment.left.attachment.right.attachment = (
        Hinge(0.0)
    )
    body.core_v1.right.attachment.attachment.left.attachment.right.attachment.attachment = Hinge(
        np.pi / 2.0
    )

    return body


def linkin_v1() -> CoreModule:
    """
    Get the linkin modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.back = Hinge(0.0)

    body.core_v1.right = Hinge(np.pi / 2.0)
    body.core_v1.right.attachment = Hinge(0.0)
    body.core_v1.right.attachment.attachment = Hinge(0.0)
    body.core_v1.right.attachment.attachment.attachment = Hinge(-np.pi / 2.0)
    body.core_v1.right.attachment.attachment.attachment.attachment = Brick(0.0)

    part2 = body.core_v1.right.attachment.attachment.attachment.attachment
    part2.front = Brick(0.0)

    part2.left = Hinge(0.0)
    part2.left.attachment = Hinge(0.0)

    part2.right = Hinge(np.pi / 2.0)
    part2.right.attachment = Hinge(-np.pi / 2.0)
    part2.right.attachment.attachment = Hinge(0.0)
    part2.right.attachment.attachment.attachment = Hinge(np.pi / 2.0)
    part2.right.attachment.attachment.attachment.attachment = Hinge(0.0)

    return body


def longleg_v1() -> CoreModule:
    """
    Get the longleg modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.left = Hinge(np.pi / 2.0)
    body.core_v1.left.attachment = Hinge(0.0)
    body.core_v1.left.attachment.attachment = Hinge(0.0)
    body.core_v1.left.attachment.attachment.attachment = Hinge(-np.pi / 2.0)
    body.core_v1.left.attachment.attachment.attachment.attachment = Hinge(0.0)
    body.core_v1.left.attachment.attachment.attachment.attachment.attachment = Brick(
        0.0
    )

    part2 = body.core_v1.left.attachment.attachment.attachment.attachment.attachment
    part2.right = Hinge(0.0)
    part2.front = Hinge(0.0)
    part2.left = Hinge(np.pi / 2.0)
    part2.left.attachment = Hinge(-np.pi / 2.0)
    part2.left.attachment.attachment = Brick(0.0)
    part2.left.attachment.attachment.right = Hinge(np.pi / 2.0)
    part2.left.attachment.attachment.left = Hinge(np.pi / 2.0)
    part2.left.attachment.attachment.left.attachment = Hinge(0.0)

    return body


def penguin_v1() -> CoreModule:
    """
    Get the penguin modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.right = Brick(0.0)
    body.core_v1.right.left = Hinge(np.pi / 2.0)
    body.core_v1.right.left.attachment = Hinge(-np.pi / 2.0)
    body.core_v1.right.left.attachment.attachment = Brick(0.0)
    body.core_v1.right.left.attachment.attachment.right = Hinge(0.0)
    body.core_v1.right.left.attachment.attachment.left = Hinge(np.pi / 2.0)
    body.core_v1.right.left.attachment.attachment.left.attachment = Hinge(
        -np.pi / 2.0
    )
    body.core_v1.right.left.attachment.attachment.left.attachment.attachment = (
        Hinge(np.pi / 2.0)
    )
    body.core_v1.right.left.attachment.attachment.left.attachment.attachment.attachment = Brick(
        -np.pi / 2.0
    )

    part2 = (
        body.core_v1.right.left.attachment.attachment.left.attachment.attachment.attachment
    )

    part2.front = Hinge(np.pi / 2.0)
    part2.front.attachment = Brick(-np.pi / 2.0)

    part2.right = Hinge(0.0)
    part2.right.attachment = Hinge(0.0)
    part2.right.attachment.attachment = Hinge(np.pi / 2.0)
    part2.right.attachment.attachment.attachment = Brick(-np.pi / 2.0)

    part2.right.attachment.attachment.attachment.left = Hinge(np.pi / 2.0)

    part2.right.attachment.attachment.attachment.right = Brick(0.0)
    part2.right.attachment.attachment.attachment.right.front = Hinge(
        np.pi / 2.0
    )

    return body


def pentapod_v1() -> CoreModule:
    """
    Get the pentapod modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.right = Hinge(np.pi / 2.0)
    body.core_v1.right.attachment = Hinge(0.0)
    body.core_v1.right.attachment.attachment = Hinge(0.0)
    body.core_v1.right.attachment.attachment.attachment = Hinge(-np.pi / 2.0)
    body.core_v1.right.attachment.attachment.attachment.attachment = Brick(0.0)
    part2 = body.core_v1.right.attachment.attachment.attachment.attachment

    part2.left = Hinge(0.0)
    part2.front = Hinge(np.pi / 2.0)
    part2.front.attachment = Brick(-np.pi / 2.0)
    part2.front.attachment.left = Brick(0.0)
    part2.front.attachment.right = Hinge(0.0)
    part2.front.attachment.front = Hinge(np.pi / 2.0)
    part2.front.attachment.front.attachment = Brick(-np.pi / 2.0)
    part2.front.attachment.front.attachment.left = Hinge(0.0)
    part2.front.attachment.front.attachment.right = Hinge(0.0)

    return body


def queen_v1() -> CoreModule:
    """
    Get the queen modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.back = Hinge(np.pi / 2.0)
    body.core_v1.right = Hinge(np.pi / 2.0)
    body.core_v1.right.attachment = Hinge(0.0)
    body.core_v1.right.attachment.attachment = Hinge(-np.pi / 2.0)
    body.core_v1.right.attachment.attachment.attachment = Brick(0.0)
    part2 = body.core_v1.right.attachment.attachment.attachment

    part2.left = Hinge(0.0)
    part2.right = Brick(0.0)
    part2.right.front = Brick(0.0)
    part2.right.front.left = Hinge(0.0)
    part2.right.front.right = Hinge(0.0)

    part2.right.right = Brick(0.0)
    part2.right.right.front = Hinge(np.pi / 2.0)
    part2.right.right.front.attachment = Hinge(0.0)

    return body


def squarish_v1() -> CoreModule:
    """
    Get the squarish modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.back = Hinge(0.0)
    body.core_v1.back.attachment = Brick(0.0)
    body.core_v1.back.attachment.front = Hinge(0.0)
    body.core_v1.back.attachment.left = Hinge(np.pi / 2.0)
    body.core_v1.back.attachment.left.attachment = Brick(-np.pi / 2.0)
    body.core_v1.back.attachment.left.attachment.left = Brick(0.0)
    part2 = body.core_v1.back.attachment.left.attachment.left

    part2.left = Hinge(np.pi / 2.0)
    part2.front = Hinge(0.0)
    part2.right = Hinge(np.pi / 2.0)
    part2.right.attachment = Brick(-np.pi / 2.0)
    part2.right.attachment.left = Brick(0.0)
    part2.right.attachment.left.left = Brick(0.0)

    return body


def snake_v1() -> CoreModule:
    """
    Get the snake modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.left = Hinge(0.0)
    body.core_v1.left.attachment = Brick(0.0)
    body.core_v1.left.attachment.front = Hinge(np.pi / 2.0)
    body.core_v1.left.attachment.front.attachment = Brick(-np.pi / 2.0)
    body.core_v1.left.attachment.front.attachment.front = Hinge(0.0)
    body.core_v1.left.attachment.front.attachment.front.attachment = Brick(0.0)
    body.core_v1.left.attachment.front.attachment.front.attachment.front = (
        Hinge(np.pi / 2.0)
    )
    body.core_v1.left.attachment.front.attachment.front.attachment.front.attachment = (
        Brick(-np.pi / 2.0)
    )
    body.core_v1.left.attachment.front.attachment.front.attachment.front.attachment.front = Hinge(
        0.0
    )
    body.core_v1.left.attachment.front.attachment.front.attachment.front.attachment.front.attachment = Brick(
        0.0
    )
    body.core_v1.left.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = Hinge(
        np.pi / 2.0
    )
    body.core_v1.left.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = Brick(
        -np.pi / 2.0
    )
    body.core_v1.left.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = Hinge(
        0.0
    )
    body.core_v1.left.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = Brick(
        0.0
    )
    body.core_v1.left.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = Hinge(
        np.pi / 2.0
    )

    return body


def stingray_v1() -> CoreModule:
    """
    Get the stingray modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.back = Hinge(np.pi / 2.0)
    body.core_v1.right = Hinge(np.pi / 2.0)
    body.core_v1.right.attachment = Hinge(-np.pi / 2.0)
    body.core_v1.right.attachment.attachment = Brick(0.0)
    body.core_v1.right.attachment.attachment.right = Brick(0.0)
    body.core_v1.right.attachment.attachment.left = Hinge(0.0)
    body.core_v1.right.attachment.attachment.front = Brick(0.0)
    body.core_v1.right.attachment.attachment.front.right = Hinge(np.pi / 2.0)
    body.core_v1.right.attachment.attachment.front.front = Hinge(np.pi / 2.0)
    body.core_v1.right.attachment.attachment.front.left = Hinge(0.0)
    body.core_v1.right.attachment.attachment.front.left.attachment = Brick(0.0)
    body.core_v1.right.attachment.attachment.front.left.attachment.right = (
        Hinge(np.pi / 2.0)
    )
    body.core_v1.right.attachment.attachment.front.left.attachment.front = (
        Hinge(0.0)
    )
    body.core_v1.right.attachment.attachment.front.left.attachment.front.attachment = (
        Brick(0.0)
    )

    return body


def tinlicker_v1() -> CoreModule:
    """
    Get the tinlicker modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.right = Hinge(np.pi / 2.0)
    body.core_v1.right.attachment = Hinge(0.0)
    body.core_v1.right.attachment.attachment = Hinge(0.0)
    body.core_v1.right.attachment.attachment.attachment = Hinge(-np.pi / 2.0)
    body.core_v1.right.attachment.attachment.attachment.attachment = Brick(0.0)
    part2 = body.core_v1.right.attachment.attachment.attachment.attachment

    part2.left = Brick(0.0)
    part2.left.front = Hinge(np.pi / 2.0)
    part2.left.right = Brick(0.0)
    part2.left.right.left = Brick(0.0)
    part2.left.right.front = Hinge(0.0)
    part2.left.right.front.attachment = Brick(0.0)
    part2.left.right.front.attachment.front = Hinge(np.pi / 2.0)
    part2.left.right.front.attachment.right = Brick(0.0)
    part2.left.right.front.attachment.right.right = Hinge(np.pi / 2.0)

    return body


def turtle_v1() -> CoreModule:
    """
    Get the turtle modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.left = Brick(0.0)
    body.core_v1.left.right = Hinge(0.0)
    body.core_v1.left.left = Hinge(np.pi / 2.0)
    body.core_v1.left.left.attachment = Hinge(-np.pi / 2.0)
    body.core_v1.left.left.attachment.attachment = Brick(0.0)

    body.core_v1.left.left.attachment.attachment.front = Brick(0.0)
    body.core_v1.left.left.attachment.attachment.left = Hinge(np.pi / 2.0)
    body.core_v1.left.left.attachment.attachment.right = Hinge(0.0)
    body.core_v1.left.left.attachment.attachment.right.attachment = Brick(0.0)
    part2 = body.core_v1.left.left.attachment.attachment.right.attachment

    part2.left = Hinge(np.pi / 2.0)
    part2.left.attachment = Hinge(-np.pi / 2.0)
    part2.front = Brick(0.0)
    part2.right = Hinge(0.0)
    part2.right.attachment = Brick(0.0)
    part2.right.attachment.right = Hinge(0.0)
    part2.right.attachment.left = Hinge(np.pi / 2.0)
    part2.right.attachment.left.attachment = Hinge(-np.pi / 2.0)
    part2.right.attachment.left.attachment.attachment = Hinge(0.0)
    part2.right.attachment.left.attachment.attachment.attachment = Hinge(0.0)

    return body


def ww_v1() -> CoreModule:
    """
    Get the ww modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.back = Hinge(0.0)
    body.core_v1.right = Hinge(np.pi / 2.0)
    body.core_v1.right.attachment = Hinge(0.0)
    body.core_v1.right.attachment.attachment = Hinge(-np.pi / 2.0)
    body.core_v1.right.attachment.attachment.attachment = Brick(0.0)
    body.core_v1.right.attachment.attachment.attachment.left = Hinge(0.0)
    body.core_v1.right.attachment.attachment.attachment.left.attachment = Brick(0.0)
    part2 = body.core_v1.right.attachment.attachment.attachment.left.attachment

    part2.left = Hinge(0.0)
    part2.front = Brick(0.0)
    part2.front.right = Hinge(np.pi / 2.0)
    part2.front.right.attachment = Brick(-np.pi / 2.0)
    part2.front.right.attachment.left = Hinge(np.pi / 2.0)
    part2.front.right.attachment.left.attachment = Hinge(0.0)
    part2.front.right.attachment.left.attachment.attachment = Hinge(
        -np.pi / 2.0
    )

    return body


def zappa_v1() -> CoreModule:
    """
    Get the zappa modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.back = Hinge(0.0)
    body.core_v1.right = Hinge(np.pi / 2.0)
    body.core_v1.right.attachment = Hinge(0.0)
    body.core_v1.right.attachment.attachment = Hinge(0.0)
    body.core_v1.right.attachment.attachment.attachment = Hinge(-np.pi / 2.0)
    body.core_v1.right.attachment.attachment.attachment.attachment = Hinge(0.0)
    body.core_v1.right.attachment.attachment.attachment.attachment.attachment = Brick(
        0.0
    )
    part2 = body.core_v1.right.attachment.attachment.attachment.attachment.attachment

    part2.front = Hinge(0.0)
    part2.front.attachment = Hinge(0.0)
    part2.left = Hinge(np.pi / 2.0)
    part2.left.attachment = Brick(-np.pi / 2.0)
    part2.left.attachment.left = Hinge(0.0)
    part2.left.attachment.left.attachment = Brick(0.0)
    part2.left.attachment.front = Hinge(0.0)

    return body


CanonicalBodies = Enum('CanonicalBodies', get_all())
