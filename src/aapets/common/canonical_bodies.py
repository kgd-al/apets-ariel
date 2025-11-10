import math
import sys
from enum import Enum
from typing import Callable, Type

import numpy as np

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule

current_module = sys.modules[__name__]

SUBMODULES = HingeModule | BrickModule
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


# def body_spider_graph():
#     graph = nx.DiGraph()
#
#     nodes: List[Tuple[int, ModuleType, ModuleRotationsIdx]] = [
#         (0, ModuleType.CORE, ModuleRotationsIdx.DEG_0)
#     ]
#     edges: List[Tuple[int, int, ModuleFaces]] = []
#     for i, f in enumerate([
#         ModuleFaces.FRONT, ModuleFaces.LEFT, ModuleFaces.BACK, ModuleFaces.RIGHT,
#     ], start=1):
#         edges.append((0, i, f))
#         nodes.append((i, ModuleType.HINGE, ModuleRotationsIdx.DEG_0))
#         edges.append((i, 4+i, ModuleFaces.FRONT))
#         nodes.append((4+i, ModuleType.BRICK, ModuleRotationsIdx.DEG_0))
#         edges.append((4+i, 8+i, ModuleFaces.FRONT))
#         nodes.append((8+i, ModuleType.HINGE, ModuleRotationsIdx.DEG_90))
#         edges.append((8+i, 12+i, ModuleFaces.FRONT))
#         nodes.append((12+i, ModuleType.BRICK, ModuleRotationsIdx.DEG_0))
#
#     for i, m_type, r_type in nodes:
#         graph.add_node(i, type=m_type.name, rotation=r_type.name)
#
#     for src, dst, face in edges:
#         graph.add_edge(src, dst, face=face.name)
#
#     return graph


def make_core():
    core = CoreModule(index=0)
    core.name = "C"
    return core


class Attacher:
    def __init__(self, start_index=1):
        self.index = start_index

    def __call__(self,
                 parent: MODULES,
                 face: ModuleFaces,
                 module_t: Type[SUBMODULES],
                 name: str,
                 rotation: float = 0) -> SUBMODULES:

        module = module_t(self.index)
        self.index += 1

        name = f"{parent.name}-{name}"
        parent.sites[face].attach_body(body=module.body, prefix=name + "-")

        module.name = name

        if rotation != 0:
            module.rotate(rotation)

        return module


def body_spider() -> CoreModule:
    core, attacher = make_core(), Attacher()

    for f in faces:
        h0 = attacher(core, f, HingeModule, f"{f.name[0]}H")
        b0 = attacher(h0, F, BrickModule, "B")
        h1 = attacher(b0, F, HingeModule, "H", rotation=90)
        b1 = attacher(h1, F, BrickModule, "B")

    return core


def body_spider45() -> CoreModule:
    core = body_spider()
    core.spec.body("core").quat = (math.cos(math.pi / 8), 0, 0, math.sin(math.pi / 8))
    return core


def body_gecko() -> CoreModule:
    """
    Get the gecko modular robot.

    :returns: the robot.
    """
    core, attacher = make_core(), Attacher()

    al = attacher(core, L, HingeModule, "LH", rotation=90)
    attacher(al, F, BrickModule, "LB")

    ar = attacher(core, R, HingeModule, "RH", rotation=90)
    attacher(ar, F, BrickModule, "RB")

    sh0 = attacher(core, B, HingeModule, "S")
    sb0 = attacher(sh0, F, BrickModule, "S")
    sh1 = attacher(sb0, F, HingeModule, "S")
    sb1 = attacher(sh1, F, BrickModule, "S")

    ll = attacher(sb1, L, HingeModule, "LH", rotation=90)
    attacher(ll, F, BrickModule, "LB")

    lr = attacher(sb1, R, HingeModule, "RH", rotation=90)
    attacher(lr, F, BrickModule, "RB")

    return core


def body_babya() -> CoreModule:
    """
    Get the babya modular robot.

    :returns: the robot.
    """
    core, attacher = make_core(), Attacher()

    al = attacher(core, L, HingeModule, "LH", rotation=90)
    attacher(al, F, BrickModule, "LB")

    h0 = attacher(core, R, HingeModule, f"RH")
    b0 = attacher(h0, F, BrickModule, "B")
    h1 = attacher(b0, F, HingeModule, "H", rotation=90)
    b1 = attacher(h1, F, BrickModule, "B")

    sh0 = attacher(core, B, HingeModule, "S")
    sb0 = attacher(sh0, F, BrickModule, "S")
    sh1 = attacher(sb0, F, HingeModule, "S")
    sb1 = attacher(sh1, F, BrickModule, "S")

    ll = attacher(sb1, L, HingeModule, "LH", rotation=90)
    attacher(ll, F, BrickModule, "LB")

    lr = attacher(sb1, R, HingeModule, "RH", rotation=90)
    attacher(lr, F, BrickModule, "RB")

    return core


def ant_v1() -> CoreModule:
    """
    Get the ant modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.left = HingeModule(0.0)
    body.core_v1.left.attachment = BrickModule(0.0)

    body.core_v1.right = HingeModule(0.0)
    body.core_v1.right.attachment = BrickModule(0.0)

    body.core_v1.back = HingeModule(np.pi / 2.0)
    body.core_v1.back.attachment = BrickModule(-np.pi / 2.0)
    body.core_v1.back.attachment.left = HingeModule(0.0)
    body.core_v1.back.attachment.left.attachment = BrickModule(0.0)
    body.core_v1.back.attachment.right = HingeModule(0.0)
    body.core_v1.back.attachment.right.attachment = BrickModule(0.0)

    body.core_v1.back.attachment.front = HingeModule(np.pi / 2.0)
    body.core_v1.back.attachment.front.attachment = BrickModule(-np.pi / 2.0)
    body.core_v1.back.attachment.front.attachment.left = HingeModule(0.0)
    body.core_v1.back.attachment.front.attachment.left.attachment = BrickModule(0.0)
    body.core_v1.back.attachment.front.attachment.right = HingeModule(0.0)
    body.core_v1.back.attachment.front.attachment.right.attachment = BrickModule(0.0)

    return body


def salamander_v1() -> CoreModule:
    """
    Get the salamander modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.left = HingeModule(np.pi / 2.0)
    body.core_v1.left.attachment = HingeModule(-np.pi / 2.0)

    body.core_v1.right = HingeModule(0.0)

    body.core_v1.back = HingeModule(np.pi / 2.0)
    body.core_v1.back.attachment = BrickModule(-np.pi / 2.0)
    body.core_v1.back.attachment.left = HingeModule(0.0)
    body.core_v1.back.attachment.front = BrickModule(0.0)
    body.core_v1.back.attachment.front.left = HingeModule(0.0)
    body.core_v1.back.attachment.front.front = HingeModule(np.pi / 2.0)
    body.core_v1.back.attachment.front.front.attachment = BrickModule(-np.pi / 2.0)

    body.core_v1.back.attachment.front.front.attachment.left = HingeModule(0.0)
    body.core_v1.back.attachment.front.front.attachment.left.attachment = BrickModule(0.0)
    body.core_v1.back.attachment.front.front.attachment.left.attachment.left = BrickModule(
        0.0
    )
    body.core_v1.back.attachment.front.front.attachment.left.attachment.front = (
        HingeModule(np.pi / 2.0)
    )
    body.core_v1.back.attachment.front.front.attachment.left.attachment.front.attachment = HingeModule(
        -np.pi / 2.0
    )

    body.core_v1.back.attachment.front.front.attachment.front = BrickModule(0.0)
    body.core_v1.back.attachment.front.front.attachment.front.left = HingeModule(0.0)
    body.core_v1.back.attachment.front.front.attachment.front.front = BrickModule(0.0)
    body.core_v1.back.attachment.front.front.attachment.front.front.left = (
        HingeModule(0.0)
    )
    body.core_v1.back.attachment.front.front.attachment.front.front.front = BrickModule(0.0)
    body.core_v1.back.attachment.front.front.attachment.front.front.front.front = (
        HingeModule(np.pi / 2.0)
    )
    body.core_v1.back.attachment.front.front.attachment.front.front.front.front.attachment = BrickModule(
        -np.pi / 2.0
    )
    body.core_v1.back.attachment.front.front.attachment.front.front.front.front.attachment.left = BrickModule(
        0.0
    )
    body.core_v1.back.attachment.front.front.attachment.front.front.front.front.attachment.front = HingeModule(
        np.pi / 2.0
    )

    return body


def blokky_v1() -> CoreModule:
    """
    Get the blokky modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.left = HingeModule(np.pi / 2.0)
    body.core_v1.back = BrickModule(0.0)
    body.core_v1.back.right = HingeModule(np.pi / 2.0)
    body.core_v1.back.front = HingeModule(np.pi / 2.0)
    body.core_v1.back.front.attachment = HingeModule(-np.pi / 2.0)
    body.core_v1.back.front.attachment.attachment = BrickModule(0.0)
    body.core_v1.back.front.attachment.attachment.front = BrickModule(0.0)
    body.core_v1.back.front.attachment.attachment.front.right = BrickModule(0.0)
    body.core_v1.back.front.attachment.attachment.front.right.left = BrickModule(0.0)
    body.core_v1.back.front.attachment.attachment.front.right.front = BrickModule(0.0)
    body.core_v1.back.front.attachment.attachment.right = BrickModule(0.0)
    body.core_v1.back.front.attachment.attachment.right.front = BrickModule(0.0)
    body.core_v1.back.front.attachment.attachment.right.front.right = BrickModule(0.0)
    body.core_v1.back.front.attachment.attachment.right.front.front = HingeModule(0.0)

    return body


def park_v1() -> CoreModule:
    """
    Get the park modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.back = HingeModule(np.pi / 2.0)
    body.core_v1.back.attachment = HingeModule(-np.pi / 2.0)
    body.core_v1.back.attachment.attachment = BrickModule(0.0)
    body.core_v1.back.attachment.attachment.right = BrickModule(0.0)
    body.core_v1.back.attachment.attachment.left = HingeModule(0.0)
    body.core_v1.back.attachment.attachment.front = BrickModule(0.0)
    body.core_v1.back.attachment.attachment.front.right = HingeModule(-np.pi / 2.0)
    body.core_v1.back.attachment.attachment.front.front = HingeModule(-np.pi / 2.0)
    body.core_v1.back.attachment.attachment.front.left = HingeModule(0.0)
    body.core_v1.back.attachment.attachment.front.left.attachment = BrickModule(0.0)
    body.core_v1.back.attachment.attachment.front.left.attachment.right = HingeModule(
        -np.pi / 2.0
    )
    body.core_v1.back.attachment.attachment.front.left.attachment.left = BrickModule(0.0)
    body.core_v1.back.attachment.attachment.front.left.attachment.front = HingeModule(
        0.0
    )
    body.core_v1.back.attachment.attachment.front.left.attachment.front.attachment = (
        BrickModule(0.0)
    )

    return body


def babyb_v1() -> CoreModule:
    """
    Get the babyb modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.left = HingeModule(np.pi / 2.0)
    body.core_v1.left.attachment = BrickModule(-np.pi / 2.0)
    body.core_v1.left.attachment.front = HingeModule(0.0)
    body.core_v1.left.attachment.front.attachment = BrickModule(0.0)
    body.core_v1.left.attachment.front.attachment.front = HingeModule(np.pi / 2.0)
    body.core_v1.left.attachment.front.attachment.front.attachment = BrickModule(0.0)

    body.core_v1.right = HingeModule(np.pi / 2.0)
    body.core_v1.right.attachment = BrickModule(-np.pi / 2.0)
    body.core_v1.right.attachment.front = HingeModule(0.0)
    body.core_v1.right.attachment.front.attachment = BrickModule(0.0)
    body.core_v1.right.attachment.front.attachment.front = HingeModule(np.pi / 2.0)
    body.core_v1.right.attachment.front.attachment.front.attachment = BrickModule(0.0)

    body.core_v1.front = HingeModule(np.pi / 2.0)
    body.core_v1.front.attachment = BrickModule(-np.pi / 2.0)
    body.core_v1.front.attachment.front = HingeModule(0.0)
    body.core_v1.front.attachment.front.attachment = BrickModule(0.0)
    body.core_v1.front.attachment.front.attachment.front = HingeModule(np.pi / 2.0)
    body.core_v1.front.attachment.front.attachment.front.attachment = BrickModule(0.0)

    body.core_v1.back = HingeModule(np.pi / 2.0)
    body.core_v1.back.attachment = BrickModule(-np.pi / 2.0)

    return body


def garrix_v1() -> CoreModule:
    """
    Get the garrix modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.front = HingeModule(np.pi / 2.0)

    body.core_v1.left = HingeModule(np.pi / 2.0)
    body.core_v1.left.attachment = HingeModule(0.0)
    body.core_v1.left.attachment.attachment = HingeModule(-np.pi / 2.0)
    body.core_v1.left.attachment.attachment.attachment = BrickModule(0.0)
    body.core_v1.left.attachment.attachment.attachment.front = BrickModule(0.0)
    body.core_v1.left.attachment.attachment.attachment.left = HingeModule(0.0)

    part2 = BrickModule(0.0)
    part2.right = HingeModule(np.pi / 2.0)
    part2.front = HingeModule(np.pi / 2.0)
    part2.left = HingeModule(0.0)
    part2.left.attachment = HingeModule(np.pi / 2.0)
    part2.left.attachment.attachment = HingeModule(-np.pi / 2.0)
    part2.left.attachment.attachment.attachment = BrickModule(0.0)

    body.core_v1.left.attachment.attachment.attachment.left.attachment = part2

    return body


def insect_v1() -> CoreModule:
    """
    Get the insect modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.right = HingeModule(np.pi / 2.0)
    body.core_v1.right.attachment = HingeModule(-np.pi / 2.0)
    body.core_v1.right.attachment.attachment = BrickModule(0.0)
    body.core_v1.right.attachment.attachment.right = HingeModule(0.0)
    body.core_v1.right.attachment.attachment.front = HingeModule(np.pi / 2.0)
    body.core_v1.right.attachment.attachment.left = HingeModule(np.pi / 2.0)
    body.core_v1.right.attachment.attachment.left.attachment = BrickModule(-np.pi / 2.0)
    body.core_v1.right.attachment.attachment.left.attachment.front = HingeModule(
        np.pi / 2.0
    )
    body.core_v1.right.attachment.attachment.left.attachment.right = HingeModule(0.0)
    body.core_v1.right.attachment.attachment.left.attachment.right.attachment = (
        HingeModule(0.0)
    )
    body.core_v1.right.attachment.attachment.left.attachment.right.attachment.attachment = HingeModule(
        np.pi / 2.0
    )

    return body


def linkin_v1() -> CoreModule:
    """
    Get the linkin modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.back = HingeModule(0.0)

    body.core_v1.right = HingeModule(np.pi / 2.0)
    body.core_v1.right.attachment = HingeModule(0.0)
    body.core_v1.right.attachment.attachment = HingeModule(0.0)
    body.core_v1.right.attachment.attachment.attachment = HingeModule(-np.pi / 2.0)
    body.core_v1.right.attachment.attachment.attachment.attachment = BrickModule(0.0)

    part2 = body.core_v1.right.attachment.attachment.attachment.attachment
    part2.front = BrickModule(0.0)

    part2.left = HingeModule(0.0)
    part2.left.attachment = HingeModule(0.0)

    part2.right = HingeModule(np.pi / 2.0)
    part2.right.attachment = HingeModule(-np.pi / 2.0)
    part2.right.attachment.attachment = HingeModule(0.0)
    part2.right.attachment.attachment.attachment = HingeModule(np.pi / 2.0)
    part2.right.attachment.attachment.attachment.attachment = HingeModule(0.0)

    return body


def longleg_v1() -> CoreModule:
    """
    Get the longleg modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.left = HingeModule(np.pi / 2.0)
    body.core_v1.left.attachment = HingeModule(0.0)
    body.core_v1.left.attachment.attachment = HingeModule(0.0)
    body.core_v1.left.attachment.attachment.attachment = HingeModule(-np.pi / 2.0)
    body.core_v1.left.attachment.attachment.attachment.attachment = HingeModule(0.0)
    body.core_v1.left.attachment.attachment.attachment.attachment.attachment = BrickModule(
        0.0
    )

    part2 = body.core_v1.left.attachment.attachment.attachment.attachment.attachment
    part2.right = HingeModule(0.0)
    part2.front = HingeModule(0.0)
    part2.left = HingeModule(np.pi / 2.0)
    part2.left.attachment = HingeModule(-np.pi / 2.0)
    part2.left.attachment.attachment = BrickModule(0.0)
    part2.left.attachment.attachment.right = HingeModule(np.pi / 2.0)
    part2.left.attachment.attachment.left = HingeModule(np.pi / 2.0)
    part2.left.attachment.attachment.left.attachment = HingeModule(0.0)

    return body


def penguin_v1() -> CoreModule:
    """
    Get the penguin modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.right = BrickModule(0.0)
    body.core_v1.right.left = HingeModule(np.pi / 2.0)
    body.core_v1.right.left.attachment = HingeModule(-np.pi / 2.0)
    body.core_v1.right.left.attachment.attachment = BrickModule(0.0)
    body.core_v1.right.left.attachment.attachment.right = HingeModule(0.0)
    body.core_v1.right.left.attachment.attachment.left = HingeModule(np.pi / 2.0)
    body.core_v1.right.left.attachment.attachment.left.attachment = HingeModule(
        -np.pi / 2.0
    )
    body.core_v1.right.left.attachment.attachment.left.attachment.attachment = (
        HingeModule(np.pi / 2.0)
    )
    body.core_v1.right.left.attachment.attachment.left.attachment.attachment.attachment = BrickModule(
        -np.pi / 2.0
    )

    part2 = (
        body.core_v1.right.left.attachment.attachment.left.attachment.attachment.attachment
    )

    part2.front = HingeModule(np.pi / 2.0)
    part2.front.attachment = BrickModule(-np.pi / 2.0)

    part2.right = HingeModule(0.0)
    part2.right.attachment = HingeModule(0.0)
    part2.right.attachment.attachment = HingeModule(np.pi / 2.0)
    part2.right.attachment.attachment.attachment = BrickModule(-np.pi / 2.0)

    part2.right.attachment.attachment.attachment.left = HingeModule(np.pi / 2.0)

    part2.right.attachment.attachment.attachment.right = BrickModule(0.0)
    part2.right.attachment.attachment.attachment.right.front = HingeModule(
        np.pi / 2.0
    )

    return body


def pentapod_v1() -> CoreModule:
    """
    Get the pentapod modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.right = HingeModule(np.pi / 2.0)
    body.core_v1.right.attachment = HingeModule(0.0)
    body.core_v1.right.attachment.attachment = HingeModule(0.0)
    body.core_v1.right.attachment.attachment.attachment = HingeModule(-np.pi / 2.0)
    body.core_v1.right.attachment.attachment.attachment.attachment = BrickModule(0.0)
    part2 = body.core_v1.right.attachment.attachment.attachment.attachment

    part2.left = HingeModule(0.0)
    part2.front = HingeModule(np.pi / 2.0)
    part2.front.attachment = BrickModule(-np.pi / 2.0)
    part2.front.attachment.left = BrickModule(0.0)
    part2.front.attachment.right = HingeModule(0.0)
    part2.front.attachment.front = HingeModule(np.pi / 2.0)
    part2.front.attachment.front.attachment = BrickModule(-np.pi / 2.0)
    part2.front.attachment.front.attachment.left = HingeModule(0.0)
    part2.front.attachment.front.attachment.right = HingeModule(0.0)

    return body


def queen_v1() -> CoreModule:
    """
    Get the queen modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.back = HingeModule(np.pi / 2.0)
    body.core_v1.right = HingeModule(np.pi / 2.0)
    body.core_v1.right.attachment = HingeModule(0.0)
    body.core_v1.right.attachment.attachment = HingeModule(-np.pi / 2.0)
    body.core_v1.right.attachment.attachment.attachment = BrickModule(0.0)
    part2 = body.core_v1.right.attachment.attachment.attachment

    part2.left = HingeModule(0.0)
    part2.right = BrickModule(0.0)
    part2.right.front = BrickModule(0.0)
    part2.right.front.left = HingeModule(0.0)
    part2.right.front.right = HingeModule(0.0)

    part2.right.right = BrickModule(0.0)
    part2.right.right.front = HingeModule(np.pi / 2.0)
    part2.right.right.front.attachment = HingeModule(0.0)

    return body


def squarish_v1() -> CoreModule:
    """
    Get the squarish modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.back = HingeModule(0.0)
    body.core_v1.back.attachment = BrickModule(0.0)
    body.core_v1.back.attachment.front = HingeModule(0.0)
    body.core_v1.back.attachment.left = HingeModule(np.pi / 2.0)
    body.core_v1.back.attachment.left.attachment = BrickModule(-np.pi / 2.0)
    body.core_v1.back.attachment.left.attachment.left = BrickModule(0.0)
    part2 = body.core_v1.back.attachment.left.attachment.left

    part2.left = HingeModule(np.pi / 2.0)
    part2.front = HingeModule(0.0)
    part2.right = HingeModule(np.pi / 2.0)
    part2.right.attachment = BrickModule(-np.pi / 2.0)
    part2.right.attachment.left = BrickModule(0.0)
    part2.right.attachment.left.left = BrickModule(0.0)

    return body


def snake_v1() -> CoreModule:
    """
    Get the snake modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.left = HingeModule(0.0)
    body.core_v1.left.attachment = BrickModule(0.0)
    body.core_v1.left.attachment.front = HingeModule(np.pi / 2.0)
    body.core_v1.left.attachment.front.attachment = BrickModule(-np.pi / 2.0)
    body.core_v1.left.attachment.front.attachment.front = HingeModule(0.0)
    body.core_v1.left.attachment.front.attachment.front.attachment = BrickModule(0.0)
    body.core_v1.left.attachment.front.attachment.front.attachment.front = (
        HingeModule(np.pi / 2.0)
    )
    body.core_v1.left.attachment.front.attachment.front.attachment.front.attachment = (
        BrickModule(-np.pi / 2.0)
    )
    body.core_v1.left.attachment.front.attachment.front.attachment.front.attachment.front = HingeModule(
        0.0
    )
    body.core_v1.left.attachment.front.attachment.front.attachment.front.attachment.front.attachment = BrickModule(
        0.0
    )
    body.core_v1.left.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = HingeModule(
        np.pi / 2.0
    )
    body.core_v1.left.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = BrickModule(
        -np.pi / 2.0
    )
    body.core_v1.left.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = HingeModule(
        0.0
    )
    body.core_v1.left.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment = BrickModule(
        0.0
    )
    body.core_v1.left.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front.attachment.front = HingeModule(
        np.pi / 2.0
    )

    return body


def stingray_v1() -> CoreModule:
    """
    Get the stingray modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.back = HingeModule(np.pi / 2.0)
    body.core_v1.right = HingeModule(np.pi / 2.0)
    body.core_v1.right.attachment = HingeModule(-np.pi / 2.0)
    body.core_v1.right.attachment.attachment = BrickModule(0.0)
    body.core_v1.right.attachment.attachment.right = BrickModule(0.0)
    body.core_v1.right.attachment.attachment.left = HingeModule(0.0)
    body.core_v1.right.attachment.attachment.front = BrickModule(0.0)
    body.core_v1.right.attachment.attachment.front.right = HingeModule(np.pi / 2.0)
    body.core_v1.right.attachment.attachment.front.front = HingeModule(np.pi / 2.0)
    body.core_v1.right.attachment.attachment.front.left = HingeModule(0.0)
    body.core_v1.right.attachment.attachment.front.left.attachment = BrickModule(0.0)
    body.core_v1.right.attachment.attachment.front.left.attachment.right = (
        HingeModule(np.pi / 2.0)
    )
    body.core_v1.right.attachment.attachment.front.left.attachment.front = (
        HingeModule(0.0)
    )
    body.core_v1.right.attachment.attachment.front.left.attachment.front.attachment = (
        BrickModule(0.0)
    )

    return body


def tinlicker_v1() -> CoreModule:
    """
    Get the tinlicker modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.right = HingeModule(np.pi / 2.0)
    body.core_v1.right.attachment = HingeModule(0.0)
    body.core_v1.right.attachment.attachment = HingeModule(0.0)
    body.core_v1.right.attachment.attachment.attachment = HingeModule(-np.pi / 2.0)
    body.core_v1.right.attachment.attachment.attachment.attachment = BrickModule(0.0)
    part2 = body.core_v1.right.attachment.attachment.attachment.attachment

    part2.left = BrickModule(0.0)
    part2.left.front = HingeModule(np.pi / 2.0)
    part2.left.right = BrickModule(0.0)
    part2.left.right.left = BrickModule(0.0)
    part2.left.right.front = HingeModule(0.0)
    part2.left.right.front.attachment = BrickModule(0.0)
    part2.left.right.front.attachment.front = HingeModule(np.pi / 2.0)
    part2.left.right.front.attachment.right = BrickModule(0.0)
    part2.left.right.front.attachment.right.right = HingeModule(np.pi / 2.0)

    return body


def turtle_v1() -> CoreModule:
    """
    Get the turtle modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.left = BrickModule(0.0)
    body.core_v1.left.right = HingeModule(0.0)
    body.core_v1.left.left = HingeModule(np.pi / 2.0)
    body.core_v1.left.left.attachment = HingeModule(-np.pi / 2.0)
    body.core_v1.left.left.attachment.attachment = BrickModule(0.0)

    body.core_v1.left.left.attachment.attachment.front = BrickModule(0.0)
    body.core_v1.left.left.attachment.attachment.left = HingeModule(np.pi / 2.0)
    body.core_v1.left.left.attachment.attachment.right = HingeModule(0.0)
    body.core_v1.left.left.attachment.attachment.right.attachment = BrickModule(0.0)
    part2 = body.core_v1.left.left.attachment.attachment.right.attachment

    part2.left = HingeModule(np.pi / 2.0)
    part2.left.attachment = HingeModule(-np.pi / 2.0)
    part2.front = BrickModule(0.0)
    part2.right = HingeModule(0.0)
    part2.right.attachment = BrickModule(0.0)
    part2.right.attachment.right = HingeModule(0.0)
    part2.right.attachment.left = HingeModule(np.pi / 2.0)
    part2.right.attachment.left.attachment = HingeModule(-np.pi / 2.0)
    part2.right.attachment.left.attachment.attachment = HingeModule(0.0)
    part2.right.attachment.left.attachment.attachment.attachment = HingeModule(0.0)

    return body


def ww_v1() -> CoreModule:
    """
    Get the ww modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.back = HingeModule(0.0)
    body.core_v1.right = HingeModule(np.pi / 2.0)
    body.core_v1.right.attachment = HingeModule(0.0)
    body.core_v1.right.attachment.attachment = HingeModule(-np.pi / 2.0)
    body.core_v1.right.attachment.attachment.attachment = BrickModule(0.0)
    body.core_v1.right.attachment.attachment.attachment.left = HingeModule(0.0)
    body.core_v1.right.attachment.attachment.attachment.left.attachment = BrickModule(0.0)
    part2 = body.core_v1.right.attachment.attachment.attachment.left.attachment

    part2.left = HingeModule(0.0)
    part2.front = BrickModule(0.0)
    part2.front.right = HingeModule(np.pi / 2.0)
    part2.front.right.attachment = BrickModule(-np.pi / 2.0)
    part2.front.right.attachment.left = HingeModule(np.pi / 2.0)
    part2.front.right.attachment.left.attachment = HingeModule(0.0)
    part2.front.right.attachment.left.attachment.attachment = HingeModule(
        -np.pi / 2.0
    )

    return body


def zappa_v1() -> CoreModule:
    """
    Get the zappa modular robot.

    :returns: the robot.
    """
    body = CoreModule()

    body.core_v1.back = HingeModule(0.0)
    body.core_v1.right = HingeModule(np.pi / 2.0)
    body.core_v1.right.attachment = HingeModule(0.0)
    body.core_v1.right.attachment.attachment = HingeModule(0.0)
    body.core_v1.right.attachment.attachment.attachment = HingeModule(-np.pi / 2.0)
    body.core_v1.right.attachment.attachment.attachment.attachment = HingeModule(0.0)
    body.core_v1.right.attachment.attachment.attachment.attachment.attachment = BrickModule(
        0.0
    )
    part2 = body.core_v1.right.attachment.attachment.attachment.attachment.attachment

    part2.front = HingeModule(0.0)
    part2.front.attachment = HingeModule(0.0)
    part2.left = HingeModule(np.pi / 2.0)
    part2.left.attachment = BrickModule(-np.pi / 2.0)
    part2.left.attachment.left = HingeModule(0.0)
    part2.left.attachment.left.attachment = BrickModule(0.0)
    part2.left.attachment.front = HingeModule(0.0)

    return body


CanonicalBodies = Enum('CanonicalBodies', get_all())
