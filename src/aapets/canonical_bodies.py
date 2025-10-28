import sys
from typing import List, Tuple

import networkx as nx

from ariel.body_phenotypes.robogen_lite.config import ModuleType, ModuleRotationsIdx, ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule

current_module = sys.modules[__name__]

SUBMODULES = HingeModule | BrickModule
MODULES = CoreModule | SUBMODULES


def canonical_body(name: str, *args, **kwargs):
    body_fn = getattr(current_module, f"body_{name}", None)
    if body_fn is None:
        raise RuntimeError(f"'{name}' is not a valid canonical body name")

    return body_fn(*args, **kwargs)


def body_spider_graph(*args, **kwargs):
    graph = nx.DiGraph()

    nodes: List[Tuple[int, ModuleType, ModuleRotationsIdx]] = [
        (0, ModuleType.CORE, ModuleRotationsIdx.DEG_0)
    ]
    edges: List[Tuple[int, int, ModuleFaces]] = []
    for i, f in enumerate([
        ModuleFaces.FRONT, ModuleFaces.LEFT, ModuleFaces.BACK, ModuleFaces.RIGHT,
    ], start=1):
        edges.append((0, i, f))
        nodes.append((i, ModuleType.HINGE, ModuleRotationsIdx.DEG_0))
        edges.append((i, 4+i, ModuleFaces.FRONT))
        nodes.append((4+i, ModuleType.BRICK, ModuleRotationsIdx.DEG_0))
        edges.append((4+i, 8+i, ModuleFaces.FRONT))
        nodes.append((8+i, ModuleType.HINGE, ModuleRotationsIdx.DEG_90))
        edges.append((8+i, 12+i, ModuleFaces.FRONT))
        nodes.append((12+i, ModuleType.BRICK, ModuleRotationsIdx.DEG_0))

    for i, m_type, r_type in nodes:
        graph.add_node(i, type=m_type.name, rotation=r_type.name)

    for src, dst, face in edges:
        graph.add_edge(src, dst, face=face.name)

    return graph


def attach(parent: MODULES,
           face: ModuleFaces,
           module: SUBMODULES,
           name: str) -> SUBMODULES:
    name = f"{parent.name}-{name}"
    parent.sites[face].attach_body(body=module.body, prefix=name + "-")
    module.name = name
    return module


def body_spider(*args, **kwargs) -> CoreModule:
    core = CoreModule(index=0)
    core.name = "C"

    faces = [
        ModuleFaces.FRONT, ModuleFaces.LEFT, ModuleFaces.BACK, ModuleFaces.RIGHT,
    ]
    F, L, B, R = faces

    for i, f in enumerate(faces, start=1):
        h0 = attach(core, f, HingeModule(index=i), f"{f.name[0]}H")
        b0 = attach(h0, F, BrickModule(index=4+i), "B")
        h1 = attach(b0, F, HingeModule(index=8+i), "H")
        b1 = attach(h1, F, BrickModule(index=12+i), "B")

    return core
