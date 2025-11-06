from typing import Tuple

import mujoco
from mujoco import mjtCamLight, MjModel, MjData, MjSpec
from mujoco._enums import mjtGeom

from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.simulation.environments import SimpleFlatWorld, BaseWorld


def make_world(robot: MjSpec, camera_zoom: float = 1):
    world = SimpleFlatWorld()
    # Evaluator.add_defaults(world.spec)

    aabb = world.get_aabb(robot, "")
    print(aabb)

    # Adjust spawn elevation
    robot.worldbody.pos[2] += -aabb[0][2]

    # Add tracking camera
    robot.worldbody.add_camera(
        name="tracking-cam",
        mode=mjtCamLight.mjCAMLIGHT_TRACKCOM,
        orthographic=True,
        # pos=(-2, 0, 1.5),
        # xyaxes=[0, -1, 0, 0.75, 0, 0.75],
        pos=(0, 0, 2),
        fovy=2 * max([-aabb[0, 0], aabb[1, 0], -aabb[0, 1], aabb[1, 1]]) / camera_zoom,
        xyaxes=[1, 0, 0, 0, 1, 0],
    )

    # Overkill but cleaner xml
    for site in robot.sites:
        robot.delete(site)

    # Spawn THE robot (most things would break with two)
    world.spawn(robot, spawn_prefix="apet", correct_collision_with_floor=False)

    # Mark the spawn position
    world.spec.worldbody.add_site(
        name="site_start",
        size=[.1, .1, .005],
        rgba=[.1, 0., 0, 1.],
        type=mjtGeom.mjGEOM_ELLIPSOID
    )

    # Adjust lighting
    world.spec.visual.headlight.active = False
    light = world.spec.light("light")
    light.castshadow = True
    light.pos = (10, 0, 2)
    light.ambient = (.1, .1, .1)
    # world.spec.delete(light)

    print(world.spec.to_xml())

    return world


def compile_world(world: BaseWorld) -> Tuple[MjModel, MjData]:
    model = world.spec.compile()
    data = mujoco.MjData(model)
    return model, data
