from typing import Tuple, Type

import mujoco
from mujoco import mjtCamLight, MjModel, MjData, MjSpec, mjtGeom

from ariel.simulation.environments import SimpleFlatWorld, BaseWorld


def make_world(
    robot: MjSpec,
    camera_zoom: float = 1,
    camera_centered: bool = False,
    world_class: Type[BaseWorld] = SimpleFlatWorld
):
    """ Make a simple flat world object

    robot: The mj specifications of the robot to spawn in the world
    camera_zoom: How much of the tracking camera should be taken by the robot
    camera_centered: Whether to center the camera at the robot center
    """

    world = world_class()

    aabb = world.get_aabb(robot, "")

    if camera_centered:
        x0, x1 = aabb[:, 0]
        y0, y1 = aabb[:, 1]
        cx, cy = .5 * (x0 + x1), .5 * (y0 + y1)
        camera_pos = (cx, cy, 2 * max(cx - x0, x1 - cx, cy - y0, y1 - cy))

    else:
        camera_pos = (0, 0, 2 * max([-aabb[0, 0], aabb[1, 0], -aabb[0, 1], aabb[1, 1]]))

    # Adjust spawn elevation
    robot.worldbody.pos[2] += -aabb[0][2]

    # Add tracking camera
    robot.worldbody.add_camera(
        name="tracking-cam",
        mode=mjtCamLight.mjCAMLIGHT_TRACKCOM,
        orthographic=True,
        # pos=(-2, 0, 1.5),
        # xyaxes=[0, -1, 0, 0.75, 0, 0.75],
        pos=camera_pos,
        fovy=camera_pos[2] / camera_zoom,
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
        size=[.01, .01, .001],
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

    return world


def compile_world(world: BaseWorld) -> Tuple[MjModel, MjData]:
    model = world.spec.compile()
    data = mujoco.MjData(model)
    return model, data
