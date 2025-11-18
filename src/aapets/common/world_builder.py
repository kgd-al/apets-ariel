import math
import pprint
from typing import Tuple, Type, Optional

import mujoco
from mujoco import mjtCamLight, MjModel, MjData, MjSpec, mjtGeom, mjtLightType

from ariel.simulation.environments import SimpleFlatWorld, BaseWorld


def make_world(
    robot: MjSpec,
    camera_zoom: Optional[float] = 1,
    camera_centered: bool = False,
    camera_angle: int = 90,
    show_start: bool = False,
    world_class: Type[BaseWorld] = SimpleFlatWorld
):
    """ Make a simple flat world object

    robot: The mj specifications of the robot to spawn in the world
    camera_zoom: How much of the tracking camera should be taken by the robot
    camera_centered: Whether to center the camera at the robot center
    camera_angle: Angle between floor and camera
    """

    world = world_class()

    # Adjust spawn elevation
    aabb = world.get_aabb(robot, "")
    robot.worldbody.pos[2] += -aabb[0][2]

    # Place camera
    if camera_centered:
        x0, x1 = aabb[:, 0]
        y0, y1 = aabb[:, 1]
        cx, cy = .5 * (x0 + x1), .5 * (y0 + y1)
        camera_distance = 2 * max(cx - x0, x1 - cx, cy - y0, y1 - cy)
        camera_pos = [cx, cy, camera_distance]

    else:
        camera_distance = 2 * max([-aabb[0, 0], aabb[1, 0], -aabb[0, 1], aabb[1, 1]])
        camera_pos = [0, 0, camera_distance]

    # Adjust for angle
    camera_xy_axes: list[float] = [1, 0, 0, 0, 1, 0]
    c_rad = math.radians(180-camera_angle)
    camera_pos[1] = camera_distance * math.cos(c_rad)
    camera_pos[2] = camera_distance * math.sin(c_rad)

    c_rad -= math.pi / 2
    camera_xy_axes[4] = math.cos(c_rad)
    camera_xy_axes[5] = math.sin(c_rad)

    if camera_zoom is not None:
        camera_args = dict(orthographic=True, fovy=camera_distance / camera_zoom)
    else:
        camera_args = dict(orthographic=False)

    # Add tracking camera
    robot.worldbody.add_camera(
        name=f"tracking-cam",
        mode=mjtCamLight.mjCAMLIGHT_TRACKCOM,
        pos=camera_pos,
        xyaxes=camera_xy_axes,
        **camera_args
    )

    # Overkill but cleaner xml
    for site in robot.sites:
        robot.delete(site)

    # Spawn THE robot (most things would break with two)
    world.spawn(robot, spawn_prefix="apet", correct_collision_with_floor=False)

    # Mark the spawn position
    if show_start:
        world.spec.worldbody.add_site(
            name="site_start",
            size=[.1, .1, .001],
            rgba=[.1, 0., 0, 1.],
            type=mjtGeom.mjGEOM_ELLIPSOID
        )

    # Adjust lighting
    world.spec.visual.headlight.active = False
    light = world.spec.light("light")
    light.castshadow = True
    light.pos = (0, 0, 1)
    light.ambient = (.2, .2, .2)
    light.specular = (0, 0, 0)
    light.mode = mjtCamLight.mjCAMLIGHT_TRACKCOM

    return world


def compile_world(world: BaseWorld) -> Tuple[MjModel, MjData]:
    model = world.spec.compile()
    data = mujoco.MjData(model)
    return model, data
