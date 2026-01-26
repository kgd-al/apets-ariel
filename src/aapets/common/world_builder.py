import math
from typing import Tuple, Type, Optional, Literal

import numpy as np
from mujoco import mjtCamLight, MjModel, MjData, MjSpec, mjtGeom, MjsCamera, mju_euler2Quat, mju_rotVecQuat

from ariel.simulation.environments import SimpleFlatWorld, BaseWorld
from ..common.config import ViewerConfig
from ..common.mujoco.state import MjState


def make_world(
    robot: MjSpec,
    robot_name: str = "apet",
    camera_zoom: Optional[float] = None,
    camera_centered: bool = True,
    camera_angle: int = 90,
    show_start: bool = False,
    world_class: Type[BaseWorld] = SimpleFlatWorld,
):
    """ Make a simple flat world object

    robot: The mj specifications of the robot to spawn in the world
    camera_zoom: How much of the tracking camera should be taken by the robot
    camera_centered: Whether to center the camera at the robot center
    camera_angle: Angle between floor and camera
    """

    world = world_class()
    robot = robot.copy()

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
    camera_pos[1] += camera_distance * math.cos(c_rad)
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
    world.spawn(robot, spawn_prefix=robot_name, correct_collision_with_floor=False)

    # Mark the spawn position
    if show_start:
        world.spec.worldbody.add_site(
            name="site_start",
            size=[.1, .1, .001],
            rgba=[0., 0.1, 0.2, 1.],
            type=mjtGeom.mjGEOM_ELLIPSOID
        )

    # Adjust lighting
    world.spec.visual.headlight.active = True
    light = world.spec.light("light")
    light.castshadow = True
    light.pos = (0, 1, 1)
    light.ambient = (.2, .2, .2)
    light.specular = (0, 0, 0)
    light.mode = mjtCamLight.mjCAMLIGHT_TRACKCOM

    return world


def adjust_shoulder_camera(world: MjSpec, args: ViewerConfig, robot: str, orthographic: bool, camera_fov=62.2):
    camera: MjsCamera = world.camera(args.camera)
    if camera is None:
        raise ValueError(f"Requested camera '{args.camera}' does not exist in\n{world.to_xml()}")

    camera.orthographic = False
    camera.mode = mjtCamLight.mjCAMLIGHT_FIXED

    if args.camera_distance is not None:
        camera.pos[0] = -args.camera_distance
        camera.pos[2] = .5 * args.camera_distance
        camera.fovy = camera_fov

    mju_euler2Quat(camera.quat, [0, -np.pi/2, -np.pi/2], "xyz")


def adjust_side_camera(
        world: MjSpec,
        args: ViewerConfig,
        robot: str,
        orthographic: bool = False):

    camera: MjsCamera = world.camera(args.camera)
    if camera is None:
        raise ValueError(f"Requested camera '{args.camera}' does not exist in\n{world.to_xml()}")

    camera.orthographic = orthographic

    if args.camera_distance is not None:
        if orthographic:
            camera.fovy = args.camera_distance
        else:
            camera.pos[2] = args.camera_distance
            camera.fovy = 45

    match args.camera_center:
        case "core":
            camera.pos[0] = 0
        case "com":
            aabb = SimpleFlatWorld.get_aabb(world, robot)
            camera.pos[0] = .5 * (aabb[1][0] + aabb[0][0])

    if args.camera_angle is not None:
        angle = np.deg2rad(90 - args.camera_angle)
        mju_euler2Quat(camera.quat, [angle, 0, 0], "xyz")
        mju_rotVecQuat(camera.pos, camera.pos, camera.quat)


def compile_world(world: BaseWorld) -> Tuple[MjState, MjModel, MjData]:
    state = MjState.from_spec(world.spec)
    return state, state.model, state.data
