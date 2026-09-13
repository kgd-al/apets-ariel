import math
from typing import Tuple, Type, Optional

import numpy as np
from mujoco import mjtCamLight, MjModel, MjData, MjSpec, mjtGeom, MjsCamera, mju_euler2Quat, mju_rotVecQuat, \
    mju_negQuat, mju_mulQuat, mjtProjection, mjtDisableBit, mj_forward

from ariel.simulation.environments import SimpleFlatWorld, BaseWorld
from .config import ViewerConfig
from .mujoco.state import MjState

def sync_attrs(dst, src, *attrs):
    """Copy each named attribute from src to dst, skipping any src lacks."""
    for attr in attrs:
        if hasattr(src, attr) and hasattr(dst, attr):
            setattr(dst, attr, getattr(src, attr))


def make_world(
    robot: MjSpec,
    robot_name: str = "apet",
    camera_zoom: Optional[float] = None,
    camera_centered: bool = True,
    camera_angle: int = 90,
    show_start: bool = False,
    world_class: Type[BaseWorld] = SimpleFlatWorld,
    adjust_elevation: bool = True,
    filter_parent_child_collisions = True,
    **kwargs
):
    """ Make a simple flat world object

    robot: The mj specifications of the robot to spawn in the world
    camera_zoom: How much of the tracking camera should be taken by the robot
    camera_centered: Whether to center the camera at the robot center
    camera_angle: Angle between floor and camera
    filter_parent_child_collisions: Whether to check for parent-child collisions (disabled by default)
    """

    world = world_class(**kwargs, load_precompiled=False)
    robot = robot.copy()
    aabb = world.get_aabb(robot, "")

    # Always carry over child specifications (if present)
    sync_attrs(world.spec.option, robot.option, "timestep", "impratio", "cone")
    sync_attrs(world.spec.compiler, robot.compiler, "meshdir")
    sync_attrs(world.spec, robot, "nkey")

    # Adjust spawn elevation
    if adjust_elevation:
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
        camera_args = dict(proj=mjtProjection.mjPROJ_ORTHOGRAPHIC, fovy=camera_distance / camera_zoom)
    else:
        camera_args = dict(proj=mjtProjection.mjPROJ_PERSPECTIVE)

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

    # Adjust size
    world.spec.stat.center = [0, 0, 0]
    world.spec.stat.extent = max(world_class.floor_size[:2])

    if not filter_parent_child_collisions:
        world.spec.option.disableflags = (
            int(world.spec.option.disableflags)
            | int(mjtDisableBit.mjDSBL_FILTERPARENT)
        )

    return world


def adjust_shoulder_camera(world: MjSpec, config: ViewerConfig, robot: str, orthographic: bool, camera_fov=62.2):
    camera: MjsCamera = world.camera(config.camera)
    if camera is None:
        raise ValueError(f"Requested camera '{config.camera}' does not exist in\n{world.to_xml()}")

    camera.proj = mjtProjection.mjPROJ_PERSPECTIVE
    camera.mode = mjtCamLight.mjCAMLIGHT_FIXED

    angle = math.radians(config.camera_angle)
    if config.camera_distance is not None:
        camera.pos[0] = -config.camera_distance
        camera.pos[1] = 0
        camera.pos[2] = math.cos(angle) * config.camera_distance
        camera.fovy = camera_fov

    mju_euler2Quat(camera.quat, [0, -np.pi/2 + angle, -np.pi/2], "xyz")


def adjust_side_camera(
        world: MjSpec,
        config: ViewerConfig,
        robot: str,
        orthographic: bool = False):
    camera: MjsCamera = world.camera(config.camera)
    if camera is None:
        raise ValueError(f"Requested camera '{config.camera}' does not exist in\n{world.to_xml()}")

    camera.proj = mjtProjection.mjPROJ_ORTHOGRAPHIC if orthographic else mjtProjection.mjPROJ_PERSPECTIVE

    if config.camera_distance is not None:
        if orthographic:
            camera.fovy = config.camera_distance
        else:
            camera.pos[2] = config.camera_distance
            camera.fovy = 45

    if config.camera_angle is not None:
        angle = np.deg2rad(90 - config.camera_angle)
        mju_euler2Quat(camera.quat, [angle, 0, 0], "xyz")

        invert_parent_quat = camera.parent.quat.copy()
        mju_negQuat(invert_parent_quat, camera.parent.quat)
        mju_mulQuat(camera.quat, invert_parent_quat, camera.quat)

        mju_rotVecQuat(camera.pos, camera.pos, camera.quat)
        # kgd_debug(f"{invert_parent_quat=}")
        # kgd_debug(f"{config.camera_angle=}")
        # kgd_debug(f"{camera.pos=}")
        # kgd_debug(f"{camera.quat=}")

    match config.camera_center:
        case "core":
            camera.pos[0] += 0
        case "com":
            aabb = SimpleFlatWorld.get_aabb(world, robot)
            camera.pos[0:1] += .5 * (aabb[1][0:1] + aabb[0][0:1])
            # kgd_debug(f"{camera.pos=}")


def compile_world(world: BaseWorld) -> Tuple[MjState, MjModel, MjData]:
    # Wasteful but safer
    state = MjState.from_spec(MjSpec.from_string(world.spec.to_xml()))
    mj_forward(state.model, state.data)
    return state, state.model, state.data
