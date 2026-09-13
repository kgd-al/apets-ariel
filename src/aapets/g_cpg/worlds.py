import numpy as np

from mujoco import MjSpec, mjtGeom, mjtJoint

from .config import Config

from ..common.world_builder import make_world


CUSTOM_FLAG = "kgd-custom-xml-flag"


def flag_as_custom(spec: MjSpec):
    spec.add_numeric(name=CUSTOM_FLAG, data=[1.0],
                     info="Denotes an xml provided by a third party. Not an evolutionary product")


def is_custom(spec: MjSpec):
    return any(n.name.endswith(CUSTOM_FLAG) for n in spec.numerics)


def default_world(robot: MjSpec | str, robot_name: str):
    if isinstance(robot, str):
        robot = MjSpec.from_string(robot)

    # Need to disable for circular morphologies
    # Otherwise self-penetrations can be filtered out
    # Ariel already provides exclusion pairs for all rotor/stators
    filter_parent_child_collisions = False

    # When working with generic mujoco specs, robot is already well positioned
    adjust_elevation = (not is_custom(robot))

    return make_world(
        robot, robot_name=robot_name,
        filter_parent_child_collisions=filter_parent_child_collisions,
        adjust_elevation=adjust_elevation
    )


def compliance_worlds(robot: MjSpec | str, config: Config):
    cn = config.controllability_sub_tasks
    cr = config.controllability_range
    d = config.controllability_distance

    worlds = {}
    for i in range(cn):
        w = default_world(robot, config.robot_name_prefix)

        a_d = cr * (-0.5 + i / (cn-1))
        a_r = np.deg2rad(a_d)
        pos = d * np.array([np.cos(a_r), np.sin(a_r), 0])
        add_ball(w.spec, pos, config.controllability_target_name)
        worlds[f"{d}m{a_d:+g}"] = w

    return worlds


def add_ball(specs: MjSpec, pos, name, radius=0.05):
    ball = specs.worldbody.add_body(
        name=name,
        pos=pos + np.array([0, 0, radius]),
        mass=.2,
    )
    ball.add_geom(
        name=name,
        type=mjtGeom.mjGEOM_SPHERE,
        size=(radius, 0, 0),
        rgba=(1, 1, 1, 1),
    )
    ball.add_joint(type=mjtJoint.mjJNT_FREE, stiffness=0, damping=0, frictionloss=.01, armature=0)

