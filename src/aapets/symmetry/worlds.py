from mujoco import MjSpec

from common.world_builder import make_world


def default_world(robot: MjSpec | str, robot_name: str):
    if isinstance(robot, str):
        robot = MjSpec.from_string(robot)
    return make_world(robot, robot_name=robot_name)
