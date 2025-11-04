from typing import Tuple

import mujoco
from mujoco import mjtCamLight, MjModel, MjData, MjSpec

from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.simulation.environments import SimpleFlatWorld, BaseWorld


def make_world(robot: MjSpec):
    world = SimpleFlatWorld()
    # Evaluator.add_defaults(world.spec)

    robot.worldbody.add_camera(
        name="tracking-cam",
        mode=mjtCamLight.mjCAMLIGHT_TRACKCOM,
        orthographic=True,
        # pos=(-2, 0, 1.5),
        # xyaxes=[0, -1, 0, 0.75, 0, 0.75],
        pos=(0, 0, 2),
        fovy=2,
        xyaxes=[0, -1, 0, 1, 0, 0],
    )

    world.spawn(robot, spawn_prefix="apet", correct_collision_with_floor=True)

    light = world.spec.light("light")
    light.castshadow = True
    light.pos = (0, 0, 2)
    light.ambient = (1, 1, 1)
    world.spec.delete(light)

    return world


def compile_world(world: BaseWorld) -> Tuple[MjModel, MjData]:
    model = world.spec.compile()
    data = mujoco.MjData(model)
    return model, data
