import numpy as np
from mujoco import mj_step, MjSpec
from torchgen import dest

from ariel.utils.morphological_descriptor import MorphologicalMeasures
from common import morphological_measures
from common.morphological_measures import measure
from .config import Config
from .types import Individual
from .worlds import default_world
from ..common.controllers.ABCpg import ABCpg
from ..common.monitors import XSpeedMonitor
from ..common.monitors.metrics_storage import EvaluationMetrics
from ..common.mujoco.callback import MjcbCallbacks
from ..common.robot_storage import RerunnableRobot
from ..common.world_builder import compile_world


def save_robot(ind: Individual, config: Config, name: str = "champion"):
    path = config.data_folder.joinpath(f"{name}.zip")
    world = default_world(ind.body, config.robot_name_prefix)
    RerunnableRobot(
        mj_spec=world.spec,
        brain=(ABCpg.name(), dict(), ind.weights),
        metrics=EvaluationMetrics(dict()),
        misc=dict(),
        config=config
    ).save(path)
    return path


def forward_locomotion(ind: Individual, config: Config):
    robot = MjSpec.from_string(ind.body)
    world = default_world(robot, config.robot_name_prefix)
    state, model, data = compile_world(world)

    brain = ABCpg.from_weights(ind.weights, state, name=config.robot_name_prefix)

    robot_name = f"{config.robot_name_prefix}1"
    fitness = XSpeedMonitor(robot_name, stepwise=False)
    monitors = {
        m.name(): m for m in [fitness]
    }

    with MjcbCallbacks(state, [brain], monitors, config):
        mj_step(model, data, nstep=int(config.duration / model.opt.timestep))

    descriptors = [
        *morphological_measures.measure(robot).major_metrics.values()
    ]

    return np.array([fitness.value]), np.array(descriptors)
