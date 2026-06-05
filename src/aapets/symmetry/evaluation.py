import numpy as np
from functools import lru_cache
from mujoco import mj_step, MjSpec

from .config import Config
from .types import Individual
from .worlds import default_world
from ..common import morphological_measures
from ..common.canonical_bodies import CanonicalBodies
from ..common.controllers.ABCpg import ABCpg
from ..common.monitors import XSpeedMonitor
from ..common.monitors.metrics_storage import EvaluationMetrics
from ..common.mujoco.callback import MjcbCallbacks
from ..common.robot_storage import RerunnableRobot
from ..common.world_builder import compile_world


def save_robot(ind: Individual, metrics: EvaluationMetrics, config: Config, name: str = "champion"):
    path = config.data_folder.joinpath(f"{name}.zip")
    world = default_world(ind.body, config.robot_name_prefix)
    RerunnableRobot(
        mj_spec=world.spec,
        brain=(ABCpg.name(), dict(), ind.weights),
        metrics=metrics,
        misc=dict(),
        config=config
    ).save(path)
    return path


@lru_cache(maxsize=1)
def descriptor_names():
    return (
        list(morphological_measures.measure(CanonicalBodies.SPIDER.get().spec).major_metrics.keys())
        + ["Speed"]
    )


def forward_locomotion(ind: Individual, config: Config, return_metrics: bool):
    robot = MjSpec.from_string(ind.body)
    world = default_world(robot, config.robot_name_prefix)
    state, model, data = compile_world(world)

    brain = ABCpg.from_weights(ind.weights, state, name=config.robot_name_prefix)

    robot_name = f"{config.robot_name_prefix}1"
    fitness_monitor = XSpeedMonitor(robot_name, stepwise=False)
    monitors = {
        m.name(): m for m in [fitness_monitor]
    }

    with MjcbCallbacks(state, [brain], monitors, config):
        mj_step(model, data, nstep=int(config.duration / model.opt.timestep))

    fitness = float(fitness_monitor.value)

    descriptors = morphological_measures.measure(robot).major_metrics
    descriptors["xspeed"] = float(np.tanh(5*max(0, fitness)))

    if return_metrics:
        return fitness, EvaluationMetrics(dict(
            xspeed=fitness,
        ))
    else:
        return np.array([fitness]), np.array(list(descriptors.values()))
