import abc
import logging
from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import Optional

import numpy as np
from mujoco import mj_step, MjSpec

from .config import Config, Task, Symmetry
from .types import Individual, StaticData
from .types import morphological_symmetry
from .worlds import default_world
from ..common import morphological_measures
from ..common.canonical_bodies import CanonicalBodies
from ..common.controllers.abstract import Controller
from ..common.monitors.behavioral import XSpeedMonitor
from ..common.monitors.behavioral import ZSpeedMonitor
from ..common.monitors.metrics_storage import EvaluationMetrics
from ..common.mujoco.callback import MjcbCallbacks
from ..common.mujoco.state import MjState
from ..common.robot_storage import RerunnableRobot
from ..common.world_builder import compile_world


@dataclass
class EvaluationResult:
    fitness: float
    descriptors: Optional[np.ndarray] = None
    metrics: Optional[EvaluationMetrics] = None


class Evaluator(abc.ABC):
    @dataclass
    class State:
        pass

    @classmethod
    @abc.abstractmethod
    def prepare(cls, ind: Individual, config: Config) -> State: ...
    """ Called before evaluating a new morphology (sets up the mujoco state)"""

    @classmethod
    @abc.abstractmethod
    def reset(cls, state: 'State') -> None: ...
    """ Called before evaluating a new brain (resets the mujoco state)"""

    @classmethod
    @abc.abstractmethod
    def evaluate(cls, state: 'State', weights: np.ndarray, config: Config, return_metrics=False) -> EvaluationResult: ...
    """ Called to compute the fitness of the current body/brain"""

    @staticmethod
    @lru_cache(maxsize=1)
    def descriptor_names():
        return (
            list(morphological_measures.measure(CanonicalBodies.SPIDER.get().spec).major_metrics.keys())
            + ["Speed"]
        )

    @classmethod
    def save_robot(cls, ind: Individual, metrics: EvaluationMetrics,
                   config: Config, data: StaticData, name: str = "champion"):
        path = config.data_folder.joinpath(f"{name}.zip")
        world = default_world(ind.body, config.robot_name_prefix)
        RerunnableRobot(
            mj_spec=world.spec,
            brain=(ind.brain_type.name(), dict(), ind.weights),
            metrics=metrics,
            misc=dict(
                genotype=ind.genome,
                genotype_rendering=dict(data=data),
            ),
            config=config
        ).save(path)
        return path


class ForwardLocomotion(Evaluator):
    @dataclass
    class State:
        robot: MjSpec
        state: MjState
        brain: Controller

    @classmethod
    def prepare(cls, ind: Individual, config: Config):
        robot = MjSpec.from_string(ind.body)
        world = default_world(robot, config.robot_name_prefix)
        state, model, data = compile_world(world)

        if config.symmetry is not Symmetry.NONE:
            symmetry = morphological_symmetry(state, config.robot_name_prefix, "body")
            if not symmetry.valid():
                logging.warning(f"Non symmetric robot {ind.id}")
                cls.save_robot(ind, EvaluationMetrics({}), config,
                               SimpleNamespace(config=config),
                               name=f"bad_symmetry_{ind.id}")

        brain = ind.brain_type.from_weights(ind.weights, state, name=config.robot_name_prefix)
        return cls.State(robot, state, brain)

    @classmethod
    def reset(cls, state: State):
        state.state.reset()
        state.brain.reset(state.state)

    @classmethod
    def evaluate(cls, state: State, weights: np.ndarray, config: Config, return_metrics: bool = False):
        robot = state.robot
        brain = state.brain
        state, model, data = state.state.unpacked

        brain.set_weights(weights)

        robot_name = f"{config.robot_name_prefix}1"
        forward_speed = XSpeedMonitor(robot_name, stepwise=False)
        vertical_speed = ZSpeedMonitor(robot_name, stepwise=False)
        monitors = {
            m.name(): m for m in [forward_speed, vertical_speed]
        }

        with MjcbCallbacks(state, [brain], monitors, config):
            mj_step(model, data, nstep=int(config.duration / model.opt.timestep))

        x_speed, z_speed = forward_speed.value, vertical_speed.value
        fitness = float(x_speed - abs(z_speed))

        descriptors = morphological_measures.measure(robot, max_size=config.max_modules).major_metrics
        descriptors["xspeed"] = float(np.tanh(5*max(0, x_speed)))
        descriptors["zspeed"] = float(np.tanh(5*max(0, z_speed)))

        if return_metrics:
            return EvaluationResult(
                fitness=fitness,
                metrics=EvaluationMetrics(dict(
                    xspeed=x_speed,
                    zspeed=z_speed,
                )))
        else:
            return EvaluationResult(
                fitness=fitness,
                descriptors=np.array(list(descriptors.values()))
            )


def evaluator(task: Task):
    return {
        Task.LOCOMOTION: ForwardLocomotion,
        Task.ABCPG: ForwardLocomotion,
    }[task]
