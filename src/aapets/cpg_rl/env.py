import functools
from typing import Callable

import numpy as np

from mujoco import mj_step

from .types import Config, Architecture, Rewards
from ..common import canonical_bodies
from ..common.controllers.abstract import Controller
from ..common.controllers.mlp_tensor import mlp_structure, MLPTensorBrain as MLP
from ..common.controllers.neighborhood_cpg import NeighborhoodCPG as CPG
from ..common.monitors import XSpeedMonitor
from ..common.monitors.metrics_storage import EvaluationMetrics
from ..common.mujoco.callback import MjcbCallbacks
from ..common.mujoco.state import MjState
from ..common.robot_storage import RerunnableRobot
from ..common.world_builder import make_world, compile_world

BrainFactory = Callable[[np.ndarray, MjState], Controller]


class EvoEnvironment:
    def __init__(self, config: Config):
        self._config = config
        self._body = canonical_bodies.get(config.body)

        robot_name = config.robot_name_prefix
        self._world = make_world(self._body.spec, robot_name=robot_name, camera_centered=True)
        self._state, _, _ = compile_world(self._world)

        match config.arch:
            case Architecture.CPG:
                neighborhood = config.cpg_neighborhood
                self._params = CPG.num_parameters(self._state, robot_name, neighborhood)
                self._brain_factory = functools.partial(
                    CPG, neighborhood=neighborhood, name=config.robot_name_prefix,
                )
                self._brain_args = (CPG.name(), dict(neighborhood=neighborhood))

            case Architecture.MLP:
                width, depth = config.mlp_width, config.mlp_depth
                self._params = MLP.num_parameters(self._state, robot_name, width, depth)
                self._brain_factory = functools.partial(
                    MLP, width=width, depth=depth, name=config.robot_name_prefix,
                )
                self._brain_args = (MLP.name(), dict(width=width, depth=depth))

        self.evaluate = functools.partial(
            self._evaluate, brain_factory=self._brain_factory,
            config=config, xml=self._state.to_string())

        self.cma_evaluate = functools.partial(
            self.evaluate, return_float=True)

    @property
    def num_parameters(self): return self._params

    @staticmethod
    def _evaluate(weights: np.ndarray, xml: str, brain_factory: BrainFactory, config: Config, return_float: bool):
        state, model, data = MjState.from_string(xml).unpacked
        brain = brain_factory(weights, state)

        match config.reward:
            case Rewards.SPEED:
                fitness = XSpeedMonitor(f"{config.robot_name_prefix}1")
            case _:
                raise NotImplementedError("Only implement speed reward, sorry.")

        with MjcbCallbacks(state, [brain], {"fitness": fitness}, config):
            mj_step(model, data, nstep=int(config.duration / model.opt.timestep))

        if return_float:
            return -fitness.value
        else:
            return EvaluationMetrics(dict(xspeed=fitness.value)), -fitness.value

    def save_champion(self, champion: np.ndarray, metrics: EvaluationMetrics):
        path = self._config.data_folder.joinpath("champion.zip")
        RerunnableRobot(
            mj_spec=self._state.spec,
            brain=(*self._brain_args, champion),
            metrics=metrics,
            misc=dict(),
            config=self._config
        ).save(path)
        return path
