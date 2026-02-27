import functools
import itertools
from typing import Callable, Optional, Any

import gymnasium as gym
import numpy as np

from mujoco import mj_step, mj_resetData
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv

from .types import Config, Architecture, Rewards, Trainer
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
from stable_baselines3.common.env_util import make_vec_env


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
                    grad=(config.trainer == Trainer.PPO)
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
    def reward_function(config: Config, stepwise: bool):
        kwargs = dict(
            robot_name=f"{config.robot_name_prefix}1",
            stepwise=stepwise,
        )
        match config.reward:
            case Rewards.SPEED:
                return XSpeedMonitor(**kwargs)
            case _:
                raise NotImplementedError("Only implement speed reward, sorry.")

    @staticmethod
    def _evaluate(weights: np.ndarray, xml: str, brain_factory: BrainFactory, config: Config, return_float: bool):
        state, model, data = MjState.from_string(xml).unpacked
        brain = brain_factory(weights, state)

        fitness = EvoEnvironment.reward_function(config, stepwise=False)

        with MjcbCallbacks(state, [brain], {"fitness": fitness}, config):
            mj_step(model, data, nstep=int(config.duration / model.opt.timestep))

        if return_float:
            return -fitness.value
        else:
            return EvaluationMetrics({fitness.name(): fitness.value}), -fitness.value

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


class GymEnvironment(EvoEnvironment, gym.Env):
    def __init__(self, config: Config):
        super().__init__(config)
        hinges = len(self._body.hinges)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(hinges,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(hinges,))

        print("[kgd-debug|GymEnv:__init__]")
        self._joints_pos, self._mapping, self._actuators, self._ranges = (
            Controller.control_data(self._state, config.robot_name_prefix))

        self._reward_function = self.reward_function(config, stepwise=True)
        self._substeps = int((1 / config.control_frequency) / self._state.model.opt.timestep)

    def observation(self):
        return np.array([a.length[0] for a in self._actuators], dtype=np.float32)

    def infos(self): return {
        self._reward_function.name(): self._reward_function.value
    }

    @property
    def done(self): return self._state.time >= self._config.duration

    def reset(self,  seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        self._reward_function.stop(self._state)
        super().reset(seed=seed, options=options)
        print("[kgd-debug|GymEnv:reset] ", f"t={self._state.time}")
        mj_resetData(self._state.model, self._state.data)
        print("[kgd-debug|GymEnv:reset] >>", f"t={self._state.time}")
        self._reward_function.start(self._state)
        return self.observation(), self.infos()

    def step(self, actions: np.ndarray):
        for i, (actuator, action) in enumerate(zip(self._actuators, np.clip(actions, -1, 1))):
            actuator.ctrl[:] = action * self._ranges[i]
        print("[kgd-debug|GymEnv:step]", f"t={self._state.time}")
        print("[kgd-debug|GymEnv:step]", f"state={self.observation()}")
        print("[kgd-debug|GymEnv:step]", f"{actions=}")
        print(f"[kgd-debug|GymEnv:__call__] ctrl={self._state.data.ctrl}")
        mj_step(self._state.model, self._state.data, self._substeps)
        self._reward_function(self._state)
        assert self._reward_function.delta is not None
        return self.observation(), self._reward_function.delta, self.done, False, self.infos()

    @staticmethod
    def make_gym_vec_env(n, *, config: Config, vec_env_cls=SubprocVecEnv):
        def maker(_config: Config): return GymEnvironment(_config)

        return make_vec_env(
            env_id=maker,
            n_envs=n,
            env_kwargs=dict(_config=config),
            vec_env_cls=vec_env_cls,
        )

    def save_champion(self, champion: ActorCriticPolicy, reward: float):
        parameters = []

        for network in [champion.mlp_extractor.policy_net, champion.action_net]:
            for p in network.parameters():
                parameters.append(p.detach().numpy().flatten())

        print(f"Saving champion:\n{champion}")

        parameters = np.concatenate(parameters)
        print(f"MLP Saved parameters:\n{parameters}")

        path = self._config.data_folder.joinpath("champion.zip")
        RerunnableRobot(
            mj_spec=self._state.spec,
            brain=(*self._brain_args, parameters),
            metrics=EvaluationMetrics({self._reward_function.name(): reward}),
            misc=dict(),
            config=self._config
        ).save(path)
        return path
