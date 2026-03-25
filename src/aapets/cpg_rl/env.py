import functools
from pathlib import Path
from typing import Callable, Optional, Any

import gymnasium as gym
import numpy as np
from mujoco import mj_step, mj_resetData, set_mjcb_control, set_mjcb_passive, MjModel, MjData, mj_forward, MjSpec
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv

from .types import Config, Architecture, Trainer, RewardToMonitor, Environment as EnvironmentType
from ..common import canonical_bodies
from ..common.canonical_bodies import CanonicalBodies
from ..common.controllers.abstract import Controller
from ..common.controllers.mlp_tensor import MLPTensorBrain as MLP, MLPTensorBrain
from ..common.controllers.neighborhood_cpg import NeighborhoodCPG as CPG, NeighborhoodCPG
from ..common.misc.debug import kgd_debug
from ..common.monitors import Monitor
from ..common.monitors.metrics_storage import EvaluationMetrics
from ..common.mujoco.callback import MjcbCallbacks
from ..common.mujoco.state import MjState
from ..common.robot_storage import RerunnableRobot
from ..common.world_builder import make_world, compile_world

BrainFactory = Callable[[np.ndarray, MjState], Controller]


class EvoEnvironment:
    def __init__(self, config: Config):
        self._config = config

        robot_name = config.robot_name_prefix
        if config.env is EnvironmentType.GYM:
            assert config.body is CanonicalBodies.SPIDER45
            xml_path = Path(__file__).parent.joinpath("ariel-ant.xml")
            specs: MjSpec = MjSpec.from_file(str(xml_path))
            self._state = MjState.from_spec(specs)
            self.hinges = 8

        else:
            self._body = canonical_bodies.get(config.body)
            self._world = make_world(self._body.spec, robot_name=robot_name, camera_centered=True)
            self._state, _, _ = compile_world(self._world)
            self.hinges = len(self._body.hinges)

        match config.arch:
            case Architecture.CPG:
                neighborhood = config.cpg_neighborhood
                self._params = NeighborhoodCPG.num_parameters_with_neighborhood(self._state, robot_name, neighborhood)
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
            config=config, xml=self._state.to_string(),
        )

        self.cma_evaluate = functools.partial(self.evaluate, final=False)

    @property
    def num_parameters(self): return self._params

    @staticmethod
    def reward_function(config: Config, stepwise: bool):
        kwargs = dict(
            robot_name=f"{config.robot_name_prefix}1",
            stepwise=stepwise,
        )
        return RewardToMonitor[config.env][config.reward](**kwargs)

    @staticmethod
    def _evaluate(weights: np.ndarray, xml: str, brain_factory: BrainFactory, config: Config, final: bool):
        state, model, data = MjState.from_string(xml).unpacked
        brain = brain_factory(weights, state)

        if not final:
            fitness = EvoEnvironment.reward_function(config, stepwise=False)
            monitors = {"fitness": fitness}
        else:
            monitors = {
                monitor.name(): monitor(robot_name=f"{config.robot_name_prefix}1", stepwise=False)
                for monitor in RewardToMonitor[config.env].values()
            }

        with MjcbCallbacks(state, [brain], monitors, config):
            mj_step(model, data, nstep=int(config.duration / model.opt.timestep))

        if not final:
            return -fitness.value
        else:
            return EvaluationMetrics({name: monitor.value for name, monitor in monitors.items()})

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
    def __init__(self, config: Config, *, name: str, monitors: dict[str, Monitor] = None):
        super().__init__(config)

        # Recompile to get truncated float values
        self._state = MjState.from_string(self._state.to_string())

        self._state.name = name

        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(self.hinges,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.hinges,))

        # print("[kgd-debug|GymEnv:__init__]")
        self._joints_pos, self._mapping, self._actuators, self._joints, self._ranges = (
            Controller.control_data(self._state, config.robot_name_prefix))

        self._reward_function = self.reward_function(config, stepwise=True)
        self._substeps = int((1 / config.control_frequency) / self._state.model.opt.timestep)

        self.callbacks_handler = MjcbCallbacks(
            state=self._state,
            controllers=[self._control],
            monitors=(monitors or dict()) | dict(fitness=self._reward_function),
            config=config)

        self._actions = None

    def observation(self):
        return MLPTensorBrain.observation(self._joints, self._ranges, self._state)

    def infos(self): return {
        self._reward_function.name(): self._reward_function.value
    }

    @property
    def done(self): return self._state.time >= self._config.duration

    @property
    def state(self): return self._state

    def _control(self, state: MjState = None):
        # print(f"[kgd-debug|GymEnv:_control] time={state.time}")
        # print(f"[kgd-debug|GymEnv:_control] qpos={state.data.qpos}")
        # kgd_debug(f"action={self._actions}", timestamp=True)
        assert self._actions is not None
        for i, (actuator, action) in enumerate(zip(self._actuators, np.clip(self._actions, -1, 1))):
            actuator.ctrl[:] = action * self._ranges[i]

    def reset(self,  seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        # print(f"\n\n\n\n[kgd-debug|GymEnv:reset] ====================")

        self.callbacks_handler.stop()

        # if self.state.time > 0:
        #     exit(42)

        super().reset(seed=seed, options=options)

        # kgd_debug(f"t={self._state.time}")
        mj_resetData(self._state.model, self._state.data)
        mj_forward(self._state.model, self._state.data)
        # kgd_debug(f"t={self._state.time}", timestamp=True)

        self.callbacks_handler.start()

        return self.observation(), self.infos()

    def step(self, actions: np.ndarray):
        # kgd_debug(f"t={self._state.time}")
        # kgd_debug(f"qpos(n)={self._state.data.qpos}")
        # kgd_debug(f"state={self.observation()}")
        # kgd_debug(f"{actions=}")
        # with np.printoptions(precision=50):
        #     kgd_debug(f"actions={actions} ctrl={self._state.data.ctrl}")
        self._actions = actions#.copy()
        mj_step(self._state.model, self._state.data, self._substeps)
        # with np.printoptions(precision=50):
        #     kgd_debug(f"actions={actions} ctrl={self._state.data.ctrl}\n")
        # kgd_debug(f"qpos(n+1)={self._state.data.qpos}")
        assert self._reward_function.delta is not None
        # kgd_debug(f"t={self.state.time} {self._reward_function.delta=}")
        # kgd_debug(f"t={self.state.time} {self.done=}")
        return self.observation(), self._reward_function.delta, self.done, False, self.infos()

    def close(self):
        self.callbacks_handler.stop()

    @staticmethod
    def make_gym_vec_env(n, *, config: Config, name: str, vec_env_cls=SubprocVecEnv):
        def maker(_config: Config, _name=name): return GymEnvironment(_config, name=_name)

        return make_vec_env(
            env_id=maker,
            n_envs=n,
            env_kwargs=dict(_config=config, _name=name),
            vec_env_cls=vec_env_cls,
        )

    def save_champion(self, champion: ActorCriticPolicy, rewards: EvaluationMetrics):
        parameters = []

        for network in [champion.mlp_extractor.policy_net, champion.action_net]:
            for p in network.parameters():
                parameters.append(p.detach().numpy().flatten())

        print(f"Saving champion:\n{champion}")

        parameters = np.concatenate(parameters)
        # print(f"MLP Saved parameters:\n{parameters}")

        path = self._config.data_folder.joinpath("champion.zip")
        RerunnableRobot(
            mj_spec=self._state.spec,
            brain=(*self._brain_args, parameters),
            metrics=rewards,
            misc=dict(),
            config=self._config
        ).save(path)
        return path
