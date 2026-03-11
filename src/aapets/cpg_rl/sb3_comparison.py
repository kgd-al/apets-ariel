from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import register
from gymnasium.envs.mujoco.ant_v5 import AntEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from aapets.common.controllers import MLPTensorBrain
from aapets.common.misc.config_base import IntrospectiveAbstractConfig


@dataclass
class Config(IntrospectiveAbstractConfig):
    policy: Annotated[Optional[Path], "Path to an existing policy (to load from)"] = None

    threads: Annotated[int, "Number of threads to use for training"] = 1
    restrict_inputs: Annotated[bool, "Whether to restrict inputs to joint position"] = False
    duration: Annotated[float, "Episode duration in seconds"] = 15
    seed: Annotated[int, "Random seed", dict(required=True)] = None
    folder: Annotated[Path, "Where to store data", dict(required=True)] = None
    budget: Annotated[int, "How many timesteps to train for"] = 2_000_000


class LowInputsAnt(AntEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64
        )

        self.joints = [
            j for i in range(self.model.njnt)
            if len((j := self.data.joint(i)).qpos) == 1
        ]

    def _get_obs(self):
        return np.array([j.qpos[0] for j in self.joints])


register(id="Ant-v5-8inputs", entry_point=LowInputsAnt)


def main():
    args = Config.parse_command_line_arguments("Just a sanity check for gym-ant performance")
    args.pretty_print()
    env_name = "Ant-v5-8inputs" if args.restrict_inputs else "Ant-v5"

    if args.policy is None:
        vec_env = make_vec_env(env_name, n_envs=args.threads)
        model = PPO("MlpPolicy", vec_env, verbose=1, seed=args.seed, device="cpu")
        model.set_logger(configure(str(args.folder), ["csv", "tensorboard"]))
        model.learn(total_timesteps=args.budget, progress_bar=True)
        model.save(str(args.folder.joinpath("model")))

        n_params = MLPTensorBrain.num_parameters_from_module(model.policy)

        env = gym.make(env_name)
        obs, _ = env.reset()
        data = env.unwrapped.data
        while data.time < args.duration:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, truncated, terminated, info = env.step(action)

        distance = np.sqrt(np.sum(data.qpos[0:2] ** 2))
        speed = distance / args.duration
        print(f"Velocity: {speed} ({data.qpos[0:2]}, {distance}, {args.duration})")

        df = pd.DataFrame.from_records([dict(
            run=args.seed, speed=speed, inputs=len(obs), params=n_params,
            distance=distance, dx=data.qpos[0], dy=data.qpos[1],
        )])
        df.index = [args.folder]
        df.to_csv(args.folder.joinpath("summary.csv"))

    else:
        model = PPO.load(args.policy)
        env = gym.make(env_name, render_mode="human")
        obs, _ = env.reset()
        data = env.unwrapped.data
        while data.time < args.duration:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, truncated, terminated, info = env.step(action)
            env.render()
        env.close()


if __name__ == "__main__":
    main()
