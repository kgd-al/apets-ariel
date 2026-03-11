from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Annotated

from ..common.canonical_bodies import CanonicalBodies
from ..common.config import EvoConfig, BaseConfig
from ..common.monitors import XSpeedMonitor
from ..common.monitors.behavioral import KernelRewardMonitor, GymRewardMonitor, GymAntGymRewardMonitor, \
    GymAntKernelRewardMonitor


class Architecture(StrEnum):
    CPG = auto()
    MLP = auto()


class Trainer(StrEnum):
    CMA = auto()
    PPO = auto()


class Rewards(StrEnum):
    SPEED = auto()
    GYM = auto()
    KERNELS = auto()


class Environment(StrEnum):
    ARIEL = auto()
    GYM = auto()


RewardToMonitor = {
    Environment.ARIEL: {
        Rewards.SPEED: XSpeedMonitor,
        Rewards.GYM: GymRewardMonitor,
        Rewards.KERNELS: KernelRewardMonitor,
    },
    Environment.GYM: {
        Rewards.SPEED: XSpeedMonitor,
        Rewards.GYM: GymAntGymRewardMonitor,
        Rewards.KERNELS: GymAntKernelRewardMonitor,
    },
}


@dataclass
class Config(BaseConfig, EvoConfig):
    arch: Annotated[Architecture, "Architecture to use", dict(choices=Architecture, required=True)] = None
    trainer: Annotated[Trainer, "Trainer to use", dict(choices=Trainer, required=True)] = None
    reward: Annotated[Rewards, "Reward type", dict(choices=Rewards, required=True)] = None

    body: Annotated[CanonicalBodies, "Morphology to use"] = CanonicalBodies.SPIDER45
    env: Annotated[Environment, "Whether to use a gym version of the requested body"] = Environment.ARIEL

    budget: Annotated[int, "Total budget in number of seconds"] = 100
    duration: Annotated[int, "Duration of a single evaluation/episode"] = 10

    threads: Annotated[int, "Total number of (requested) threads"] = 0

    cpg_neighborhood: Annotated[int, "Maximal module distance between connected cpgs in the network"] = None

    mlp_width: Annotated[int, "Width of each hidden layer in the MLP"] = None
    mlp_depth: Annotated[int, "Number of hidden layers in the MLP"] = None
