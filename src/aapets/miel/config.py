from dataclasses import dataclass
from enum import auto, StrEnum
from pathlib import Path
from typing import Annotated, Optional

from aapets.common import BaseConfig, EvoConfig


class ExperimentType(StrEnum):
    # FETCH = "fetch"
    # # Rotate until target is in sight
    # # Move with CPG modulation until within range of the target (ball or human)
    # # If ball:
    # #   lock ball, switch to human tracking
    # # Else:
    # #   unlock ball, wait for ball no longer in sight / command

    # # Elementary actions
    # TARGET = "target"  # Move towards point
    # ROTATE = "rotate"  # Rotate until aligned with point

    # Step-by-step progression
    LOCOMOTION = auto()  # Go wherever
    DIRECTED_LOCOMOTION = auto()  # Go in one direction
    TARGETED_LOCOMOTION = auto()  # Go towards something, with sufficient information to do so
    TRACKING = auto()  # First look for object, then go to it


@dataclass
class MielConfig(BaseConfig, EvoConfig):
    experiment: Annotated[ExperimentType, "Experiment to perform"] = None

    # number of initial mutations for abrain's genome
    initial_mutations: Annotated[int, "Number of initial mutations for each random genome"] = 10

    body_brain_mutation_ratio: Annotated[float, "Probability of mutating the body, otherwise brain"] = 0.1
    body_mutation_rate: Annotated[float, "Probability of cumulative mutations (poisson law)"] = 0.9

    max_modules: Annotated[int, "Maximal number of modules"] = 20

    resume: Annotated[Path, "Resume evolution from provided checkpoint archive"] = None
    threads: Annotated[int, "Number of concurrent evaluations"] = None

    population_size: Annotated[int, "Population size (duh)"] = 10
    generations: Annotated[int, "Number of generations (double duh)"] = 10

    tournament_size: Annotated[int, "How many individuals to compare for fitness"] = 4
    elitism: Annotated[int, "How many champions to keep unchanged each generation"] = 1

    descriptors: Annotated[list[str], "Behavioral descriptors (NOT fitness)",
                           dict(nargs=2, type=str)] = None

    fixed_body: Annotated[Optional[str], ("Canonical morphology to use."
                                          " Switches evolution to pure numerical"
                                          " optimization")] = None

    body_genotype_size: Annotated[int, "Number of float per chromosome for NDE decoding"] = 64

    def __post_init__(self):
        pass
