from dataclasses import dataclass
from typing import Annotated

from ..common.config import EvoConfig, BaseConfig


@dataclass
class Config(BaseConfig, EvoConfig):
    @classmethod
    def yaml_tag(cls): return "MainConfig"

    population_size: Annotated[int, "Population size (duh)"] = 8  # Must be a multiple of 4
    generations: Annotated[int, "Number of generations (double duh)"] = 10

    initial_mutations_body: Annotated[int, "Number of times a random body is mutated"] = 10
    initial_mutations_brain: Annotated[int, "Number of times a random brain is mutated"] = 10

    max_modules: Annotated[int, "Maximal number of modules"] = 10
