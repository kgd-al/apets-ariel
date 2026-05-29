from dataclasses import dataclass
from typing import Annotated

from ..common.config import EvoConfig, BaseConfig


@dataclass
class Config(BaseConfig, EvoConfig):
    population_size: Annotated[int, "Population size (duh)"] = 8  # Must be a multiple of 4
    generations: Annotated[int, "Number of generations (double duh)"] = 10
    threads: Annotated[int, "Number of threads to use (defaults to os.cpu_count()-1)"] = None

    novelty_knn: Annotated[int, "Number of queried neighbours when testing novelty of an individual"] = 5
    novelty_add_threshold: Annotated[float, "Minimum required novelty to be added to the archive"] = .1

    initial_mutations_body: Annotated[int, "Number of times a random body is mutated"] = 10
    initial_mutations_brain: Annotated[int, "Number of times a random brain is mutated"] = 10

    max_modules: Annotated[int, "Maximal number of modules"] = 10
