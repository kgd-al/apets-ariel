from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from ..common.config import EvoConfig, BaseConfig


@dataclass
class Config(BaseConfig, EvoConfig):
    data_folder: Annotated[Path, "Data storage for current experiment",
                           dict(required=True)] = None

    population_size: Annotated[int, "Population size (duh)"] = 8  # Must be a multiple of 4
    generations: Annotated[int, "Number of generations (double duh)"] = 10
    learning: Annotated[int, "Number of learning steps"] = 10
    threads: Annotated[int, "Number of threads to use (defaults to os.cpu_count()-1)"] = None

    novelty_knn: Annotated[int, "Number of queried neighbours when testing novelty of an individual"] = 15
    novelty_add_threshold: Annotated[float, "Minimum required novelty to be added to the archive"] = .25
    novelty_archive_decimals: Annotated[float, "Number of descriptor decimals when stored in the archive"] = 6
    novelty_initial_intake: Annotated[float, "Proportion of the first population used to fill the archive"] = .05
    novelty_intake: Annotated[float, "Proportion of any non-initial populations added to the archive"] = .02

    initial_mutations_body: Annotated[int, "Number of times a random body is mutated"] = 10
    initial_mutations_brain: Annotated[int, "Number of times a random brain is mutated"] = 10

    probability_mutation: Annotated[float, "Probability of an offspring being mutated after crossover"] = .8
    probability_crossover: Annotated[float, "Probability of individuals being crossed"] = .8

    max_modules: Annotated[int, "Maximal number of modules"] = 10

    plot_only: Annotated[bool, ("Disables evolution and loads from provided directory to"
                                " (re)generate plots based on stored data")] = False
