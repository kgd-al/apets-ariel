from ..common import canonical_bodies
from ..common.config import BaseConfig, EvoConfig


from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional


@dataclass
class Arguments(BaseConfig, EvoConfig):
    body: Annotated[str, "Morphology to use",
                    dict(choices=canonical_bodies.get_all())] = None

    budget: Annotated[int, "Number of CMA-ES evaluations to perform"] = 10
    threads: Annotated[Optional[int], ("Number of threads to use. A positive number requests that number of core, zero"
                                       "disables parallelism and -1 requests everything")] = 1

    initial_std: Annotated[float, "Initial standard deviation for CMA-ES"] = .5

    symlink_last: Annotated[bool, "Make a symbolic link to the last run"] = True

    rerun: Annotated[Optional[Path], "Path to the archive to use for re-evaluation"] = None
