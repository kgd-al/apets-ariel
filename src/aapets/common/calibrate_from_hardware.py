from aapets.common.misc.config_base import IntrospectiveAbstractConfig
from dataclasses import dataclass
from typing import Annotated
from pathlib import Path
import pprint
import pickle
import matplotlib.pyplot as plt

from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule


@dataclass
class Config(IntrospectiveAbstractConfig):
    datafile: Annotated[Path, "Location of ground-truth datafile", dict(required=True)] = None
    output: Annotated[Path, "Where to store the output"] = "./"


def main():
    args = Config.parse_command_line_arguments(
        description="Small utility that compares ground-truth obtained from a physical robot"
                    " to simulated dynamics to try and match hinges actuation")

    with open(args.datafile, "rb") as f:
        data = pickle.load(f)
        n = len(data)

        plt.rcParams['text.usetex'] = True
        fig, axes = plt.subplots(n, 2, figsize=(16, 2*n), sharex=True, sharey=True)
        for (f, d), ax in zip(data.items(), axes):
            core = CoreModule()
            hinge = HingeModule()

            for i, (_n, _d) in enumerate(d.items()):
                ax[i].plot(_d)
                ax[i].set_title(f"${2*f}\\pi x$: {_n}")

        fig.tight_layout()
        fig.savefig(args.output.joinpath("calibration.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
