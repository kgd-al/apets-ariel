from dataclasses import dataclass
from enum import StrEnum, auto
from pathlib import Path
from typing import Annotated

from mujoco import mj_step

from aapets.common.canonical_bodies import CanonicalBodies
from aapets.common.config import BaseConfig
from aapets.common.monitors.plotters.brain_activity import BrainActivityPlotter
from aapets.common.mujoco.callback import MjcbCallbacks
from aapets.common.world_builder import make_world, compile_world
from common.controllers.ABCpg import ABCpg


class ShowcaseType(StrEnum):
    DYNAMICS = auto()


@dataclass
class Config(BaseConfig):
    type: Annotated[ShowcaseType, "What to showcase"] = ShowcaseType.DYNAMICS

    out: Annotated[Path, "Where to store generated stuff"] = Path("tmp/cpg-showcase")

    @classmethod
    def yaml_tag(cls): return "CLI"


def showcase_cpg_dynamics(args: Config):
    robot_name = "test"

    for i in range(100):
        robot = CanonicalBodies.SPIDER45.get()
        world = make_world(robot.spec, robot_name=robot_name)
        state, model, data = compile_world(world)

        cpg = ABCpg.random(state=state, name=robot_name, seed=args.seed+i)
        cpg.set(alpha=0, beta=1)

        monitors = dict(
            plotter=BrainActivityPlotter(20, robot_name, args.out.joinpath(f"basics_{i:03}.png")),
        )

        args.duration = 10
        with MjcbCallbacks(state, [cpg], monitors, args):
            mj_step(model, data, nstep=int(args.duration / model.opt.timestep))


if __name__ == "__main__":
    _args = Config.parse_command_line_arguments("Simple showcaser for the a/b modulated cpg")
    _args.out.mkdir(parents=True, exist_ok=True)
    match _args.type:
        case ShowcaseType.DYNAMICS:
            showcase_cpg_dynamics(_args)
