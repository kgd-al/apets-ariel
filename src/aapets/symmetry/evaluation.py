import itertools

import numpy as np
from mujoco import mj_step

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from common.monitors import XSpeedMonitor
from common.mujoco.callback import MjcbCallbacks
from .config import Config
from .genotypes import Genome
from ..common.world_builder import make_world, compile_world
from ..fetch.controllers.ABCpg import ABCpg


def save_robot(genome: Genome, config: Config):
    


def forward_locomotion(genome: Genome, config: Config):
    robot = construct_mjspec_from_graph(genome.body.to_networkx())
    world = make_world(robot.spec, robot_name=config.robot_name_prefix)
    state, model, data = compile_world(world)

    hinges = len(robot.hinges)

    brain = ABCpg.from_cppn(genome.brain, state, name=config.robot_name_prefix)

    robot_name = f"{config.robot_name_prefix}1"
    fitness = XSpeedMonitor(robot_name, stepwise=False)
    monitors = {
        m.name(): m for m in [fitness]
    }

    with MjcbCallbacks(state, [brain], monitors, config):
        mj_step(model, data, nstep=int(config.duration / model.opt.timestep))

    return fitness.value, 0
