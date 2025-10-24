#!/usr/bin/env python3

import argparse
import json
import logging
import multiprocessing
import pprint
import sys
import time
from dataclasses import fields, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional, Annotated

import humanize
from qdpy.base import ParallelismManager

from abrain.core.genome import logger as genome_logger

from aapets.evaluation_result import EvaluationResult
from aapets.genotype import Genotype
from aapets.misc.config_base import ConfigBase
from config import CommonConfig, EvoConfig, SimuConfig
from evaluator import Evaluator
from map_elite import QDIndividual, Grid, Algorithm, Logger


@dataclass
class Config(CommonConfig):
    snapshots: Annotated[int, "Number of checkpoints to make"] = 10

    batch_size: Annotated[int, "Number of concurrent evaluations ~= population size"] = 10

    # specs = None


def eval_mujoco(ind: QDIndividual):
    assert isinstance(ind, QDIndividual)
    r: EvaluationResult = Evaluator.evaluate(ind.genotype)
    ind.update(r)
    return ind


def main(config: Config):
    start = time.perf_counter()
    # =========================================================================

    if config.experiment is None:
        raise RuntimeError("Experiment must be specified (see --experiment)")

    if config.output_folder is None:
        raise RuntimeError("Output folder must be specified (see --output-folder)")

    if config.descriptors is None:
        raise RuntimeError("Descriptors must be specified (see --descriptors)")

    # =========================================================================

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s|%(process)s|%(levelname)s|%(module)s] %(message)s",
        stream=sys.stdout
    )
    genome_logger.setLevel(logging.INFO)

    for m in ['matplotlib', 'OpenGL.arrays.arraydatatype', 'OpenGL.acceleratesupport']:
        logger = logging.getLogger(m)
        logger.setLevel(logging.WARNING)
        logging.info(f"Muting {logger}")

    logging.captureWarnings(True)

    # =========================================================================

    scenario_data = Evaluator.initialize(config, verbose=False)

    # =========================================================================

    grid = Grid(shape=(16, 16),
                max_items_per_bin=1,
                fitness_domain=[scenario_data.fitness_bounds],
                features_domain=scenario_data.descriptor_bounds)

    logging.info(f"Grid size: {grid.shape}")
    logging.info(f"   bounds: {grid.features_domain}")
    logging.info(f"     bins: "
                 f"{[(d[1]-d[0]) / s for d, s in zip(grid.features_domain, grid.shape)]}")

    algo = Algorithm(
        grid, Genotype,
        config, labels=[scenario_data.fitness_name, *scenario_data.descriptor_names])
    run_folder = Path(config.output_folder)

    # if args.specs is not None:
    #     Config.env_specifications = tuple(args.specs.split(","))

    config_path = run_folder.joinpath("config.yaml")
    config.write_yaml(config_path)
    logging.info(f"Stored configuration in {config_path.absolute()}")

    # Create a logger to pretty-print everything and generate output data files
    save_every = round(algo.budget / (config.batch_size * config.snapshots))
    logger = Logger(algo,
                    save_period=save_every,
                    log_base_path=config.output_folder)
    config._logger = logger

    with ParallelismManager(max_workers=config.threads) as mgr:
        mgr.executor._mp_context = multiprocessing.get_context("fork")  # TODO: Very brittle
        mgr.executor._initializer = lambda: print("Initializing...")
        logging.info("Starting illumination!")
        best = algo.optimise(evaluate=eval_mujoco, executor=mgr.executor, batch_mode=True)

    if best is not None:
        with open(run_folder.joinpath("best.json"), 'w') as f:
            data = Algorithm.to_json(best)
            logging.info(f"best:\n{pprint.pformat(data)}")
            json.dump(data, f)
    else:
        logging.warning("No best individual found")

    # Print results info
    logging.info(algo.summary())

    # Plot the results
    logger.summary_plots()

    logging.info(f"All results are available under {logger.log_base_path}")
    logging.info(f"Unified storage file is {logger.log_base_path}/{logger.final_filename}")

    duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start))
    logging.info(f"Completed evolution in {duration}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Main evolution script")
    Config.populate_argparser(parser)
    # parser.print_help()
    parsed_config = parser.parse_args(namespace=Config())

    main(parsed_config)
