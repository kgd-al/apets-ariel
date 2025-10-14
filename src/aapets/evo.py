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

from config import CommonConfig, EvoConfig, SimuConfig
from evaluator import Evaluator
from map_elite import QDIndividual, Grid, Algorithm, Logger


@dataclass
class Config(CommonConfig):
    id: Annotated[Optional[int], "Name of the run (default to time)"] = None
    output_folder: Annotated[Path, "Where to store the data"] = Path("./tmp/qdpy/toy-revolve")
    snapshots: Annotated[int, "Number of checkpoints to make"] = 10
    overwrite: Annotated[bool, "Whether to clear the output folder"] = False

    # specs = None

    verbosity: int = 1

    seed: Optional[int] = None
    batch_size: int = 10
    budget: int = 100
    tournament: int = 5
    threads: int = 1


def eval_mujoco(ind: QDIndividual):
    assert isinstance(ind, QDIndividual)
    assert ind.id() is not None, "ID-less individual"
    r: Evaluator.Result = Evaluator.evaluate(ind.genome)
    ind.update(r)
    return ind


def main(config: Config):
    start = time.perf_counter()
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

    scenario = get_scenario(config.simu.experiment)

    # =========================================================================

    grid = Grid(shape=(16, 16),
                max_items_per_bin=1,
                fitness_domain=scenario.fitness_bounds(),
                features_domain=scenario.descriptor_bounds())

    logging.info(f"Grid size: {grid.shape}")
    logging.info(f"   bounds: {grid.features_domain}")
    logging.info(f"     bins: "
                 f"{[(d[1]-d[0]) / s for d, s in zip(grid.features_domain, grid.shape)]}")

    algo = Algorithm(grid, config, labels=[scenario.fitness_name(), *scenario.descriptor_names()])
    run_folder = Path(config.output_folder)

    # if args.specs is not None:
    #     Config.env_specifications = tuple(args.specs.split(","))

    config_path = run_folder.joinpath("config.json")
    Config.write_json(config_path)
    logging.info(f"Stored configuration in {config_path.absolute()}")

    # Create a logger to pretty-print everything and generate output data files
    save_every = round(config.budget / (config.batch_size * config.snapshots))
    logger = Logger(algo,
                    save_period=save_every,
                    log_base_path=config.output_folder)
    config._logger = logger

    with ParallelismManager(max_workers=config.threads) as mgr:
        mgr.executor._mp_context = multiprocessing.get_context("fork")  # TODO: Very brittle
        mgr.executer._initializer = lambda: print("Initializing...")
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
    parsed_config = parser.parse_args(namespace=Config())

    parser.print_help()

    main(parsed_config)
