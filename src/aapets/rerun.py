#!/usr/bin/env python3

import argparse
import json
import logging
import math
import numbers
import pprint
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional, Annotated

import humanize
from colorama import Fore, Style

from aapets.evaluation_result import EvaluationResult
from aapets.evaluator import Evaluator
from config import CommonConfig
from genotype import Genotype
from map_elite import QDIndividual


@dataclass
class Options(CommonConfig):
    robot: Annotated[Optional[Path], "Path to a robot genotype"] = None
    config: Annotated[Optional[Path],
                      ("Path to a specific configuration file,"
                       " can be derived from robot path (if any)")] = None

    no_run: Annotated[bool, "Whether to disable evaluation (for genotype data and/or checks"] = False
    check_performance: Annotated[bool, "If a genome is given, test for determinism"] = True


def generate_defaults(args):
    config = Options()
    config.descriptors = ["speed", "weight"]

    data = Genotype.Data(config=config, seed=0)
    genome = Genotype.random(data)
    ind = QDIndividual(genome)

    def_folder = Path("tmp/defaults/")
    def_folder.mkdir(parents=True, exist_ok=True)

    ind_file = args.robot = def_folder.joinpath("genome.json")
    ind.save_to(ind_file)

    cnf_file = args.config = def_folder.joinpath("config.json")
    config.write_yaml(cnf_file)

    if args.verbosity > 0:
        print("Generated default files", [ind_file, cnf_file])


def try_locate(base: Path, name: str, levels: int = 0, strict: bool = True):
    if not base.exists():
        raise FileNotFoundError(f"Genome not found at {base}")

    path = base
    attempts = 0
    while attempts <= levels:
        path = path.parent
        candidate = path.joinpath(name)
        if candidate.exists():
            return candidate
        attempts += 1

    if strict:
        raise FileNotFoundError(f"Could not find file for '{name}' "
                                f"at most {levels} level(s) from '{base}'")
    return None


def main() -> int:
    start = time.perf_counter()
    # ==========================================================================
    # Parse command-line arguments

    parser = argparse.ArgumentParser(description="Rerun evolved champions")
    Options.populate_argparser(parser)
    options = parser.parse_args(namespace=Options())

    if options.verbosity <= 0:
        logging.basicConfig(level=logging.WARNING)
    elif options.verbosity <= 2:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG)

    for m in ['matplotlib',
              'OpenGL.arrays.arraydatatype', 'OpenGL.acceleratesupport']:
        logger = logging.getLogger(m)
        logger.setLevel(logging.WARNING)

    if options.verbosity >= 2:
        print("Command line-arguments:")
        pprint.PrettyPrinter(indent=2, width=1).pprint(options.__dict__)

    defaults = (options.robot is None)
    if defaults:
        generate_defaults(options)

    if options.config is None:
        options.config = try_locate(options.robot, "config.yaml", 2)
    options = Options.read_yaml(options.config).override_with(options)

    # if options.movie:
    #     save_folder = True
    #     options.runner.record = RunnerOptions.Record(
    #         video_file_path=Path(options.robot.stem + ".movie.mp4"),
    #         width=options.width, height=options.height)
    #
    # elif options.viewer:
    #     options.runner.view = RunnerOptions.View(
    #         start_paused=(not options.record and not options.auto_start),
    #         speed=options.speed,
    #         auto_quit=options.auto_quit,
    #         cam_id=options.cam_id,
    #         settings_save=options.settings_save,
    #         settings_restore=options.settings_restore,
    #     )

    if options.verbosity > 1:
        print("Deduced options:", end='\n\t')
        pprint.pprint(options)

        print("Loaded configuration from", options.config.absolute().resolve())
        options.print()

    # ==========================================================================
    # Prepare and launch

    ind = QDIndividual.load_from(options.robot)
    genome = ind.genotype

    # if options.save_cppn:
    #     path = str(options.robot.parent.joinpath(options.robot.stem + ".cppn"))
    #     genome.brain.to_dot(path + ".png")
    #     genome.brain.to_dot(path + ".pdf")
    #     genome.brain.to_dot(path + ".dot")

    if options.no_run:
        return 0

    Evaluator.initialize(options)
    result = Evaluator.evaluate(genome)

    # ==========================================================================
    # Process results

    err = 0

    if options.check_performance and not defaults:
        err = EvaluationResult.performance_compare(
            ind.evaluation_result(), result, options.verbosity
        )

    if options.verbosity > 1:
        duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start))
        print(f"Evaluated {options.robot.absolute().resolve()} in {duration} / {options.duration}s")

    return err


if __name__ == "__main__":
    exit(main())
