#!/usr/bin/env python3

import argparse
import logging
import pprint
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Optional

import humanize
from mujoco import mj_step, mj_forward

from ..common import canonical_bodies, controllers, morphological_measures
from ..common.config import BaseConfig, ViewerConfig, AnalysisConfig, ViewerModes
from ..common.controllers import RevolveCPG
from ..common.misc.config_base import Unset
from ..common.monitors import BrainActivityPlotter, TrajectoryPlotter, metrics
from ..common.monitors.metrics_storage import EvaluationMetrics
from ..common.monitors.plotters.record import MovieRecorder
from ..common.mujoco.callback import MjcbCallbacks
from ..common.mujoco.state import MjState
from ..common.mujoco.viewer import passive_viewer, interactive_viewer
from ..common.robot_storage import RerunnableRobot
from ..common.world_builder import make_world, compile_world

if __name__ == "__main__":
    # Access configuration in standalone mode
    from ..zoo.evolve import Arguments as ZooArguments


@dataclass
class Arguments(BaseConfig, ViewerConfig, AnalysisConfig):
    robot_archive: Annotated[Optional[Path], "Path to the rerunnable-robot archive"] = Unset

    run: Annotated[bool, "Whether to enable/disable evaluation (e.g. for genotype data and/or checks)"] = True
    check_performance: Annotated[bool, "If a genome is given, test for determinism"] = True

    default_body: Annotated[str, "Name of a canonical body to use when generating the defaults",
                            dict(choices=canonical_bodies.get_all())] = "spider45"


def generate_defaults(args: Arguments):
    if args.seed is None:
        args.seed = 0

    robot = canonical_bodies.get(args.default_body)
    world = make_world(robot.spec)

    state, _, _ = compile_world(world)
    brain = RevolveCPG.random(state, args.seed)

    rr = RerunnableRobot(
        mj_spec=world.spec,
        brain=("RevolveCPG", brain.extract_weights()),
        metrics=EvaluationMetrics(dict()),
        # metrics=EvaluationMetrics(dict(
        #     single_nested=dict(
        #         xyspeed=0.0107572,
        #         xspeed=0,
        #     ),
        #     double_nested=dict(
        #         internal_dict=dict(
        #             xyspeed=0,
        #             weight=1.92
        #         ),
        #         xspeed=0,
        #     ),
        #     xspeed=0,
        # )),
        config=BaseConfig(**{
            k: v for k, v in vars(args).items() if hasattr(BaseConfig, k)
        }),
        misc=dict()
    )

    def_folder = Path(f"tmp/defaults/")
    def_folder.mkdir(parents=True, exist_ok=True)

    args.robot_archive = def_folder.joinpath(f"{args.seed}.zip")
    rr.save(args.robot_archive)

    if args.verbosity > 0:
        print("Generated default file", args.robot_archive)


def main(args: Arguments) -> int:
    start = time.perf_counter()

    # ==========================================================================
    # Parse command-line arguments

    if args.verbosity <= 0:
        logging.basicConfig(level=logging.WARNING)
    elif args.verbosity <= 2:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG)

    if args.robot_archive is Unset:
        raise ValueError("No robot archive provided. Please specify a valid file or None to use defaults")

    elif str(args.robot_archive).upper() == "NONE":
        args.robot_archive = None

    elif not args.robot_archive.exists():
        raise ValueError(f"No such file: {args.robot_archive} does not exist")

    for m in ['matplotlib',
              'OpenGL.arrays.arraydatatype', 'OpenGL.acceleratesupport']:
        logger = logging.getLogger(m)
        logger.setLevel(logging.WARNING)

    if args.verbosity >= 2:
        print("Command line-arguments:")
        pprint.PrettyPrinter(indent=2, width=1).pprint(args.__dict__)

    defaults = (args.robot_archive is None)
    if defaults:
        generate_defaults(args)

    if args.verbosity > 1:
        print("Deduced options:", end='\n\t')
        pprint.pprint(args)

    # ==========================================================================
    # Prepare and launch

    record = RerunnableRobot.load(args.robot_archive)
    args.override_with(record.config, verbose=True)

    output_prefix = args.robot_archive.with_suffix("")
    plot_ext = args.plot_format

    genotype = record.misc.get("genotype")
    if genotype is not None:
        if args.render_brain_genotype and (render := getattr(genotype, "render_genotype")):
            render(output_prefix.with_suffix(".genome."))

    elif args.render_brain_genotype:
        logger.warning("Genotype plotting requested but no genotype was found in the archive.")

    if args.morphological_measures:
        print("Morphological measures:", pprint.pformat(
            morphological_measures.measure(record.mj_spec, args.robot_name_prefix).all_metrics))

    if not args.run:
        return 0

    if args.camera is not None:  # Adjust camera *before* compilation
        adjust_camera(record.mj_spec, args, args.robot_name_prefix)

    state = MjState.from_spec(record.mj_spec)
    model, data = state.model, state.data
    mj_forward(model, data)

    brain = controllers.get(record.brain[0])(record.brain[1], state)
    if args.render_brain_phenotype:
        brain.render_phenotype(output_prefix.with_suffix(f".{brain.name}.{plot_ext}"))

    monitors_kwargs = dict(
        name=f"{record.config.robot_name_prefix}1"
    )
    monitors = {
        name: metrics(name, **monitors_kwargs) for name in record.metrics.keys()
    }

    robot_name = f"{args.robot_name_prefix}1"

    if args.plot_brain_activity:
        monitors["brain_activity"] = BrainActivityPlotter(
            args.sample_frequency, robot_name,
            output_prefix.with_suffix(f".brain_activity.{plot_ext}")
        )

    if args.plot_trajectory:
        monitors["trajectory"] = TrajectoryPlotter(
            args.sample_frequency, robot_name,
            output_prefix.with_suffix(f".trajectory.{plot_ext}")
        )

    if args.movie:
        monitors["movie_recorder"] = MovieRecorder(
            args.movie_framerate, args.movie_width, args.movie_height,
            output_prefix.with_suffix(".mp4"),
            camera=args.camera, shadows=True
        )

    with MjcbCallbacks(state, [brain], monitors, args) as callback:
        match args.viewer:
            case ViewerModes.NONE:
                mj_step(model, data, nstep=int(args.duration / model.opt.timestep))

            case ViewerModes.INTERACTIVE:
                interactive_viewer(model, data, args)

            case ViewerModes.PASSIVE:
                passive_viewer(state, args)

    result = EvaluationMetrics.from_template(callback.metrics, record.metrics)

    # ==========================================================================
    # Process results

    if args.verbosity > 0:
        result.pretty_print()

    err = 0

    if args.check_performance and not defaults:
        if abs(data.time - args.duration) < 1e-3:
            err = EvaluationMetrics.compare(
                record.metrics, result, args.verbosity
            )
        elif args.verbosity > 0:
            print("Re-evaluation had different duration, not comparing performance.")

    if args.verbosity > 1:
        duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start))
        print(f"Evaluated {args.robot_archive.absolute().resolve()} in {duration} / {args.duration}s")

    return err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerun evolved champions")
    Arguments.populate_argparser(parser)

    exit(main(parser.parse_args(namespace=Arguments())))
