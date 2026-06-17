import shutil
import time
from datetime import timedelta
from pathlib import Path

import humanize

from aapets.bin.rerun import Arguments as RerunArguments, main as _rerun
from aapets.common.config import ViewerModes
from aapets.g_cpg.config import Config
from aapets.g_cpg.deap_impl import DEAPWrap


def rerun(args: Config, champion: Path):
    rerun_args = RerunArguments.copy_from(args)
    rerun_args.robot_archive = champion

    rerun_args.movie = True
    rerun_args.viewer = ViewerModes.NONE

    rerun_args.movie = "mp4"
    rerun_args.camera = f"{args.robot_name_prefix}1_tracking-cam"
    rerun_args.camera_angle = 45
    rerun_args.camera_distance = 2
    rerun_args.camera_center = "com"

    rerun_args.plot_format = "png"
    rerun_args.plot_trajectory = True
    rerun_args.plot_brain_activity = True
    rerun_args.plot_rewards = True
    rerun_args.render_brain_genotype = False
    rerun_args.render_brain_phenotype = False
    rerun_args.record_position = True
    rerun_args.record_joints = True

    return _rerun(rerun_args)


def main(args: Config):
    start_time = time.perf_counter()
    err = 0

    if args.verbosity > 0:
        args.pretty_print()
        print()

    if args.plot_only:
        if not args.data_folder.exists():
            raise FileNotFoundError(f"Cannot plot data from {args.data_folder} as it does not exist")
        print("Only (re)generating plots. Not running an evolution.")
        DEAPWrap.plot(args.data_folder)
        exit(0)

    elif args.data_folder.exists():
        if args.overwrite:
            shutil.rmtree(args.data_folder)
        else:
            raise FileExistsError(f"Destination folder '{args.data_folder}' already exists"
                                  f" and overwriting was not requested")
    args.data_folder.mkdir(parents=True, exist_ok=False)

    algo = DEAPWrap(args)
    champion = algo.run(args.generations)
    print()

    # Re-evaluate manually to get metrics
    result = algo.evaluate(champion, return_metrics=True)
    if result.fitness != champion.fitness.values[0]:
        err += 1
        print(f"/!\\ Fitness reevaluation gave different value /!\\\n"
              f"\t{result.fitness} != {champion.fitness.values[0]}\n")

    # Save and plot
    path = algo.save(champion, result.metrics)
    algo.plot(args.data_folder)
    print()

    # Re-evaluate "remotely" to generate all additional data (including video)
    err += rerun(args, path)

    duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start_time))
    print(f"Completed evolution in {duration} with exit code {err}")

    return err


if __name__ == "__main__":
    exit(main(Config.parse_command_line_arguments("NSGA-II test")))

# Get some learning in there (with RevDE or CMA?)
# Move onto actual fitness (5 targets with appropriate inputs for ABCpg)
# Add forced symmetries (morphological and controller)
