import pprint
import shutil
import time
from datetime import timedelta
from pathlib import Path

import humanize
import mujoco
import pandas as pd

from aapets.bin.rerun import Arguments as RerunArguments, main as _rerun
from aapets.common.config import ViewerModes
from aapets.common.metrics_storage import EvaluationMetrics
from aapets.g_cpg.config import Config, Task
from aapets.g_cpg.deap_impl import DEAPWrap
from aapets.g_cpg.types import Individual

def get_time(): return time.perf_counter()

def rerun(args: Config, champion: Path):
    print()
    print("#"*60)
    print("Reevaluating", champion)

    # Generic rerun. For phenotype rendering and such
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
    rerun_args.plot_rewards = False
    rerun_args.render_brain_genotype = False
    rerun_args.render_brain_phenotype = False
    rerun_args.record_position = True
    rerun_args.record_joints = True

    err = _rerun(rerun_args)

    if args.task is Task.COMPLIANCE:
        # Task-specific rerun. For brain activity and such
        rerun_args = RerunArguments.copy_from(args)
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

        print(f"{champion.parent}.glob({champion.stem + '_*.zip'})")
        for f in champion.parent.glob(champion.stem + "_*.zip"):
            rerun_args.robot_archive = f
            print()
            print("#"*60)
            print("Re-evaluating for sub-task", f)
            err += _rerun(rerun_args)
            print()

        # TODO Merge trajectories
        # TODO Plot alpha/beta values alongside brain activity

    return err

def make_summary(args, champion: Individual, metrics: EvaluationMetrics, start_time: float):
    folder = args.data_folder

    model = mujoco.MjModel.from_xml_string(champion.body)
    def count(what: str):
        return sum(1 for i in range(model.nbody) if model.body(i).name.endswith(what))
    
    hinges, bricks = count("hinge"), count("brick")

    summary = dict(
        task=args.task.value,
        symmetry=args.symmetry.value,
        run=args.seed,
        fitness=champion.fitness.values[0],
        modules=hinges+bricks,
        bricks=bricks,
        hinges=hinges,
        params=len(champion.weights),
        wall_time=get_time() - start_time,
    )
    summary.update(metrics.data)

    summary = pd.DataFrame.from_dict({k: [v] for k, v in summary.items()})
    summary.index = [folder]

    summary.to_csv(folder.joinpath("summary.csv"))
    print(summary.to_string())

def main(args: Config):
    start_time = get_time()
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
        ignored_files = ["slurm"]
        files = [f for f in args.data_folder.glob("*")
                 if not any(f.name.startswith(prefix) for prefix in ignored_files)]
        if args.overwrite:
            shutil.rmtree(args.data_folder)
        elif len(files) > 0:
            raise FileExistsError(f"Destination folder '{args.data_folder}' already exists,"
                                  f" is not empty (excluding {ignored_files})"
                                  f" and overwriting was not requested")
    args.data_folder.mkdir(parents=True, exist_ok=True)

    if args.rev_de_knn_sample_size < args.rev_de_knn_neighborhood:
        raise ValueError("RevDEKNN has lower sample size than KNN neighborhood")

    algo = DEAPWrap(args)
    champion = algo.run(args.generations)
    print()

    # Re-evaluate manually to get metrics
    print("#" * 40)
    print("Re-evaluating in-place:")
    result = algo.evaluate(champion, return_metrics=True)
    if result.fitness != champion.fitness.values[0]:
        err += 1
        print(f"/!\\ Fitness reevaluation gave different value /!\\\n"
              f"\t{result.fitness} != {champion.fitness.values[0]}\n")
    else:
        print("Final evaluation metrics:")
        pprint.pprint(result)

    # Save and plot
    path = algo.save(champion, result.metrics)
    print("Saved champion to", path)
    algo.plot(args.data_folder)
    print()

    make_summary(args, champion, result.metrics, start_time)

    # Re-evaluate "remotely" to generate all additional data (including video)
    args.verbosity = 1
    err += rerun(args, path)

    duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start_time))
    print(f"Completed evolution in {duration} with exit code {err}")

    return err


if __name__ == "__main__":
    exit(main(Config.parse_command_line_arguments("NSGA-II test")))

# Get some learning in there (with RevDE or CMA?)
# Move onto actual fitness (5 targets with appropriate inputs for ABCpg)
# Add forced symmetries (morphological and controller)
