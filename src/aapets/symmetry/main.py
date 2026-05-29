import shutil

from aapets.symmetry.config import Config
from aapets.symmetry.deap_impl import DEAPWrap
from aapets.common.config import ViewerModes
from aapets.bin.rerun import Arguments as RerunArguments, main as rerun


def main(args: Config):
    print(args)

    assert args.data_folder is not None
    if args.data_folder.exists():
        if args.overwrite:
            shutil.rmtree(args.data_folder)
        else:
            raise FileExistsError(f"Destination folder '{args.data_folder}' already exists"
                                  f" and overwriting was not requested")
    args.data_folder.mkdir(parents=True, exist_ok=False)

    algo = DEAPWrap(args)
    champion = algo.run(args.generations)

    algo.do_plots()
    path = algo.save(champion)

    rerun_args = RerunArguments.copy_from(args)
    rerun_args.robot_archive = path

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

    rerun(rerun_args)

    # Save champion + rerun
    # Look at pareto (with pymoo?)
    # Move onto actual fitness (5 targets with appropriate inputs for ABCpg)
    # Get some learning in there (with RevDE or CMA?)
    # Add forced symetries (morphological and controller)


if __name__ == "__main__":
    main(Config.parse_command_line_arguments("NSGA-II test"))
