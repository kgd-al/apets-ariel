import shutil

from aapets.symmetry.config import Config
from aapets.symmetry.deap_impl import DEAPWrap


def main(args: Config):
    print(args)

    assert args.data_folder is not None
    if args.overwrite:
        shutil.rmtree(args.data_folder)
    elif args.data_folder.exists():
        raise FileExistsError(f"Destination folder '{args.data_folder}' already exists"
                              f" and overwriting was not requested")
    args.data_folder.mkdir(parents=True, exist_ok=False)

    algo = DEAPWrap(args)
    algo.run(args.generations)

    algo.finalize()

    # Save champion + rerun
    # Look at pareto (with pymoo?)
    # Move onto actual fitness (5 targets with appropriate inputs for ABCpg)
    # Get some learning in there (with RevDE or CMA?)
    # Add forced symetries (morphological and controller)


if __name__ == "__main__":
    main(Config.parse_command_line_arguments("NSGA-II test"))
