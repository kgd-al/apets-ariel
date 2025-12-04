import argparse
import functools
import logging
import os
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Annotated

import optuna
from optuna import Trial
from optuna.study import StudyDirection

from aapets.common.misc.config_base import IntrospectiveAbstractConfig
from .evo import main as evo_main
from ..config import WatchmakerConfig, RunTypes


@dataclass
class OptunaConfig(IntrospectiveAbstractConfig):
    watchmaker_seeds: Annotated[str, "RNG seeds for the evolution"] = "1-30"
    threads: Annotated[int, "Number of concurrent threads (<= 1 disables parallelism)"] = 1
    timeout: Annotated[int, "Number of seconds to run for"] = 60
    data_folder: Annotated[Path, "Where to store persistent data"] = Path("results/optuna")


def to_range(seeds):
    return range(*[int(x) for x in seeds.split("-")])


def optimize(trial: Trial, _args: WatchmakerConfig, seeds):
    trial.user_attrs["seeds"] = seeds
    values = []
    for seed in to_range(seeds):
        _args.seed = seed
        _args.mutation_scale = trial.suggest_float("mutation_scale", 0, 2)
        _args.data_folder = _args.data_folder.joinpath(f"{trial.number}")
        values.append(evo_main(_args))

    return sum(values) / len(values)


def main(_, args: OptunaConfig):
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    args.data_folder.mkdir(parents=True, exist_ok=True)

    study_name = f"mutation-rate-optimization-seeds{args.watchmaker_seeds}"

    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///{args.data_folder.joinpath(study_name)}.db"
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction=StudyDirection.MAXIMIZE
    )
    study.set_metric_names(["fitness"])

    watchmaker_args = WatchmakerConfig(
        body="spider45",
        run_type=RunTypes.HILL,
        max_evaluations=201,
        mutation_range=2,
        mutation_scale=-1,
        parallelism=False,
        population_size=5,
        data_folder=args.data_folder,
        verbosity=0,
        no_plots=True,
        # timing=True,
    )
    watchmaker_args.update()
    for w_key, w_value in watchmaker_args.as_dict().items():
        if isinstance(w_value, Path):
            w_value = str(w_value)
        study.set_user_attr("watchmaker_" + w_key, w_value)
    watchmaker_args.write_yaml(args.data_folder.joinpath(study_name).with_suffix(".yaml"))

    study.optimize(functools.partial(optimize, _args=watchmaker_args, seeds=args.watchmaker_seeds),
                   timeout=args.timeout, n_jobs=1, show_progress_bar=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Main evolution script")
    OptunaConfig.populate_argparser(parser)
    # parser.print_help()
    config = parser.parse_args(namespace=OptunaConfig())

    if config.threads > 1:
        with Pool(processes=config.threads) as pool:
            pool.map(functools.partial(main, args=config), range(config.threads))
    else:
        main((), config)
