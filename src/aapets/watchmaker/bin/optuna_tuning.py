import functools

import optuna

from .evo import main as evo_main
from ..config import WatchmakerConfig, RunTypes


def optimize(trial, args):
    args.mutation_scale = trial.suggest_float("mutation_scale", 0, 2)
    return evo_main(args)


def main():
    study = optuna.create_study()
    args = WatchmakerConfig(
        body="spider45",
        run_type=RunTypes.HILL,
        max_evaluations=11,#101,
    )
    args.update()

    study.optimize(functools.partial(optimize, args=args),
                   timeout=30, n_jobs=1, show_progress_bar=True)

    print(study.best_params)
    print(study)


if __name__ == "__main__":
    main()
