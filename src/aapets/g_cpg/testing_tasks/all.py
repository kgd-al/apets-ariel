
from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Annotated, Optional

import pandas as pd
import rich
from rich.progress import Progress

from .config import TestingConfig
from . import pathing, fetching

from aapets.common.metrics_storage import GOOD, RESET

from aapets.g_cpg.config import Config


# Current tasks:
# - (WIP) Drone pathing
# - (?) Different durations (e.g. 60s evaluation, do they go further or just exploit the simulation?)
# - (?) Ball catching (very visual, too many parameters though)
# - (?) Obstacle avoidance


@dataclass
class Arguments(TestingConfig):
    threads: Annotated[Optional[int], "Max number of parallel processes to use (or cpu_count()-1)"] = None
    from_scratch: Annotated[bool, "Whether to reuse previously computed data or restart from zero"] = False


def prepare_tasks(args: Arguments):
    # modules = [pathing, fetching]
    modules = [fetching]
    return [task for module in modules for task in getattr(module, "prepare_tasks")(args)]


def persistent_data(champion: Path): return champion.with_suffix(".evaluation.csv")


if __name__ == "__main__":
    parser = ArgumentParser(description="Performs a suite of test on a number of robots"
                                        " to test their polyvalence")
    parser.add_argument("file", nargs="+", type=Path)
    Arguments.populate_argparser(parser)
    args = parser.parse_args(namespace=Arguments())
    args.pretty_print()

    n_files = len(args.file)
    
    tasks = prepare_tasks(args)
    n_tasks = len(tasks)

    progress_args = (
        rich.progress.SpinnerColumn(),
        *Progress.get_default_columns(),
        rich.progress.TimeElapsedColumn(),
        rich.progress.MofNCompleteColumn(),
    )
    progress_kwargs = dict(
        redirect_stdout=True,#(cli_args.verbosity > 0),
    )
    start_time = time.perf_counter()
    workers = args.threads or (os.cpu_count()-1)
    print("Using", workers, "workers")
    with Progress(*progress_args, **progress_kwargs) as progress, \
         ProcessPoolExecutor(max_workers=workers) as executor:
        
        taskbar = progress.add_task("Evaluating...", total=n_files * n_tasks)
        futures = []
        series, needs_write = dict(), defaultdict(list)
        already_completed = 0

        for champion in args.file:
            df_path = persistent_data(champion)
            if not df_path.exists():
                s = pd.Series(dtype=float, name="score")
                s.index.name = "name"
            else:
                s = pd.read_csv(df_path, index_col=0)
            series[champion] = s

            for task in tasks:
                if task.name not in s.index or args.from_scratch:
                    futures.append(executor.submit(task, champion))
                    needs_write[champion].append(task.name)
                else:
                    already_completed += 1

        progress.update(taskbar, advance=already_completed, description=f"Skipping existing {already_completed}")

        for future in as_completed(futures):
            champion, task, score = future.result()
            series[champion].loc[task] = score
            progress.update(taskbar, advance=1, description=f"{champion} / {task}: {score:.2f}%")

            needs_write[champion].remove(task)
            if len(needs_write[champion]) == 0:
                series[champion].to_csv(persistent_data(champion))
                print("> Wrote", persistent_data(champion))

        progress.update(
            taskbar,
            description=f"\n{GOOD}Evaluated {n_files} champions on {n_tasks} tasks"
                        f" in {time.perf_counter() - start_time:.3f} seconds{RESET}")
