import time
from multiprocessing import Queue
from pathlib import Path
from typing import List

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

from .types import Individual


def shaded_plots(df: pd.DataFrame, out: Path):
    chapters = df.columns.get_level_values(0).unique()
    chapters = [c for c in chapters if c != "_"]

    fig, axes = plt.subplots(len(chapters), 1,
                             figsize=(10, 4 * len(chapters)),
                             sharex=True)

    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

    gens = df[("_", "gen")]
    for ax, chapter in zip(axes, chapters):
        df_c = df[chapter]
        cols = df_c.columns
        if (n := len(cols)) % 2 != 1:
            continue
        alpha = 1/(2*len(cols))
        mid = (n - 1)//2

        handles = []

        ax.plot(gens, df_c[cols[mid]], label=cols[mid])
        handles.append(ax.plot([], [], color=color, linewidth=2, label=cols[mid])[0])

        for i in range(mid):
            a, b = mid-i-1, mid+i+1
            ax.fill_between(gens, df_c[cols[a]], df_c[cols[b]], alpha=alpha, color=color)
            handles.append(Patch(facecolor=color, alpha=(mid-i) * alpha, label=f"{cols[a]}-{cols[b]}"))
        ax.set_title(chapter)
        ax.legend(handles=handles)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("generation")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Plotted distributions for {len(chapters)} chapters in", out)


def min_max_plots(df: pd.DataFrame, out: Path):
    chapters = df.columns.get_level_values(0).unique()
    chapters = [c for c in chapters if c != "_"]

    fig, axes = plt.subplots(len(chapters), 1,
                             figsize=(10, 4 * len(chapters)),
                             sharex=True)

    for ax, chapter in zip(axes, chapters):
        for col in df[chapter].columns:
            ax.plot(df[("_", "gen")], df[chapter][col], label=col)
        ax.set_title(chapter)
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("generation")
    plt.tight_layout()

    print(f"Plotted ranges for {len(chapters)} chapters in", out)
    plt.savefig(out, dpi=150)


class Genealogy:
    def __init__(self, folder: Path):
        self.path = folder.joinpath("genealogy.csv")
        self.file = open(self.path, "w")
        self.file.write(",".join(self.header()) + "\n")

    @staticmethod
    def header(): return ["Gen", "ID", "Fitness", "Parent1", "Parent2"]

    def write(self, gen: int, ind: Individual):
        self.file.write(",".join(str(x) for x in
                                 [gen, ind.id, ind.fitness.values[0], *ind.parents])
                        + "\n")

    def close(self): self.file.close()


class LearningLog:
    _queue: Queue = None

    @classmethod
    def queue(cls): return cls._queue

    @classmethod
    def init_queue(cls, q: Queue):
        cls._queue = q

    @staticmethod
    def header(): return ["Time", "id", "L_step", "L_id", "fitness"]

    @classmethod
    def log_pop(cls, ind_id, l_step, fitnesses: List[float]):
        for i, f in enumerate(fitnesses):
            cls._queue.put((time.perf_counter_ns(), ind_id, l_step, i, -f[0]))

    @classmethod
    def close_queue(cls):
        cls._queue.put(None)

    @staticmethod
    def file(folder: Path): return folder.joinpath("learning.csv")

    @classmethod
    def writer(cls, q: Queue, folder: Path):
        def fmt(items: List): return ",".join(map(str, items)) + "\n"
        with open(cls.file(folder), "w") as f:
            f.write(fmt(cls.header()))
            f.flush()

            while True:
                record = q.get()
                if record is None:
                    break

                f.write(fmt(record))
                f.flush()

    @classmethod
    def plot(cls, folder: Path):
        print("Not printing learning curves (yet?)")
        return
        df = pd.read_csv(cls.file(folder))
        print(df)
