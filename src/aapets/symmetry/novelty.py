from pathlib import Path

import math
import numpy as np
import pandas as pd
import pickle
from dataclasses import dataclass, field, asdict
from matplotlib import pyplot as plt, gridspec
from sklearn.neighbors import NearestNeighbors
from typing import Optional, List, Tuple

import seaborn as sns

from ..common.misc.debug import kgd_debug
from .config import Config


class NoveltyArchive:
    @dataclass
    class LoggedData:
        sizes: List[int] = field(default_factory=list)
        thresholds: List[float] = field(default_factory=list)
        novelties: List[List[Tuple[float, bool, bool]]] = field(default_factory=list)
        descriptors: List[List[np.ndarray]] = None

    BINS = 25  # Just for the plotting
    DATA_FILE = "novelty.pkl"
    PLOT_FILE = "novelty_{}.png"

    ROW_WIDTH = 4
    ROW_HEIGHT = 2

    def __init__(self, config: Config, names: Optional[List[str]] = None):
        self.k = config.novelty_knn
        self.add_threshold = config.novelty_add_threshold
        self.decimals = config.novelty_archive_decimals
        self.archive = set()

        self.thresholds = [config.novelty_initial_intake, config.novelty_intake]

        self.names = names
        self.data = self.LoggedData()

        self.max = 0

    def _tuple(self, arr: np.ndarray):
        return tuple(np.round(arr, self.decimals).tolist())

    def process_generation(self, footprints: List[np.ndarray]):
        kgd_debug(len(footprints))
        local_max = 0

        @dataclass
        class _Data:
            footprint: Tuple
            novelty: float = -np.inf
            new: bool = False
            added: bool = False

        footprints = [self._tuple(footprint) for footprint in footprints]
        footprints_dict = {f: _Data(f) for f in footprints}

        fit_data = list(self.archive | set(footprints_dict.keys()))
        k = min(self.k, len(fit_data))
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(fit_data)

        new = 0
        for data in footprints_dict.values():
            if any(x < 0 or 1 < x for x in data.footprint):
                raise RuntimeError(f"Footprint must be normalized in [0, 1]:\n{data.footprint}")

            dist, _ = nn.kneighbors([data.footprint])
            data.novelty = n = dist.mean()
            data.new = (data.footprint not in self.archive)
            new += data.new

            self.max = max(self.max, n)
            local_max = max(local_max, n)

        if new > 0:  # Just in case
            intake = 1-self.thresholds[len(self.archive) != 0]
            threshold = np.quantile([data.novelty for data in footprints_dict.values() if data.new], q=intake)
            initial_length = len(self.archive)
            for data in footprints_dict.values():
                old_length = len(self.archive)
                if data.new and data.novelty >= threshold:
                    self.archive.add(data.footprint)
                    data.added = True
                    if len(self.archive) != old_length + 1:
                        raise RuntimeError("Could not add footprint")
        else:
            threshold = np.nan

        # ===
        # Keep data for further analysis/plotting

        # Just the size over time
        self.data.sizes.append(len(self.archive))
        self.data.thresholds.append(threshold)

        # Novelty distribution
        self.data.novelties.append([(data.novelty, data.added, data.new)
                                    for data in footprints_dict.values()])

        # Descriptors distribution
        if self.data.descriptors is None:
            # Make it a list of shape D x G * BINS
            self.data.descriptors = [[] for _ in next(iter(footprints))]

        for d in self.data.descriptors:
            d.append(np.zeros(self.BINS))

        for footprint in footprints:
            for i, f in enumerate(footprint):
                self.data.descriptors[i][-1][min(self.BINS - 1, int(f * self.BINS))] += 1

        n_footprints = len(footprints)
        for i in range(len(self.data.descriptors)):
            self.data.descriptors[i][-1] /= n_footprints

        return [footprints_dict[footprint].novelty for footprint in footprints]

    def save(self, folder: Path):
        with open(folder.joinpath(self.DATA_FILE), "wb") as f:
            pickle.dump(dict(
                archive=self.archive, names=self.names,
                data=asdict(self.data)), f)

    @classmethod
    def plot_from(cls, folder: Path):
        with open(folder.joinpath(cls.DATA_FILE), "rb") as f:
            data = pickle.load(f)
            archive = data["archive"]
            names = data.get("names")

            log_data = cls.LoggedData(**data.get("data"))

            cls.plot_to(archive, log_data, names, folder)

    @classmethod
    def plot_to(cls, archive, data, names, folder: Path) -> None:
        cls.plot_size(data, folder)
        cls.plot_distribution(data, names, folder)

    @classmethod
    def plot_size(cls, data: LoggedData, folder: Path):
        fig, axes = plt.subplots(nrows=2, ncols=1)
        axes[0].plot(data.sizes)
        axes[0].set_ylabel("Archive size")

        df = pd.DataFrame(
            [(gen, novelty, added, new)
             for gen, pop in enumerate(data.novelties)
             for novelty, added, new in pop],
            columns=["Gen", "Novelty", "Added", "New"]
        )

        def _plot(**kwargs):
            sns.stripplot(x="Gen", y="Novelty", order=sorted(df.Gen.unique()),
                          ax=axes[1], jitter=True, **kwargs)

        _plot(data=df[df.Added], color="steelblue", alpha=.5)
        _plot(data=df[(~df.Added & df.New)], color="lightgray", alpha=.25, marker="v")
        _plot(data=df[~df.New], color="lightgray", alpha=.25, marker="^")
        axes[1].plot(data.thresholds, linestyle="dashed")

        file = folder.joinpath(cls.PLOT_FILE.format("size"))
        fig.savefig(file, bbox_inches="tight")
        print("Plotted archive size to", file)

    @classmethod
    def plot_distribution(cls, data: LoggedData, names: List[str], folder: Path) -> None:
        assert isinstance(data.descriptors, list)
        plots = len(data.descriptors)
        rows = math.ceil(math.sqrt(plots))
        cols = math.ceil(plots / rows)
        names = names or [f"D_{i}" for i in range(plots)]

        print(plots, rows, cols)

        fig = plt.figure(figsize=(cls.ROW_WIDTH * cols, cls.ROW_HEIGHT * rows))
        gs = gridspec.GridSpec(rows, cols + 1, width_ratios=[1] * cols + [0.05], figure=fig)
        im = None

        axes = np.array([
            [fig.add_subplot(gs[r, c]) for c in range(cols)]
            for r in range(rows)
        ])
        cbar_ax = fig.add_subplot(gs[:, -1])  # spans all rows

        for i, (name, ax) in enumerate(zip(names, axes.flatten())):
            row, col = i // cols, i % cols

            distribution = np.array(data.descriptors[i]).T
            im = ax.imshow(
                distribution,
                aspect='auto',
                origin='lower',
                cmap="Blues",
                extent=[0, distribution.shape[1]-1, 0, 1],
            )
            ax.set_title(name)
            if row == rows - 1:
                ax.set_xlabel("Generation")
            if col == 0:
                ax.set_ylabel("Distribution")
            else:
                ax.set_yticks([])

        for ax in axes.flatten()[plots:]:
            ax.axis("off")

        if im is not None:
            fig.colorbar(im, cax=cbar_ax, label="Frequency")

        plt.tight_layout()
        file = folder.joinpath(cls.PLOT_FILE.format("distribution"))
        fig.savefig(file, bbox_inches="tight")
        print("Plotted descriptors distribution to", file)
