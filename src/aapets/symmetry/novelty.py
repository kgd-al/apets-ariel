import math
import pickle
from pathlib import Path
from typing import Optional, List

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, gridspec
from matplotlib.image import AxesImage
from sklearn.neighbors import NearestNeighbors
from torch._export.db.examples import class_method

from .config import Config


class NoveltyArchive:
    BINS = 25  # Just for the plotting
    DATA_FILE = "novelty.pkl"
    PLOT_FILE = "novelty.png"

    ROW_WIDTH = 4
    ROW_HEIGHT = 2

    def __init__(self, config: Config):
        self.k = config.novelty_knn
        self.add_threshold = config.novelty_add_threshold
        self.archive = []

        self.detailed = config.novelty_plots
        self.detailed_data: Optional[List] = None

    def process_generation(self, footprints):
        k = min(self.k, len(self.archive) + len(footprints))
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(self.archive + footprints)

        novelties = []
        for footprint in footprints:
            assert all(0 <= x <= 1 for x in footprint), "Footprint must be normalized in [0, 1]"

            dist, _ = nn.kneighbors([footprint])
            n = dist.mean()
            novelties.append(n)

            if n > self.add_threshold:
                self.archive.append(footprint)

        if self.detailed:
            if self.detailed_data is None:
                # Make it a list of shape D x G * BINS
                self.detailed_data = [[] for _ in footprints[0]]

            for d in self.detailed_data:
                d.append([0] * self.BINS)

            for footprint in footprints:
                for i, f in enumerate(footprint):
                    self.detailed_data[i][-1][min(self.BINS-1, int(f * self.BINS))] += 1

        return novelties

    def save(self, folder: Path):
        with open(folder.joinpath(self.DATA_FILE), "wb") as f:
            pickle.dump(dict(archive=self.archive, data=self.detailed_data), f)

    @classmethod
    def plot_from(cls, folder: Path):
        with open(folder.joinpath(cls.DATA_FILE), "rb") as f:
            data = pickle.load(f)
            archive = data["archive"]

            if (detailed_data := data.get("data")) is not None:
                cls.plot_to(archive, detailed_data, folder)

    @classmethod
    def plot_to(cls, archive, detailed_data, folder: Path) -> None:
        assert isinstance(detailed_data, list)
        plots = len(detailed_data)
        rows, cols = math.floor(math.sqrt(plots)), math.ceil(math.sqrt(plots))
        names = [f"D_{i}" for i in range(plots)]

        fig = plt.figure(figsize=(cls.ROW_WIDTH * cols, cls.ROW_HEIGHT * rows))
        gs = gridspec.GridSpec(rows, cols + 1, width_ratios=[1] * cols + [0.05], figure=fig)

        axes = np.array([
            [fig.add_subplot(gs[r, c]) for c in range(cols)]
            for r in range(rows)
        ])
        cbar_ax = fig.add_subplot(gs[:, -1])  # spans all rows

        for i, (name, ax) in enumerate(zip(names, axes.flatten())):
            row, col = i // cols, i % cols
            is_last = (i == plots - 1)

            data = np.array(detailed_data[i]).T
            print(data.shape)
            im = ax.imshow(
                data,
                aspect='auto',
                origin='lower',
                # ax=ax,
                # yticklabels=[f"{(i+.5)/cls.BINS:g}" for i in range(cls.BINS)] if col == 0 else False,
                cmap="Blues",
                # cbar=is_last,
                # cbar_ax=fig.add_axes([0.92, 0.15, 0.02, 0.7]) if is_last else None,
                extent=[0, data.shape[1]-1, 0, 1]
            )
            ax.set_title(name)
            if row == rows - 1:
                ax.set_xlabel("Generation")
            if col == 0:
                ax.set_ylabel("Distribution")
            else:
                ax.set_yticks([])

            if is_last:
                fig.colorbar(im, cax=cbar_ax)

        plt.tight_layout()  # leave room for colorbar
        file = folder.joinpath(cls.PLOT_FILE)
        fig.savefig(file, bbox_inches="tight")
        print("Plotted detailed novelty data to", file)
