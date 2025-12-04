import time
from pathlib import Path

import graphviz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from aapets.watchmaker.config import WatchmakerConfig


class Plotter:
    def __init__(self, watchmaker: 'Watchmaker'):
        self.watchmaker = watchmaker
        self._evolution_file = self.evolution_file(watchmaker.config)
        self._interaction_file = self.interaction_file(watchmaker.config)

    @staticmethod
    def evolution_file(config: WatchmakerConfig): return config.data_folder.joinpath("evolution.csv")

    @staticmethod
    def interaction_file(config: WatchmakerConfig): return config.data_folder.joinpath("interactions.csv")

    def reset(self):
        self.record_evolution_data(header=True)
        self.record_interaction_data(None, header=True)

    def record_evolution_data(self, header=False):
        with open(self._evolution_file, "w" if header else "a") as f:
            if header:
                f.write("GenID IndID ParID Speed "
                        + " ".join([f"Gene{i}" for i in range(self.watchmaker.genetic_data.size)])
                        + "\n")
            else:
                for individual in self.watchmaker.population:
                    f.write(f"{self.watchmaker.generation} {individual.to_string()}\n")

    def record_interaction_data(self, selected_index, header=False):
        with open(self._interaction_file, "w" if header else "a") as f:
            if header:
                f.write("GenID Selection_time Selected_ix Selected_value"
                        " Max_value Min_value Precision_abs Precision_rel\n")
            else:
                f_select = self.watchmaker.population[selected_index].fitness

                f_min, f_max = np.quantile([i.fitness for i in self.watchmaker.population], [0, 1])
                precision_rel = (f_select - f_min) / (f_max - f_min)
                precision_abs = f_max - f_select

                f.write(" ".join(str(x) for x in [
                    self.watchmaker.generation,
                    time.time() - self.watchmaker.selection_time,
                    selected_index, f_select,
                    f_min, f_max,
                    precision_abs, precision_rel,
                    "\n"
                ]))

    @classmethod
    def do_final_plots(cls, config: WatchmakerConfig):
        e_df = pd.read_csv(cls.evolution_file(config), sep=" ")
        i_df = pd.read_csv(cls.interaction_file(config), sep=" ", index_col=False)
        out = config.data_folder
        # out = Path(".")
        ext = config.plot_extension

        pop_size = len(e_df[e_df.GenID == 0])
        config.population_size = pop_size

        e_df.Speed *= 100
        i_df.Selected_value *= 100

        cls.do_fitness_plot(e_df, i_df, out, config)
        cls.do_genealogy_plot(e_df, out, config)
        cls.do_interaction_plot(i_df, out, config)

    @staticmethod
    def do_fitness_plot(e_df: pd.DataFrame, i_df: pd.DataFrame, folder: Path, config: WatchmakerConfig):
        fig, ax = plt.subplots()

        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Fitness (cm / s)")

        n = config.population_size
        max_id = e_df.IndID.max()

        index = e_df.index
        index = list((n - 1) * (1 + (index / n).astype(int)) + 1)
        ax.plot((n - 1) * (i_df.index+1) + 1, i_df.Selected_value, 'k--', linewidth=.2)
        artist = ax.scatter(index, e_df.Speed, c=100 * e_df.IndID / max_id, s=2)
        cb = fig.colorbar(artist)
        cb.set_label("Recency (%)")

        fig.savefig(folder.joinpath(f"fitness.{config.plot_extension}"), bbox_inches="tight")

    @staticmethod
    def do_genealogy_plot(e_df: pd.DataFrame, folder: Path, config: WatchmakerConfig):
        dot = graphviz.Digraph()

        n = 9
        dot.attr('node', colorscheme=f"ylorrd{n}", style="filled")

        f_min, f_max = e_df.Speed.agg(["min", "max"])
        f_range = f_max - f_min

        def fillcolor(_f): return f"{int((n - 1) * (_f - f_min) / f_range) + 1}"

        edges = set()

        with dot.subgraph(name="cluster_genealogy") as g:
            g.attr(label="Genealogy")
            for gen_id, ind_id, par_id, f in e_df[["GenID", "IndID", "ParID", "Speed"]].itertuples(index=False):
                g.node(name=f"{ind_id}", fillcolor=fillcolor(f))
                if par_id >= 0 and (par_id, ind_id) not in edges:
                    g.edge(f"{par_id}", f"{ind_id}")
                    edges.add((par_id, ind_id))

        with dot.subgraph(name="cluster_colormap") as g:
            g.attr(label="Colormap\n(cm / s)")
            g.attr("node", shape="rectangle")
            g.attr(ranksep="0")
            for i in range(n):
                f = f_range * (i / (n-1)) + f_min
                g.node(name=f"CM{i}",
                       label=f"{f:.2g}",
                       fillcolor=fillcolor(f))
                if i > 0:
                    g.edge(f"CM{i}", f"CM{i-1}", style="invis")

        dot.render(outfile=folder.joinpath(f"genealogy.{config.plot_extension}"), cleanup=True)

    @staticmethod
    def do_interaction_plot(i_df: pd.DataFrame, folder: Path, config: WatchmakerConfig):
        fig, axes = plt.subplots(ncols=2)

        i_df.Precision_rel *= 100
        i_df.Precision_abs *= 100
        for col, label, ax in zip(["Precision_rel", "Precision_abs"],
                                  ["Relative precision (%)", "Absolute precision (cm / s)"],
                                  axes):
            ax.set_xlabel("Steps")
            ax.set_ylabel(col)
            ax.plot(i_df.index, i_df[col])

            axr = ax.twinx()
            axr.set_ylabel("Selection time")
            axr.plot(i_df.index, i_df.Selection_time)

        # index = e_df.index
        # index = list((n - 1) * (1 + (index / n).astype(int)) + 1)
        # ax.plot((n - 1) * (i_df.index+1) + 1, i_df.Selected_value, 'k--', linewidth=.2)
        # artist = ax.scatter(index, e_df.Speed, c=100 * e_df.IndID / max_id, s=2)
        # cb = fig.colorbar(artist)
        # cb.set_label("Recency (%)")

        fig.tight_layout()
        fig.savefig(folder.joinpath(f"interactions.{config.plot_extension}"), bbox_inches="tight")
