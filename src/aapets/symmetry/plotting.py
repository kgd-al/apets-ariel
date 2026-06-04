from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


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

