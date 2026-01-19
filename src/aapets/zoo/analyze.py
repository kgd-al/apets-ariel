import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, rcParams
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_pdf import PdfPages

from aapets.common import showcaser
from aapets.common.misc.config_base import IntrospectiveAbstractConfig


@dataclass
class Arguments(IntrospectiveAbstractConfig):
    regenerate: Annotated[bool, "Whether to regenerate intermediate data"] = False
    verbose: Annotated[bool, "Whether to be verbose"] = False


@lru_cache(maxsize=1)
def bodies_images():
    cache_folder = Path("tmp/cache")
    return showcaser.get_all_images(cache_folder, plt.imread)


def symlink(src, dst, verbose):
    if src.exists():
        if src.is_symlink():
            src.unlink()
        else:
            raise ValueError(f"Not overwriting regular file {src} with symlink")
    src.symlink_to(dst)
    if verbose:
        print("  ", src, "->", dst)


def add_body_images(fig):
    ax = fig.axes[0]
    ax.set_xlabel("")

    tick_labels = ax.xaxis.get_ticklabels()

    for tick in tick_labels:
        img = bodies_images().get(tick.get_text())
        ib = OffsetImage(img, zoom=.1)
        ib.image.axes = ax
        ab = AnnotationBbox(
            ib,
            tick.get_position(),
            frameon=False,
            box_alignment=(0.5, 2.2),
            boxcoords=("data", "axes points"),
        )
        ax.add_artist(ab)


def radarchart(*args, color, **kwargs):
    ax = plt.gca()
    y = np.array([a.to_numpy() for a in args])
    y = np.append(y, y[0])
    x = np.linspace(0, 2 * math.pi, len(y))
    ax.set_xticks(ticks=x[:-1], labels=[a.name for a in args])
    kwargs.update(dict(
        facecolor=(*color, .25),
        edgecolor=color,
    ))
    ax.fill(x, y, **kwargs)
    ax.set_ylim(0, 1)


def main():
    args = Arguments.parse_command_line_arguments("Analyze evolution data from zoo experiments")
    print(args)

    root = Path(__file__).parent.parent.parent.parent.joinpath("remote").joinpath("zoo")
    assert root.exists(), f"{root} does not exist"

    output = root.joinpath("_results")
    output.mkdir(exist_ok=True, parents=True)

    dfs = []
    df_path = output.joinpath("data.csv")

    if args.regenerate or not df_path.exists():
        for file in root.glob("[a-z]*/*/summary.csv"):
            _df = pd.read_csv(file, index_col=0)
            _df.index = ["/".join(file.parts[-3:-1])]
            _df.fitness *= -1
            dfs.append(_df)

        df: pd.DataFrame = pd.concat(dfs)
        df.to_csv(df_path)

    else:
        df = pd.read_csv(df_path, index_col=0)

    bodies = df["body"].unique()

    if args.verbose:
        print("Aggregated dataframe:")
        print(df)
        print()
        print(df.groupby("body").count()["run"])
        print()

    best_folder = root.joinpath("_best")
    if args.regenerate or not best_folder.exists():
        best_folder.mkdir(exist_ok=True, parents=True)
        champions = df.loc[
            df.groupby("body", dropna=False)["fitness"].idxmax()
        ]
        if args.verbose:
            print("Champions:")
            print(champions)
            print()
        for run_path, body in champions["body"].items():
            src, target = best_folder.joinpath(body), root.joinpath(run_path)
            symlink(src, target, args.verbose)
            src = best_folder.joinpath(body).with_suffix(".mp4")
            target = root.joinpath(run_path).joinpath("champion.mp4")
            symlink(src, target, args.verbose)

    gp = df.groupby("body")["fitness"].median()
    group_order = gp[gp.sort_values().index].index

    with PdfPages(output.joinpath("results.pdf")) as pdf:

        rcParams['figure.figsize'] = .9 * len(bodies), 5
        g = sns.violinplot(data=df, x="body", y="fitness",
                           order=group_order,
                           inner="quart", density_norm="count")
        add_body_images(g.figure)

        g.figure.suptitle("Fitness distribution per body type")
        pdf.savefig(g.figure, bbox_inches="tight")
        plt.close()

        mmetrics = ["branching", "limbs", "length_of_limbs", "coverage", "joints", "proportion", "symmetry"]

        mdf = df.groupby("body", as_index=False).first()
        if (mdf[mmetrics].min().min() < 0) or (1 < mdf[mmetrics].max().max()):
            print("/!\\/!\\/!\\/!\\/!\\/!\\")
            print("/!\\/!\\/!\\/!\\/!\\/!\\")
            print("/!\\/!\\/!\\/!\\/!\\/!\\")
            print("Invalid value in df")
            print("/!\\/!\\/!\\/!\\/!\\/!\\")
            print("/!\\/!\\/!\\/!\\/!\\/!\\")
            print("/!\\/!\\/!\\/!\\/!\\/!\\")

        g = sns.FacetGrid(mdf, col="body", col_wrap=math.ceil(math.sqrt(len(bodies))),
                          subplot_kws=dict(projection='polar'), height=4.5,
                          sharex=False, sharey=False, despine=False)

        g.map(radarchart, *mmetrics)
        pdf.savefig(g.figure, bbox_inches="tight")
        plt.close()

        for metric in mmetrics:
            g = sns.scatterplot(data=mdf, x="body", y=metric)
            g.figure.suptitle(f"Distribution for morphological descriptor '{metric}' per body type")
            add_body_images(g.figure)
            pdf.savefig(g.figure, bbox_inches="tight")
            plt.close()


if __name__ == '__main__':
    main()
