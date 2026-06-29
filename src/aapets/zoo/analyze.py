import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import seaborn as sns
from PIL.ImageOps import fit
from matplotlib import pyplot as plt, rcParams
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA

from aapets.misc import showcaser
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


def radarchart(body, *args, color, **kwargs):
    fig, ax = plt.gcf(), plt.gca()

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

    cax = fig.add_axes(ax.get_position().bounds, polar=False, frameon=False)
    cax.axis("off")
    cax.set_xlim(0, 200)
    cax.set_ylim(0, 200)
    cax.imshow(
        bodies_images().get(body.to_list()[0]),
        zorder=-10, alpha=.2, extent=(25, 175, 25, 175)
    )
    cax.autoscale(False)
    fig.canvas.draw()


def main():
    args = Arguments.parse_command_line_arguments("Analyze evolution data from zoo experiments")
    print(args)

    root = Path(__file__).parent.parent.parent.parent.joinpath("remote").joinpath("zoo")
    assert root.exists(), f"{root} does not exist"

    output = root.joinpath("_results")
    output.mkdir(exist_ok=True, parents=True)

    df_path = root.joinpath("summaries.csv")
    df = pd.read_csv(df_path, index_col=0)

    bodies = df["body"].unique()

    if args.verbose:
        print("Aggregated dataframe:")
        print(df)
        print()
        print(df.groupby("body").count()["run"])
        print()

    gp = df.groupby("body")["fitness"].median()
    group_order = gp[gp.sort_values().index].index

    mmetrics = ["branching", "limbs", "length_of_limbs", "coverage", "joints", "proportion", "symmetry"]

    mdf = df.groupby("body", as_index=False).first()

    sns.set_style('darkgrid')
    with PdfPages(output.joinpath("results.pdf")) as pdf:
        g = sns.pairplot(mdf[mmetrics])
        pdf.savefig(g.figure, bbox_inches="tight")
        plt.close()

        # ---

        fig = plt.figure(1, figsize=(8, 6))
        ax = fig.add_subplot(111)

        pca = PCA(n_components="mle")
        pca.fit(mdf[mmetrics])
        print("Explained variance:", pca.explained_variance_)
        print("Explained variance ratios:", pca.explained_variance_ratio_)
        print("Components:", pca.components_)
        x_reduced = pca.transform(mdf[mmetrics])
        g = sns.scatterplot(
            x=x_reduced[:, 0],
            y=x_reduced[:, 1],
            hue=mdf.body, hue_order=group_order,
            palette="magma",
            s=40,
        )

        def fmt_evr(i): return f"{100 * pca.explained_variance_ratio_[i]:.2f}%"

        ax.set(
            title="First two principal components",
            xlabel=f"1st Principal Component ({fmt_evr(0)})",
            ylabel=f"2nd Principal Component ({fmt_evr(1)})",
        )
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        g.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # ---

        rcParams['figure.figsize'] = .9 * len(bodies), 5
        g = sns.violinplot(data=df, x="body", y="fitness",
                           order=group_order,
                           inner="quart", density_norm="count", cut=0)
        add_body_images(g.figure)

        g.figure.suptitle("Fitness distribution per body type")
        pdf.savefig(g.figure, bbox_inches="tight")
        plt.close()

        # ---

        g = sns.FacetGrid(mdf, col="body", col_wrap=math.ceil(math.sqrt(len(bodies))),
                          col_order=group_order,
                          subplot_kws=dict(projection='polar'), height=4.5,
                          sharex=False, sharey=False, despine=False)

        g.map(radarchart, "body", *mmetrics)
        pdf.savefig(g.figure, bbox_inches="tight")
        plt.close()

        # ---

        for metric in mmetrics:
            g = sns.stripplot(data=mdf, x="body", y=metric, order=group_order)
            g.figure.suptitle(f"Distribution for morphological descriptor '{metric}' per body type")
            add_body_images(g.figure)
            pdf.savefig(g.figure, bbox_inches="tight")
            plt.close()

        # ---


if __name__ == '__main__':
    main()
