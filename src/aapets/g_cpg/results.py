
import argparse
import glob
import itertools
from pathlib import Path
import warnings

from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from statannotations.Annotator import Annotator
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from aapets.g_cpg.config import Symmetry, Task


matplotlib.use("agg")
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


parser = argparse.ArgumentParser("Summarizes summary.csv files")
parser.add_argument("root", type=Path)
parser.add_argument("--purge", default=False, action="store_true", help="Purge old showcased files")
parser.add_argument("--synthesis", default=False, action="store_true", help="Only produce synthesis plots")
parser.add_argument("-v", default=False, action="store_true",
                    help="Print logging info and debug")
# for plot_type in ["trajectories", "paretos", "relations",
#                   "perf_violins", "all_violins", "training_curves",
#                   "diversity"]:
#     parser.add_argument(f"--no-plot-{plot_type.replace('_', '-')}",
#                         dest=f"plot_{plot_type}",
#                         default=True, action="store_false",
#                         help=f"Whether to plot {plot_type}")
# parser.add_argument("--no-print-paretos",
#                     dest="print_paretos",
#                     default=True, action="store_false",
#                     help="Whether to print pareto fronts")

args = parser.parse_args()


sns.set_style("darkgrid")
plt.rcParams['text.usetex'] = True
textwidth = 347.12354 / 72.27  # inches
paper_plots = {
    "font.size": 6,
    "lines.markersize": 4,  # 10
    "lines.markeredgewidth": .5,  # 1.0
    "lines.linewidth": 1,  # 3
    "patch.linewidth": .5,  # 1
}
if args.synthesis:
    matplotlib.rcParams.update(paper_plots)

# ==============================================================================

runs = glob.glob("**/run-*/summary.csv", root_dir=args.root, recursive=True)

# ==============================================================================

col_mapping = {}

symmetry = col_mapping["symmetry"] = "Symmetry type"
speed = col_mapping["xspeed"] = "Speed (m/s)"
task = col_mapping["task"] = "Training type"
modules = col_mapping["modules"] = "Modules"
hinges = col_mapping["hinges"] = "Hinges"
bricks = col_mapping["bricks"] = "Bricks"

multi_eval = "multi-eval"

sym_order = [Symmetry.NONE.value, Symmetry.BODY.value, Symmetry.BOTH.value]
train_order = [Task.LOCOMOTION, Task.COMPLIANCE]

# ==============================================================================


str_root = str(args.root)
df_file = args.root.joinpath("summaries.csv")
if args.purge and df_file.exists():
    df_file.unlink()

if df_file.exists():
    df = pd.read_csv(df_file, index_col=0)
    print("Loaded existing df:")
    print(df)

else:
    df = pd.concat(
        pd.read_csv(args.root.joinpath(r), index_col=0)
        for r in tqdm(runs, desc="Reading csvs")
    )

    df.index = df.index.map(lambda _p: _p.replace("/home/kgd/data", str(args.root.parent.parent)))

    try:
        series = []
        for r in tqdm(runs, desc="Reading eval csvs"):
            f: Path = args.root.joinpath(r).with_stem("champion.evaluation")
            s = pd.read_csv(f, index_col=0).squeeze("columns")  # -> Series
            s.name = str(f.parent)
            series.append(s)

        df = df.join(pd.concat(series, axis=1).T.add_prefix(f"{multi_eval}_"))
        
    except Exception as e:
        print("Failed to merge multi-task results. Did you run them?")
        raise e

    try:
        def compute_x_speed(_path):
            __df = pd.read_csv(Path(_path).joinpath("champion.trajectory.csv")).iloc[-1]
            return __df["x"] / __df["t"]
        df["xspeed"] = df.index.map(compute_x_speed)

        def compute_avg_y(_path):
            return pd.read_csv(Path(_path).joinpath("champion.pos.csv"))["apet1_core-y"].mean()
        df["avg_y"] = df.index.map(compute_avg_y)
        df["|avg_y|"] = df["avg_y"].abs()

        def process_positions(row):
            prefix = "apet1_core-"
            pos_df = pd.read_csv(Path(row.name).joinpath("champion.pos.csv"))
            _x, _y, _z = [pos_df[prefix + d] for d in "xyz"]
            _roll, _pitch = pos_df[prefix + "R"], pos_df[prefix + "P"]
            return (
                _y.mean(), _z.mean(), _z.std(),
                _roll.mean(), _roll.std(), _pitch.mean(), _pitch.std(),
                _x.iloc[-1] - _x.iloc[0]
            )
        df[["avg_y", "avg_z", "std_z", "avg_roll", "std_roll", "avg_pitch", "std_pitch", "dX"]] = (
            df.apply(process_positions, axis=1, result_type="expand"))

        def compute_d_o(_path):
            joints_df = pd.read_csv(Path(_path).joinpath("champion.joints.csv"))
            joints_df = joints_df[[c for c in joints_df.columns if c[-5:] == "-ctrl"]]
            return (joints_df.iloc[1:].reset_index(drop=True) - joints_df.iloc[:-1]).abs().mean().mean()
        df["avg_d_o"] = df.index.map(compute_d_o)

        df["instability_avg"] = df[["avg_roll", "avg_pitch"]].abs().max(axis=1)
        df["instability_std"] = df[["std_roll", "std_pitch"]].max(axis=1)

    except Exception as e:
        print("Ignoring mild error", e)
        raise e

    df = df.infer_objects()

    print("Saving aggregated df:")
    print(df.columns)
    print(df)
    print("--------------")
    print()

    df.to_csv(df_file)

# ==============================================================================
#

df.rename(inplace=True, columns=col_mapping)

evals = [c for c in df.columns if c.startswith(multi_eval)]

def pretty_multieval(e):
    name, sign = e.split("_")[1], ""
    if name[0] == "-":
        sign = " (Clockwise)"
    elif name[0] == "+":
        sign = " (Counter-clockwise)"
    if sign != "":
        name = name[1:]
    return name.capitalize() + sign

evals_renaming = {e: pretty_multieval(e) for e in evals}
df.rename(inplace=True, columns=evals_renaming)
evals = sorted(list(evals_renaming.values()))


# ==============================================================================

class InfsAsNans:
    def __init__(self, df: pd.DataFrame, col: str):
        self.df, self.col = df, col
        self.mask = None

    def __enter__(self):
        self.mask = (self.df[self.col] == -np.inf)
        self.df.loc[self.mask, self.col] = np.nan
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.df.loc[self.mask, self.col] = -np.inf


# ==============================================================================

def maybe_save(_g, _is_synthesis, *, title, cols=None, ratio=None):
    if not isinstance(_g, Figure):
        _g = _g.figure

    if _is_synthesis:
        if cols is not None:
            original_size = _g.get_size_inches()
            width = textwidth / (cols * 1.01)
            height = width if ratio is None else width * ratio
            _g.set_size_inches(width, height)
        synthesis_pdf.savefig(_g, bbox_inches="tight")
        if cols is not None:
            _g.set_size_inches(*original_size)

    if title is not None:
        _g.suptitle(title, y=1, verticalalignment="bottom")
    _g.tight_layout()
    if not args.synthesis:
        summary_pdf.savefig(_g, bbox_inches="tight")
    plt.close()

# ==============================================================================

violinplot_common_args = dict(
    inner="box", cut=0, gap=.25,
    common_norm=True, density_norm="width"
)

stripplot_common_args = dict(
    color='black', size=3, legend=False
)

group_pairs = list(itertools.combinations(sym_order, 2))
annotator_configuration = dict(
    test="Mann-Whitney", verbose=0, loc="outside",
    hide_non_significant=False, text_format="star",
    comparisons_correction="bonferroni"
)

# ==============================================================================

pdf_summary_file = args.root.joinpath(".summary.pdf")
pdf_synthesis_file = args.root.joinpath(".synthesis.pdf")
print("Plotting...")
with PdfPages(pdf_summary_file) as summary_pdf, PdfPages(pdf_synthesis_file) as synthesis_pdf:

    _args = dict(
        data=df, x=symmetry, y=speed,
        order=sym_order,
    )

    g = sns.catplot(kind='violin', **(violinplot_common_args | _args | dict(hue=symmetry, inner="quart", col=task)))
    g.map_dataframe(sns.stripplot, **_args, **stripplot_common_args)

    for ax in g.axes.flatten():
        annotator = Annotator(ax=ax, pairs=group_pairs, plot='violinplot', **_args)
        annotator.configure(**annotator_configuration)
        _, corrected_results = annotator.apply_test().annotate(line_offset_to_group=.1)

    maybe_save(g, True, title="Speed for each training group and symmetry type")

    if True:
        g = sns.violinplot(**(violinplot_common_args | _args | dict(inner="quart", hue=task, split=True)))
        sns.stripplot(**_args, **stripplot_common_args)

        spider_df = pd.read_csv(args.root.parent.parent.joinpath("cpg_rl").joinpath("summaries.csv"))
        spider_df = spider_df[(spider_df["sub-arch"] == "cpg-6") & (spider_df.reward == "speed")]

        _args = dict(
            data=spider_df, x=4, y="speed", ax=g, color="red", 
        )
        sns.violinplot(**(violinplot_common_args | _args), label="Uncalibrated spider")
        sns.stripplot(**_args,
              marker="D", edgecolor='black', linewidth=1, jitter=False,
              zorder=10, legend=False)

        maybe_save(g, False, title="Speed for each training group and symmetry type (with cpg_rl spider)")

    for c in [modules, hinges, bricks]:
        g = sns.relplot(kind="scatter", data=df, x=c, y=speed, hue=symmetry, col=task)
        maybe_save(g, True, title=f"Speed versus number of {c}")

    for c in evals:
        _args = dict(
            data=df, x=symmetry, y=c,
            order=sym_order, hue=task, dodge=True
        )

        with InfsAsNans(df, c):
            g = sns.violinplot(**(violinplot_common_args | _args
                                            | dict(inner="quart", split=True,
                                                    common_norm=True, density_norm="count")))
            sns.stripplot(**_args, **(stripplot_common_args | dict(color=None, edgecolor='black', linewidth=1)))

            # for ax in g.axes.flatten():
            #     annotator = Annotator(ax=ax, pairs=group_pairs, plot='violinplot', **_args)
            #     annotator.configure(**annotator_configuration)
            #     _, corrected_results = annotator.apply_test(nan_policy='omit').annotate(line_offset_to_group=.1)

            maybe_save(g, True, title=f"Performance on {c} task for each training group and symmetry type")


    # ==========
    ts_col = "ts"
    ts_order = [f"{t}_{s}" for t in train_order for s in sym_order]
    df[ts_col] = pd.Categorical(
        df[task] + "_" + df[symmetry],
        categories=ts_order, ordered=True,
    )

    inter_group_pairs = list(itertools.combinations(ts_order, 2))

    for metric in evals:
        _args = dict(data=df, x=symmetry, y=metric, order=sym_order, hue=task, dodge=True)
        g = sns.barplot(**_args, errorbar="sd")
        ax = g.axes

        sns.stripplot(**_args,
                        **(stripplot_common_args | dict(color=None, edgecolor="black", linewidth=.5)),
                        ax=ax)

        # labels = [s.capitalize() for s in sym_order] * len(train_order)
        # ax.set_xticks(range(len(labels)), labels=labels)
        # ax.set_xlabel("Symmetry + Training type")
        # n = len(sym_order)

        # for i, t in enumerate(train_order):
        #     center = i * n + (n - 1) / 2
        #     ax.text(center, -0.08, t.capitalize(), transform=ax.get_xaxis_transform(),
        #             ha="center", va="top")

        # with InfsAsNans(df, metric):
        #     counts = df.groupby(ts_col, observed=True)[metric].count().reindex(ts_order)
        #     for i, n in enumerate(counts):
        #         ax.text(i, ax.get_ylim()[1]*0.02, f"n={n}", ha="center", va="bottom", fontsize=8)

        # ax.set_title(metric)
        # annotator = Annotator(ax=ax, pairs=inter_group_pairs, plot='barplot', **_args)
        # annotator.configure(**(annotator_configuration | dict(hide_non_significant=True)))
        # _, corrected_results = annotator.apply_test(nan_policy='omit').annotate(line_offset_to_group=.1)

        maybe_save(ax, True, title=f"Performance on {metric} task for each training group and symmetry type")

    # =============

for file in [pdf_summary_file, pdf_synthesis_file]:
    if file.exists():
        new_path = file.parent.joinpath(file.name[1:])
        file.rename(new_path)
        print("Generated", new_path)

print("Done.")
