
import argparse
import glob
import itertools
from pathlib import Path
import warnings

from matplotlib.colors import LinearSegmentedColormap
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
parser.add_argument("--no-multi-task-evaluation", dest="evals", default=False, action="store_false",
                    help="Whether to try and merge the results fom multi-task evaluation")

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
    def read_csv(r):
        __path = args.root.joinpath(r)
        __df = pd.read_csv(__path, index_col=0)
        __df.index = [str(__path.parent)]
        return __df
    df = pd.concat(read_csv(r) for r in tqdm(runs, desc="Reading csvs"))

    # df.index = df.index.map(lambda _p: _p.replace("/home/kgd/data", str(args.root.parent.parent)))

    if args.evals:
        try:
            series = []
            for r in tqdm(runs, desc="Reading eval csvs"):
                f: Path = args.root.joinpath(r).with_stem("champion.evaluation")
                s = pd.read_csv(f, index_col=0).squeeze("columns")  # -> Series
                s.name = str(f.parent)
                series.append(s)

            df = df.join(pd.concat(series, axis=1).T.add_prefix(f"{multi_eval}_"))
            
        except Exception as e:
            e.add_note("Failed to merge multi-task results. Did you run them?")
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

df = df.assign(
    **{task: pd.Categorical(df[task], categories=train_order, ordered=True),
       symmetry: pd.Categorical(df[symmetry], categories=sym_order, ordered=True)}
).sort_values([task, symmetry])

evals = [c for c in df.columns if c.startswith(multi_eval)]
sided_evals = [e for e in evals if e.split("_")[1][0] == "+"]

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

sided_evals = [(evals_renaming[e], evals_renaming[e.replace("_+", "_-")]) for e in sided_evals]

success_ratio = "Success ratio"
df[success_ratio] = 100 * df[evals].apply(np.isfinite).sum(axis=1) / len(evals)
success_avg = "Average Success"
df[success_avg] = df[evals].replace(-np.inf, np.nan).mean(axis=1)


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

def _pareto(_df, _lhs, _rhs, _lhs_sign=+1, _rhs_sign=+1):
    _points = np.array([(_lhs_sign * a, _rhs_sign * b) for a, b in zip(_df[_lhs], df[_rhs])])
    _original_points = _points.copy()
    # Credit goes to https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    is_efficient = np.arange(_points.shape[0])
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(_points):
        nondominated_point_mask = np.any(_points > _points[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        _points = _points[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    _order = sorted(is_efficient, key=lambda i: np.atan2(_original_points[i][1], _original_points[i][0]))
    return _df.iloc[_order]

hw_candidates = _pareto(df[df.index.str.contains(r"fixed/.*/spider", regex=True)], "|avg_y|", "std_z", -1, -1)
print(hw_candidates[["|avg_y|", "std_z"]])
print(" ".join(hw_candidates.index))
print("Got the pareto: exiting")
exit(42)

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
    print("Saved", title if title is not None else "Untitled figure")
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
ts_group_pairs = [
    t for t in itertools.combinations([(s, t) for t in train_order for s in sym_order], 2)
    if t[0][0] == t[1][0] or t[0][1] == t[1][1]
]

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
            g.axes.set_ylim(0, 100)

            # for ax in g.axes.flatten():
            #     annotator = Annotator(ax=ax, pairs=group_pairs, plot='violinplot', **_args)
            #     annotator.configure(**annotator_configuration)
            #     _, corrected_results = annotator.apply_test(nan_policy='omit').annotate(line_offset_to_group=.1)

            maybe_save(g, True, title=f"Performance on {c} task for each training group and symmetry type")

    # --

    for ep, em in sided_evals:
        name = ep[:ep.find('(')]
        _args = dict(
            data=df, x=symmetry, y=(df[ep]-df[em]).abs(),
            order=sym_order, hue=task, dodge=True
        )
        _violin_args = dict(inner="quart", split=True, common_norm=True, density_norm="count")
        g = sns.violinplot(**(violinplot_common_args | _args | _violin_args))
        sns.stripplot(**_args, **(stripplot_common_args | dict(color=None, edgecolor='black', linewidth=1)))
        g.axes.set_ylabel("Asymmetry")

        # annotator = Annotator(ax=g.axes, pairs=ts_group_pairs, plot='violinplot', **(_args | _violin_args))
        # annotator.configure(**annotator_configuration)
        # _, corrected_results = annotator.apply_test(nan_policy='omit').annotate(line_offset_to_group=.1)

        maybe_save(g, True, title=f"Performance asymmetry on {name} for each training group and symmetry type")

    # --

    _args = dict(
        data=df, x=symmetry, y=success_ratio,
        order=sym_order, hue=task, dodge=True
    )
    _violin_args = dict(inner="quart", split=True, common_norm=True, density_norm="count")
    g = sns.violinplot(**(violinplot_common_args | _args | _violin_args))
    sns.stripplot(**_args, **(stripplot_common_args | dict(color=None, edgecolor='black', linewidth=1)))

    annotator = Annotator(ax=g.axes, pairs=ts_group_pairs, plot='violinplot', **(_args | _violin_args))
    annotator.configure(**annotator_configuration)
    _, corrected_results = annotator.apply_test(nan_policy='omit').annotate(line_offset_to_group=.1)

    maybe_save(g, True, title="Overall success rate for each training group and symmetry type")

    # --

    cmap = LinearSegmentedColormap.from_list("red_green", ["red", "green"])
    cmap.set_bad("white")
    sort_keys = [task, symmetry, "run"]
    _sorted_df = df[sort_keys + evals].sort_values(sort_keys)
    g = sns.heatmap(_sorted_df[evals],
                    cmap=cmap, vmin=0, vmax=100,
                    yticklabels=True,
                    linewidths=0.5, linecolor="lightgray", square=True,
                    cbar_kws={"label": "score"})
    ax = g.axes
    sym_sizes = _sorted_df.groupby([task, symmetry], sort=False, observed=True).size()
    sym_pos = np.cumsum(sym_sizes.tolist())[:-1]        # every sym-block boundary

    train_sizes = _sorted_df.groupby(task, sort=False, observed=True).size()
    train_pos = np.cumsum(train_sizes.tolist())[:-1]    # only training-block boundaries

    sym_only_pos = [y for y in sym_pos if y not in train_pos]  # avoid drawing both at same spot

    for y in sym_only_pos:
        ax.axhline(y, color="black", linewidth=1.5)
    for y in train_pos:
        ax.axhline(y, color="black", linewidth=3)
    maybe_save(g, True, cols=.25, title="Overall performance (natural order)")

    # --

    sort_keys = [success_ratio, success_avg]    

    gap = ""
    _sorted_df = df[sort_keys + evals].copy()
    _sorted_df[gap] = np.nan

    _sorted_df = _sorted_df[evals + [gap] + sort_keys].sort_values(sort_keys, ascending=False)
    g = sns.heatmap(_sorted_df,
                    cmap=cmap, vmin=0, vmax=100,
                    yticklabels=True,
                    linewidths=0.5, linecolor="lightgray", square=True,
                    cbar_kws={"label": "Score (%)"})
    maybe_save(g, True, cols=.25, title="Overall performance (descending)")

    # =============

for file in [pdf_summary_file, pdf_synthesis_file]:
    if file.exists():
        new_path = file.parent.joinpath(file.name[1:])
        file.rename(new_path)
        print("Generated", new_path)

print("Done.")
