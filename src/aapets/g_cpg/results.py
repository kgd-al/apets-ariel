
import argparse
import glob
import itertools
from pathlib import Path
import warnings

from matplotlib.collections import FillBetweenPolyCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from statannotations.Annotator import Annotator
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from aapets.g_cpg.config import FixedMorphology, Symmetry, Task

from .testing_tasks.all import invalid


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
parser.add_argument("--no-multi-task-evaluation", dest="evals", default=True, action="store_false",
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
m_type = "Morphology type"
m_value = "Morphology"
success_ratio = "Success ratio"
success_avg = "Average Success"

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
    evals = [c for c in df.columns if c.startswith(multi_eval)]
    print("Loaded existing df:")
    print(df)

else:
    def read_csv(r):
        __path = args.root.joinpath(r)
        __df = pd.read_csv(__path, index_col=0)
        __df.index = [str(__path.parent)]
        __df[m_type] = _m_type = __path.parent.parent.parent.parent.name
        if _m_type == "fixed":
            __df[m_value] = __path.parent.parent.name
        return __df
    df = pd.concat(read_csv(r) for r in tqdm(runs, desc="Reading csvs"))

    # df.index = df.index.map(lambda _p: _p.replace("/home/kgd/data", str(args.root.parent.parent)))

    if args.evals:
        try:
            series, missing = [], []
            for r in tqdm(runs, desc="Reading eval csvs"):
                f: Path = args.root.joinpath(r).with_stem("champion.evaluation")
                if not f.exists():
                    missing.append(f)
                    continue
                s = pd.read_csv(f, index_col=0).squeeze("columns")  # -> Series
                s.name = str(f.parent)
                series.append(s)

            df = df.join(pd.concat(series, axis=1).T.add_prefix(f"{multi_eval}_"))

            if len(missing) > 0:
                print("Missing evaluations for:")
                for f in missing:
                    print(">", f)

            evals = [c for c in df.columns if c.startswith(multi_eval)]
            _base_evals = [e.replace(f"{multi_eval}_", "") for e in evals]
            def compute_success_ratio(_path):
                success, total = 0, 0
                for e, _e in zip(evals, _base_evals):
                    if not invalid(Path(_path).joinpath("champion.zip"), _e):
                        total += 1
                        if np.isfinite(df.loc[_path, e]):
                            success += 1

                return 100 * success / total if total > 0 else 0
            df[success_ratio] = df.index.map(compute_success_ratio)
            df[success_avg] = df[evals].replace(-np.inf, np.nan).mean(axis=1)

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

def pretty_multieval(e):
    name, sign = e.split("_")[1], ""
    if name[0] == "-":
        sign = " (Clockwise)"
    elif name[0] == "+":
        sign = " (Counter-clockwise)"
    if sign != "":
        name = name[1:]
    return name.capitalize() + sign

sided_evals = [e for e in evals if e.split("_")[1][0] == "+"]
for e_plus in sided_evals:
    e_minus = e_plus.replace("_+", "_-")
    for _e in [e_plus, e_minus]:
        evals.remove(_e)
    e = e_plus.replace("_+", "_")
    evals.append(e)
    df[e] = df[[e_plus, e_minus]].mean(axis=1)

evals_renaming = {e: pretty_multieval(e) for e in evals}
df.rename(inplace=True, columns=evals_renaming)
evals = sorted(list(evals_renaming.values()))
print("Test tasks in dataframe:", evals)

sorted_evals = [
    'Shuttlerun', 'Circle', 'Figure8',
    'Fetch', 'Obstacles'
]
assert set(evals) == set(sorted_evals), f"Evaluations mismatch: {set(evals)} {set(sorted_evals)}"

cycle = sns.color_palette()
evals_palette = {
    "Shuttlerun": cycle[0],
    "Circle": cycle[1],
    "Figure8": cycle[2],
    "Fetch": cycle[3], 
    "Obstacles": cycle[4], 
    success_ratio: "gray"
}

assert set(df[m_type].unique()) == {"evo", "fixed"}
evo_df = df[df[m_type] == "evo"]
fixed_df = df[df[m_type] == "fixed"]

morphos = sorted(list(df[m_value].dropna().unique()))
print("Fixed morphologies in dataframe:", morphos)
sorted_morphos = ['spider', 'ariel_ant', 'gym_ant']
assert set(morphos) == set(sorted_morphos)


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

# hw_candidates = _pareto(df[df.index.str.contains(r"fixed/.*/spider", regex=True)], "|avg_y|", "std_z", -1, -1)
# print(hw_candidates[["|avg_y|", "std_z"]])
# print(" ".join(hw_candidates.index))
# print("Got the pareto: exiting")
# exit(42)

# ==============================================================================

def section_page(pdf, title, subtitle=None):
    fig = plt.figure(figsize=(8.5, 11))  # match your page size
    fig.text(0.5, 0.5, title, ha='center', va='center', fontsize=28, weight='bold')
    if subtitle:
        fig.text(0.5, 0.42, subtitle, ha='center', va='center', fontsize=14, color='gray')
    plt.axis('off')
    pdf.savefig(fig)
    plt.close(fig)

# ==============================================================================

def maybe_save(_g, _is_synthesis, *, title, cols=None, ratio=None, tight=True):
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
    if tight:
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
    color='gray', size=3, legend=False,
    edgecolor=None, linewidth=1
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

def barplot_with_success_rate(__df, __cols, __split, __title, hatches=None):
    conditions = list(__df[__split].unique()) if __split is not None else []

    _task, _perf = "Task", "Performance (\\%)"
    melt_cols, id_vars = [c for c in __cols], []
    if __split is not None:
        melt_cols.append(__split)
        id_vars.append(__split)
    __m_df = __df[melt_cols].melt(id_vars=id_vars, var_name=_task, value_name=_perf)

    fig, axes = plt.subplots(1, 2, sharey=False, width_ratios=[len(evals), 1 + .3*len(conditions)],
                             gridspec_kw=dict(wspace=0.05))
    _bp_args = dict(data=__m_df, x=_task, y=_perf, hue=__split or _task,
                    ax=axes[0], legend=False, patch_artist=True)
    if __split is None:
        _bp_args["palette"] = [evals_palette[col] for col in __cols]
    else:
        _bp_args["palette"] = "gray"

    sns.boxplot(**_bp_args, showfliers=False)
    axes[0].set_ylabel("Performance (\\%)")
    pretty_labels = [e.replace(" (Clockwise)", "\n(CW)").replace(" (Counter-clockwise)", "\n(CCW)")
                     for e in __cols]

    # axes[0].set_ylim(0, 100)    
    axes[0].set_xticks(range(len(pretty_labels)), labels=pretty_labels)
    axes[0].set_xlabel("Task")

    ax1 = axes[1].twinx()
    ax1.sharey(axes[0])
    _vp_args = dict(data=__df, x=__split, y=success_ratio, ax=ax1)
    sns.boxplot(**_vp_args, color=evals_palette[success_ratio], showfliers=False)
    sns.stripplot(**_vp_args, **stripplot_common_args)
    axes[1].yaxis.set_visible(False)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel(__split)

    if __split is not None:
        for cond_idx, (condition, container) in enumerate(zip(conditions, axes[0].containers)):
            sub = __m_df[__m_df[__split] == condition]
            has_finite = sub.groupby(_task)[_perf].apply(lambda s: np.isfinite(s).any())

            for patch, col in zip(container, [c for c in sorted_evals if has_finite.get(c)]):
                patch = patch.box
                patch.set_facecolor(evals_palette[col])
                if hatches is not None:
                    patch.set_hatch(hatches[cond_idx])  
                patch.set_edgecolor('black')
                patch.set_linewidth(0.5)

        handles = []
        for cond_idx, child in enumerate([c for c in ax1.get_children() if (isinstance(c, PathPatch))]):
            if hatches is not None:
                h = hatches[cond_idx]
                child.set_hatch(h)
                if h:
                    child.set_facecolor((0, 0, 0, 0))
                child.set_edgecolor('black')
            handles.append(child)

        axes[0].legend(handles=handles, labels=conditions, ncols=len(conditions))

    if __split is not None:
        offset = 0
        pairs = [((t, h[0]), (t, h[1])) for h in itertools.combinations(conditions, r=2) for t in __m_df[_task].unique()]
        annotator = Annotator(pairs=pairs, plot='barplot', **_bp_args)
        annotator.configure(**annotator_configuration)
        annotator.apply_test().annotate(line_offset_to_group=offset)

        pairs = list(itertools.combinations(conditions, r=2))
        annotator = Annotator(pairs=pairs, plot='violinplot', **_vp_args)
        annotator.configure(**annotator_configuration)
        annotator.apply_test().annotate(line_offset_to_group=offset)


    maybe_save(fig, True, title=__title, tight=False)


# ==============================================================================

pdf_summary_file = args.root.joinpath(".summary.pdf")
pdf_synthesis_file = args.root.joinpath(".synthesis.pdf")
print("Plotting...")
with PdfPages(pdf_summary_file) as summary_pdf, PdfPages(pdf_synthesis_file) as synthesis_pdf:
    section_page(summary_pdf, "Comparative performance for evolved morphologies")

    # ----

    sl_df = evo_df[(evo_df[symmetry] == "both") & (evo_df[task] == "locomotion")]
    # #
    g = sns.violinplot(data=sl_df[evals], orient='h', order=sorted_evals,
                       **(violinplot_common_args | dict(density_norm="count")))
    g.axes.set_xlabel("Performance (\\%)")
    maybe_save(g, True, title="Overall performance on multi-task testing (violin plot)")

    # ----

    _task, _perf = "Task", "Performance (\\%)"
    s_df = evo_df[evo_df[task] == "locomotion"][sorted_evals + [symmetry]].melt(
        id_vars=symmetry, var_name=_task, value_name=_perf)
    # #
    g = sns.catplot(kind='violin', data=s_df, 
                    x=_perf, y=_task, col=symmetry, hue=_task,
                    order=sorted_evals,
                    **(violinplot_common_args | dict(
                       density_norm="count", common_norm=True, legend=False)))
    for ax in g.axes.flatten(): 
        ax.axvline(80, color='red', linestyle="--", zorder=10)
    maybe_save(g, True, title="Impact of symmetry on multi-task testing performance (violin plot)")

    # ----

    _task, _perf = "Task", "Performance (\\%)"
    t_df = evo_df[evo_df[symmetry] == "both"][sorted_evals + [task]].melt(
        id_vars=task, var_name=_task, value_name=_perf)
    # #
    g = sns.catplot(kind='violin', data=t_df, 
                    x=_perf, y=_task, col=task, hue=_task,
                    order=sorted_evals,
                    **(violinplot_common_args | dict(
                       density_norm="count", common_norm=True, legend=False)))
    for ax in g.axes.flatten():
        ax.axvline(80, color='red', linestyle="--", zorder=-10)
    maybe_save(g, True, title="Impact of training on multi-task testing performance (violin plot)")

    # ----    

    _task, _perf = "Task", "Performance (\\%)"
    t_df = fixed_df[sorted_evals + [m_value, task]].melt(
        id_vars=[m_value, task], var_name=_task, value_name=_perf)
    # #
    g = sns.catplot(kind='violin', data=t_df, 
                    x=_perf, y=_task, col=m_value, row=task, hue=_task,
                    order=sorted_evals, col_order=sorted_morphos,
                    **(violinplot_common_args | dict(
                       density_norm="count", common_norm=True, legend=False)))
    for ax in g.axes.flatten(): 
        ax.axvline(80, color='red', linestyle="--", zorder=10)
    maybe_save(g, True, title="Impact of training on multi-task testing performance (violin plot)")

    # ====
    # ----    

    # Maybe boxplot instead?
    barplot_with_success_rate(
        sl_df, sorted_evals, None,
        "Overall performance on multi-task testing (bar+violin plot)")

    barplot_with_success_rate(
        evo_df[evo_df[task] == "locomotion"], sorted_evals, symmetry,
        "Impact of symmetry on multi-task testing performance",
        ['///', '\\\\\\', ''])

    barplot_with_success_rate(
        evo_df[evo_df[symmetry] == "both"], sorted_evals, task,
        "Impact of training type on multi-task testing performance",
        ['', 'XX'])

    for t in fixed_df[task].unique():
        barplot_with_success_rate(
            fixed_df[fixed_df[task] == t], sorted_evals, m_value,
            "Multi-task testing performance on multiple fixed morphologies")

    _args = dict(data=fixed_df, x=m_value, y=success_ratio, hue=task)
    g = sns.violinplot(**_args, split=True, **(violinplot_common_args | dict(density_norm="count", inner=None)))
    sns.swarmplot(**_args, dodge=True, **(stripplot_common_args | dict(palette="dark:black")), ax=g.axes)
    maybe_save(g, True, title="Impact of training on multi-task testing performance (fixed morphos)")

    # ----    

    _args = dict(data=evo_df, x=symmetry, y=speed, order=sym_order)

    g = sns.catplot(kind='violin', **(violinplot_common_args | _args | dict(hue=symmetry, inner="quart", col=task)))
    g.map_dataframe(sns.stripplot, **_args, **stripplot_common_args)

    for ax in g.axes.flatten():
        annotator = Annotator(ax=ax, pairs=group_pairs, plot='violinplot', **_args)
        annotator.configure(**annotator_configuration)
        _, corrected_results = annotator.apply_test().annotate(line_offset_to_group=.1)

    maybe_save(g, False, title="Speed for each training group and symmetry type")

    # ----

    for c in [modules, hinges, bricks]:
        g = sns.relplot(kind="scatter", data=evo_df, x=c, y=speed, hue=symmetry, col=task)
        maybe_save(g, False, title=f"Speed versus number of {c}")

    # ----

    for c in evals:
        _args = dict(
            data=evo_df, x=symmetry, y=c,
            order=sym_order, hue=task, dodge=True
        )

        with InfsAsNans(evo_df, c):
            g = sns.violinplot(**(violinplot_common_args | _args
                                            | dict(inner="quart", split=True,
                                                    common_norm=False, density_norm="count")))
            sns.stripplot(**_args, **(stripplot_common_args | dict(color=None, edgecolor='black', linewidth=1)))
            g.axes.set_ylim(0, 100)

            # for ax in g.axes.flatten():
            #     annotator = Annotator(ax=ax, pairs=group_pairs, plot='violinplot', **_args)
            #     annotator.configure(**annotator_configuration)
            #     _, corrected_results = annotator.apply_test(nan_policy='omit').annotate(line_offset_to_group=.1)

            maybe_save(g, False, title=f"Performance on {c} task for each training group and symmetry type")

    # ----

    _args = dict(
        data=evo_df, x=symmetry, y=success_ratio,
        order=sym_order, hue=task, dodge=True
    )
    _violin_args = dict(inner="quart", split=True, common_norm=True, density_norm="count")
    g = sns.violinplot(**(violinplot_common_args | _args | _violin_args))
    sns.stripplot(**_args, **(stripplot_common_args | dict(color=None, edgecolor='black', linewidth=1)))

    annotator = Annotator(ax=g.axes, pairs=ts_group_pairs, plot='violinplot', **(_args | _violin_args))
    annotator.configure(**annotator_configuration)
    _, corrected_results = annotator.apply_test(nan_policy='omit').annotate(line_offset_to_group=.1)

    maybe_save(g, False, title="Overall success rate for each training group and symmetry type")

    # ----

    cmap = LinearSegmentedColormap.from_list("red_green", ["red", "green"])
    cmap.set_bad("white")
    sort_keys = [task, symmetry, "run"]
    _sorted_df = evo_df[sort_keys + evals].sort_values(sort_keys)
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
    maybe_save(g, False, cols=.25, title="Overall performance (natural order)")

    # ----

    sort_keys = [success_ratio, success_avg]    

    gap = ""
    _sorted_df = evo_df[sort_keys + evals].copy()
    _sorted_df[gap] = np.nan

    _sorted_df = _sorted_df[evals + [gap] + sort_keys].sort_values(sort_keys, ascending=False)
    g = sns.heatmap(_sorted_df,
                    cmap=cmap, vmin=0, vmax=100,
                    yticklabels=True,
                    linewidths=0.5, linecolor="lightgray", square=True,
                    cbar_kws={"label": "Score (%)"})
    maybe_save(g, False, cols=.25, title="Overall performance (descending)")

    # =============

for file in [pdf_summary_file, pdf_synthesis_file]:
    if file.exists():
        new_path = file.parent.joinpath(file.name[1:])
        file.rename(new_path)
        print("Generated", new_path)

print("Done.")
