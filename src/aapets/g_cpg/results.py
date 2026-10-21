
import argparse
import glob
import itertools
from pathlib import Path
import warnings

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from statannotations.Annotator import Annotator
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from aapets.common.morphological_measures import measure
from aapets.common.robot_storage import RerunnableRobot
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
parser.add_argument("--failure-threshold", default=50, type=float,
                    help="Minimum score to count as a pass"
                         " (for task that measure performance instead of speed)")
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
parser.add_argument("--no-morphology", dest="morphos", default=True, action="store_false",
                    help="Whether to try and merge the morphological descriptors")
parser.add_argument("--no-multi-task-evaluation", dest="evals", default=True, action="store_false",
                    help="Whether to try and merge the results fom multi-task evaluation")

parser.add_argument("--all-morphos-correlations", default=False, action="store_true",
                    help="Whether to investigate correlations between morphology and success")

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

sym_order = [Symmetry.BOTH.value, Symmetry.BODY.value, Symmetry.NONE.value]
train_order = [Task.LOCOMOTION, Task.DIRECTION, Task.COMPLIANCE]
morphos_order = [FixedMorphology.SPIDER, FixedMorphology.ARIEL_ANT,
                 FixedMorphology.GYM_ANT,
                 FixedMorphology.UNITREE_GO1, FixedMorphology.UNITREE_GO1_FIXED]

merged_evals = dict(
    Pathing=["-circle", "+circle", "-figure8", "+figure8", "shuttlerun"]
)

sorted_evals = ['Pathing', 'Obstacles', 'Fetch', 'Carrying', 'Parking']
score_based_evals = ['Fetch', 'Carrying', 'Parking']


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
            __df[m_value] = __path.parent.parent.parent.name
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
                        if np.isfinite(v := df.loc[_path, e]) and (
                            e not in score_based_evals or v > args.failure_threshold
                        ):
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
        # raise e

    df = df.infer_objects()

    print("Saving aggregated df:")
    print(df.columns)
    print(df)
    print("--------------")
    print()

    df.to_csv(df_file)

# ==============================================================================

mm_file = args.root.joinpath("morphologies.csv")
if args.purge and mm_file.exists():
    mm_file.unlink()

if mm_file.exists():
    mm_df = pd.read_csv(mm_file, index_col=0)

else:
    try:
        series = []
        for r in tqdm(runs, desc="Reading morphology csvs"):
            if Path(r).parts[0] != "evo":
                continue
            f: Path = args.root.joinpath(r).with_stem("champion.morphology")
            if not f.exists():
                specs = RerunnableRobot.load(f.parent.joinpath("champion.zip")).mj_spec
                s = pd.Series(measure(specs, "apet").major_metrics)
                s.to_csv(f)
            else:
                s = pd.read_csv(f, index_col=0).squeeze("columns")
            s.name = str(f.parent)
            series.append(s)

        mm_df = pd.DataFrame(series)
        mm_df.to_csv(mm_file)

    except Exception as e:
        e.add_note("Failed to process morphological descriptors")
        raise e

print()

# ==============================================================================
#

df.rename(inplace=True, columns=col_mapping)

print("Replacing stock Unitree Go1 with modified, working version")
df = df[df[m_value] != FixedMorphology.UNITREE_GO1.value].copy()
df.loc[df[m_value] == FixedMorphology.UNITREE_GO1_FIXED.value, m_value] = FixedMorphology.UNITREE_GO1.value
morphos_order.remove(FixedMorphology.UNITREE_GO1_FIXED.value)

df = df.assign(
    **{task: pd.Categorical(df[task], categories=train_order, ordered=True),
       symmetry: pd.Categorical(df[symmetry], categories=sym_order, ordered=True),
       m_value: pd.Categorical(df[m_value], categories=morphos_order, ordered=True)}
).sort_values([task, symmetry, m_value])

def pretty_multieval(e):
    name, sign = e.split("_")[1], ""
    if name[0] == "-":
        sign = " (Clockwise)"
    elif name[0] == "+":
        sign = " (Counter-clockwise)"
    if sign != "":
        name = name[1:]
    return name.capitalize() + sign

for new_e, e_columns in merged_evals.items():
    new_e = f"{multi_eval}_{new_e}"
    prefixed_cols = [f"{multi_eval}_{_e}" for _e in e_columns]
    df[new_e] = df[prefixed_cols].mean(axis=1)
    for _e in prefixed_cols:
        evals.remove(_e)
    evals.append(new_e)

evals_renaming = {e: pretty_multieval(e) for e in evals}
df.rename(inplace=True, columns=evals_renaming)
evals = sorted(list(evals_renaming.values()))
print("Test tasks in dataframe:", evals)

assert set(evals) == set(sorted_evals), f"Evaluations mismatch: {set(evals)} {set(sorted_evals)}"

assert set(df[m_type].unique()) == {"evo", "fixed"}
evo_df = df[df[m_type] == "evo"]
fixed_df = df[df[m_type] == "fixed"]

morphos = sorted(list(df[m_value].dropna().unique()))
print("Fixed morphologies in dataframe:", morphos)
assert set(morphos) == set(morphos_order)

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

def section_page(pdf, title, subtitle=None):
    fig = plt.figure(figsize=(8.5, 11))  # match your page size
    fig.text(0.5, 0.5, title, ha='center', va='center', fontsize=28, weight='bold')
    if subtitle:
        fig.text(0.5, 0.42, subtitle, ha='center', va='center', fontsize=14, color='gray')
    plt.axis('off')
    pdf.savefig(fig)
    plt.close(fig)

# ==============================================================================

def maybe_save(_g, _is_synthesis, *, title, cols=None, ratio=None, tight=True, **kwargs):
    if not isinstance(_g, Figure):
        _g = _g.figure

    if _is_synthesis:
        if cols is not None:
            original_size = _g.get_size_inches()
            width = textwidth / (cols * 1.01)
            height = width if ratio is None else width * ratio
            _g.set_size_inches(width, height)
        synthesis_pdf.savefig(_g, bbox_inches="tight", **kwargs)
        if cols is not None:
            _g.set_size_inches(*original_size)

    if title is not None:
        _g.suptitle(title, y=1, verticalalignment="bottom")
    if tight:
        _g.tight_layout()
    if not args.synthesis:
        summary_pdf.savefig(_g, bbox_inches="tight", **kwargs)
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

def fmt_label(lbl): return lbl.replace("_", " ").capitalize()

# ==============================================================================

def boxplot_with_success_rate(__df, __cols, __split, __title,
                              hatches=None, stats=False, labels=True, synthesis=True):
    
    conditions = list(__df[__split].unique()) if __split is not None else []

    v_lim = (-5, 105)
    fig, axes = plt.subplot_mosaic(
        [[c, "overall"] for c in __cols],
        width_ratios=[len(evals), 1 + .3*len(conditions)],
        gridspec_kw=dict(wspace=0.05))

    _bp_args = dict(data=__df, y=__split, hue=__split,
                    legend=False, patch_artist=True)

    first_ax = axes[__cols[0]]
    for c in __cols:
        ax = axes[c]
        sns.boxplot(**_bp_args, x=c, showfliers=False, ax=ax)

        ax.set_xlim(*v_lim)    
        ax.set_ylabel(c)

        if ax != first_ax:
            ax.sharex(first_ax)
        
        if c != __cols[-1]:
            ax.tick_params(labelbottom=False)
            ax.set_xlabel(None)
        else:
            ax.set_xlabel("Performance (\\%)")

        ax.set_yticks(ax.get_yticks(), [fmt_label(yl.get_text()) for yl in ax.get_yticklabels()])

        if labels:
            if __split:
                counts = __df[[c, __split]].replace(-np.inf, np.nan).groupby(
                    __split, observed=False)[c].count().to_dict()
            else:
                counts = {None: __df[c].replace(-np.inf, np.nan).count()}  
            trans = blended_transform_factory(ax.transAxes, ax.transData)

            ax.tick_params(axis="y", pad=5)
            for count, y_pos in zip(counts.values(), ax.get_yticks(), strict=True):
                ax.text(
                    0, y_pos, f"{count}",
                    transform=trans,
                    ha='right', va='center_baseline',
                    fontsize='x-small', color='gray',
                )

    ax1 = axes["overall"]
    ax1.yaxis.set_visible(False)
    ax1.set_yticklabels([])
    ax1.set_xlabel(__split)
    ax1 = ax1.twinx()
    _vp_args = dict(data=__df, x=__split, y=success_ratio, hue=__split, ax=ax1)
    sns.boxplot(**_vp_args, showfliers=False, legend=True)
    sns.stripplot(**_vp_args, **(stripplot_common_args | dict(
        color=sns.color_palette()[0] if __split is None else None)))
    ax1.set_xticks([], [])
    ax1.set_ylim(*v_lim)

    if __split is not None and stats:
        offset = 0
        pairs = list(itertools.combinations(conditions, r=2))
        annotator = Annotator(pairs=pairs, plot='violinplot', **_vp_args)
        annotator.configure(**annotator_configuration)
        annotator.apply_test().annotate(line_offset_to_group=offset)

    maybe_save(fig, synthesis, title=__title, tight=False)


# ==============================================================================

def spiderplot(df, x, y, col, *, band=(.25, .75), title=None, synthesis=True):
    def closed(a): return np.append(a, a[0])
    columns = list(df[col].unique())
    levels = list(df[x].unique())
    pal = dict(zip(levels, sns.color_palette("colorblind", len(levels)+1)[1:]))
    rows = int(np.sqrt(len(columns)))
    cols = len(columns) // rows
    fig, axes = plt.subplots(rows, cols, subplot_kw=dict(projection="polar"))
    for c, ax in zip(columns, axes.flatten()):
        g = df[df[col] == c].groupby(x, observed=True)[y]
        q = g.quantile([band[0], .5, band[1]]).unstack()
        lo, med, hi = q.to_numpy().T
        th = np.linspace(0, 2*np.pi, len(q), endpoint=False)
        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)
        ax.fill_between(closed(th), closed(lo), closed(hi), alpha=.3)
        ax.plot(closed(th), closed(med), marker="o")
        ax.scatter(th, med, c=[pal[k] for k in q.index], s=40, zorder=3, clip_on=False)
        ax.set_xticks([], [])
        ax.set_xlabel(fmt_label(c))
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(["", "50", "", "100"], fontsize=8, color="0.4")
        ax.vlines(th, 0, 105, color="0.6", lw=1, zorder=0)
        ax.set_rlabel_position(180/len(th))

    fig.canvas.draw()                                     # finalise the constrained layout
    inv = fig.transFigure.inverted()
    top = max(ax.get_tightbbox().transformed(inv).y1 for ax in axes.flatten())   # incl. titles
    fig.legend(
        [Line2D([], [], marker="o", ls="", color=pal[k]) for k in levels],
        [fmt_label(lbl) for lbl in levels], ncol=len(levels), #frameon=False,
        loc="lower center", bbox_to_anchor=(0.5, top + .01))

    maybe_save(fig, synthesis, title=title, cols=.75, tight=True)


# ==============================================================================

def plot_brain_activity(subpath: str, title: str, only_sections = None):
    __df = pd.read_csv(args.root.joinpath(subpath))

    columns = [c for c in __df.columns if c[-3:] == "pos"]
    if "index" in __df.columns and only_sections is not None:
        cols = len(only_sections)
    else:
        cols = 1

    fig, axes = plt.subplots(len(columns), cols, sharex="col")
    palette = sns.color_palette("colorblind", 3)

    def do_plot(_axes, _df):
        n = len(_axes)
        for i, (c, ax) in enumerate(zip(columns, _axes)):
            color = palette[2 if c.split("-")[1][0] in "rb" else 1]
            ax.margins(x=0, y=0.03)
            ax.plot(_df["Time"], _df[c], lw=0.5, color=color)
            ax.set_yticks([], [])
            ax.xaxis.set_major_locator(MultipleLocator(5))
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.grid(True, axis="x", which="minor", linewidth=0.2)
            ax.set_zorder(n - i)

        fig.canvas.draw()
        for x_lbl in _axes[-1].xaxis.get_major_ticks():
            if x_lbl.get_loc() == 0:
                x_lbl.label1.set_horizontalalignment("left")
                break
        _axes[-1].set_xlabel("Time (s)")

    if cols > 1:
        starts = np.flatnonzero(__df["index"].to_numpy() == 0)
        starts = np.append(starts, __df.index[-1])
        __dfs = [__df.iloc[starts[i]:starts[i+1]] for i in only_sections]
        for _df, _axes in zip(__dfs, axes.T):
            do_plot(_axes, _df)
    else:
        do_plot(axes, __df)

    fig.legend(
        [Line2D([], [], ls="-", color=palette[i+1]) for i in range(2)],
        ["left", "right"], ncol=2, title="Actuator's side",
        frameon=False, columnspacing=0.3,
        loc="lower center", bbox_to_anchor=(0.5, 1))

    fig.subplots_adjust(hspace=0, wspace=.025, left=0.0, right=1.0, top=1.0, bottom=0.0)

    maybe_save(fig, True, title=title, cols=.5, ratio=2/len(columns), pad_inches=0)

# ==============================================================================

pdf_summary_file = args.root.joinpath(".summary.pdf")
pdf_synthesis_file = args.root.joinpath(".synthesis.pdf")
print("Plotting...")
with PdfPages(pdf_summary_file) as summary_pdf, PdfPages(pdf_synthesis_file) as synthesis_pdf:
    section_page(summary_pdf, "Comparative performance for evolved morphologies")

    # ----

    sl_df = evo_df[(evo_df[symmetry] == "both") & (evo_df[task] == "locomotion")]
    
    # ----

    # ====
    # ----    

    boxplot_with_success_rate(
        sl_df, sorted_evals, None,
        "Overall performance for best condition")

    plot_brain_activity("evo/locomotion/both/run-6/champion.eval.+figure8.csv",
                        title="CPG activity for figure of 8 task")
    plot_brain_activity("evo/locomotion/both/run-6/champion.eval.parking.csv",
                        title="CPG activity for parking task",
                        only_sections=[0, 7])

    boxplot_with_success_rate(
        evo_df[evo_df[task] == "locomotion"], sorted_evals, symmetry,
        "Impact of symmetry on multi-task testing performance")

    boxplot_with_success_rate(
        evo_df[evo_df[symmetry] == "both"], sorted_evals, task,
        "Impact of training type on multi-task testing performance")

    for t in fixed_df[task].unique():
        boxplot_with_success_rate(
            fixed_df[fixed_df[task] == t], sorted_evals, m_value,
            f"Multi-task testing performance on multiple fixed morphologies with {t} training",
            synthesis=False)

    spiderplot(fixed_df, x=task, y=success_ratio, col=m_value,
               title="Impact of training type on fixed morphologies")

    plot_brain_activity("../../../papers/g_cpg/hardware/zoo-run-35.hinges.csv",
                        title="CPG activity for hardware figure of 8 task")

    # long_df = df.melt(
    #     id_vars=[m_value, task],                 # column(s) to keep as-is (x axis / hue)
    #     value_vars=sorted_evals,
    #     var_name="Task",              # holds the old column names
    #     value_name="value",              # holds the old column values
    # )
    # _args = dict(x=m_value, y="value", hue=task, order=sorted_morphos)
    # g = sns.catplot(data=long_df, col="Task", sharey=True, **_args, kind="box")
    # g.map_dataframe(sns.swarmplot, **_args, dodge=True, **(stripplot_common_args | dict(palette="dark:black")))
    # maybe_save(g, True, title="Impact of training on multi-task testing performance (fixed morphos)")

    # _args = dict(data=fixed_df, x=m_value, y=success_ratio, hue=task, order=sorted_morphos)
    # g = sns.boxplot(**_args)
    # sns.swarmplot(**_args, **(stripplot_common_args | dict(palette="dark:black", size=1)), ax=g.axes)
    # maybe_save(g, True, title="Impact of training on multi-task testing performance (fixed morphos)")

    # # ----    

    # _args = dict(data=evo_df, x=symmetry, y=speed, order=sym_order)

    # g = sns.catplot(kind='violin', **(violinplot_common_args | _args | dict(hue=symmetry, inner="quart", col=task)))
    # g.map_dataframe(sns.stripplot, **_args, **stripplot_common_args)

    # for ax in g.axes.flatten():
    #     annotator = Annotator(ax=ax, pairs=group_pairs, plot='violinplot', **_args)
    #     annotator.configure(**annotator_configuration)
    #     _, corrected_results = annotator.apply_test().annotate(line_offset_to_group=.1)

    # maybe_save(g, False, title="Speed for each training group and symmetry type")

    # # ----

    # for c in [modules, hinges, bricks]:
    #     g = sns.relplot(kind="scatter", data=evo_df, x=c, y=speed, hue=symmetry, col=task)
    #     maybe_save(g, False, title=f"Speed versus number of {c}")

    # # ----

    # for c in evals:
    #     _args = dict(
    #         data=evo_df, x=symmetry, y=c,
    #         order=sym_order, hue=task, dodge=True
    #     )

    #     with InfsAsNans(evo_df, c):
    #         g = sns.violinplot(**(violinplot_common_args | _args
    #                                         | dict(inner="quart", split=True,
    #                                                 common_norm=False, density_norm="count")))
    #         sns.stripplot(**_args, **(stripplot_common_args | dict(color=None, edgecolor='black', linewidth=1)))
    #         g.axes.set_ylim(0, 100)

    #         # for ax in g.axes.flatten():
    #         #     annotator = Annotator(ax=ax, pairs=group_pairs, plot='violinplot', **_args)
    #         #     annotator.configure(**annotator_configuration)
    #         #     _, corrected_results = annotator.apply_test(nan_policy='omit').annotate(line_offset_to_group=.1)

    #         maybe_save(g, False, title=f"Performance on {c} task for each training group and symmetry type")

    # # ----

    # _args = dict(
    #     data=evo_df, x=symmetry, y=success_ratio,
    #     order=sym_order, hue=task, dodge=True
    # )
    # _violin_args = dict(inner="quart", split=True, common_norm=True, density_norm="count")
    # g = sns.violinplot(**(violinplot_common_args | _args | _violin_args))
    # sns.stripplot(**_args, **(stripplot_common_args | dict(color=None, edgecolor='black', linewidth=1)))

    # # annotator = Annotator(ax=g.axes, pairs=ts_group_pairs, plot='violinplot', **(_args | _violin_args))
    # # annotator.configure(**annotator_configuration)
    # # _, corrected_results = annotator.apply_test(nan_policy='omit').annotate(line_offset_to_group=.1)

    # maybe_save(g, False, title="Overall success rate for each training group and symmetry type")

    # ----

    def adjust_font_size(g):
        probe = plt.figure()
        t = probe.text(0, 0, "Ag", fontsize=8)          # includes ascender + descender
        cell = t.get_window_extent(probe.canvas.get_renderer()).height / probe.dpi * 1.15
        plt.close(probe)
    
        fig, ncols, nrows = g.figure, len(evals), len(_sorted_df)
        for _ in range(2):
            fig.draw_without_rendering()
            pos, (W, H) = fig.axes[0].get_position(), fig.get_size_inches()
            fig.set_size_inches(ncols * cell + (W - pos.width * W),
                                nrows * cell + (H - pos.height * H))

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
    adjust_font_size(g)

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
        
    maybe_save(g, False, title="Overall performance (natural order)")

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
    adjust_font_size(g)
    maybe_save(g, False, title="Overall performance (descending)")

    # ----

    for morpho in morphos_order:
        sort_keys = [success_ratio, success_avg]    

        gap = ""
        _sorted_df = fixed_df[df[m_value] == morpho][sort_keys + evals].copy()
        _sorted_df[gap] = np.nan

        _sorted_df = _sorted_df[evals + [gap] + sort_keys].sort_values(sort_keys, ascending=False)
        g = sns.heatmap(_sorted_df,
                        cmap=cmap, vmin=0, vmax=100,
                        yticklabels=True,
                        linewidths=0.5, linecolor="lightgray", square=True,
                        cbar_kws={"label": "Score (%)"})
        adjust_font_size(g)
        maybe_save(g, False, title=f"Overall performance ({morpho} - descending)")

    # =============

if args.all_morphos_correlations:
    pdf_corr_summary = pdf_summary_file.with_name("correlations.pdf")
    with PdfPages(pdf_corr_summary) as summary_pdf:
        for c_lhs in [task, symmetry]:
            section_page(summary_pdf, f"Morphological differences between {c_lhs}")

            for c_rhs in mm_df.columns:
                g = sns.boxplot(x=df[c_lhs], y=mm_df[c_rhs], hue=df[c_lhs])
                maybe_save(g, True, title=f"Impact of {c_lhs.lower()} on {c_rhs}")

        section_page(summary_pdf, "Relationships between morphology and success")

        # ----

        for __df, __df_name in [(df, "all"), (evo_df, "evo")]:
            for c in ["avg_z", "std_z", "avg_pitch", "avg_roll", "std_pitch", "std_roll",
                    "instability_avg", "instability_std"]:
                g = sns.lmplot(data=__df, x=c, y=success_ratio)
                maybe_save(g, False, title=f"Relationship between {c} and {success_ratio} ({__df_name})")

        # ----

        for c_rhs in [success_ratio] + evals:
            for c_lhs in mm_df.columns:
                __data = pd.DataFrame({c_lhs: mm_df[c_lhs], c_rhs: evo_df[c_rhs], symmetry: evo_df[symmetry]})
                g = sns.lmplot(data=__data, x=c_lhs, y=c_rhs, hue=symmetry)
                maybe_save(g, False, title=f"Relationship between {c_lhs} and {c_rhs} (evo)")


for file in [pdf_summary_file, pdf_synthesis_file]:
    if file.exists():
        new_path = file.parent.joinpath(file.name[1:])
        file.rename(new_path)
        print("Generated", new_path)

print("Done.")
