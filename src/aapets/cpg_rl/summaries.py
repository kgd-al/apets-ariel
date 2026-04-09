import argparse
import glob
import itertools
import math
import shutil
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, transforms
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from statannotations.Annotator import Annotator
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from aapets.cpg_rl.types import RewardToMonitor, Rewards

matplotlib.use("agg")
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

parser = argparse.ArgumentParser("Summarizes summary.csv files")
parser.add_argument("root", type=Path)
parser.add_argument("--purge", default=False, action="store_true", help="Purge old showcased files")
parser.add_argument("--synthesis", default=False, action="store_true", help="Only produce synthesis plots")
parser.add_argument("--distance-filtering", default=0,
                    help="Remove individuals that moved less than the given threshold")
parser.add_argument("-v", default=False, action="store_true",
                    help="Print logging info and debug")
for plot_type in ["trajectories", "paretos", "relations", "perf_violins", "all_violins", "training_curves"]:
    parser.add_argument(f"--no-plot-{plot_type.replace('_', '-')}",
                        dest=f"plot_{plot_type}",
                        default=True, action="store_false",
                        help=f"Whether to plot {plot_type}")
parser.add_argument(f"--no-print-paretos",
                    dest=f"print_paretos",
                    default=True, action="store_false",
                    help=f"Whether to print pareto fronts")

args = parser.parse_args()

# ==============================================================================

sns.set_style("darkgrid")
plt.rcParams['text.usetex'] = True
textwidth = 347.12354 / 72.27  # inches
matplotlib.rcParams.update({
    "font.size": 6,
    "lines.markersize": 4,  # 10
    "lines.markeredgewidth": .5,  # 1.0
    "lines.linewidth": 1  # 3
})

groups_color_palette = sns.color_palette()[:3]

# Have to specify backward so that CM0 gets the last and CM2 the first
subgroups_color_palette = [sns.color_palette()[0]] + [
    sns.color_palette(p)[i] for i in [1, 2] for p in ["dark", "deep", "bright"]
]

# ==============================================================================

runs = glob.glob("**/run-*/", root_dir=args.root, recursive=True)

# ==============================================================================

col_mapping = {}

trainer = col_mapping["trainer"] = "Trainer"
env = col_mapping["env"] = "Environment"
groups = col_mapping["groups"] = "Groups"
sub_groups = col_mapping["sub-groups"] = "Detailed groups"
archs = col_mapping["arch"] = "Controller"
sub_archs = col_mapping["sub-arch"] = "Architectures"
params = col_mapping["params"] = "Parameters"
reward = col_mapping["reward"] = "Reward"
normal_reward = col_mapping["normalized_reward"] = "Normalized Reward"
param_impact = col_mapping["pi"] = "Efficiency"
kernels_reward = col_mapping["kernels"] = "Kernels"
gym_reward = col_mapping["gym"] = "Gymnasium"
speed_reward = col_mapping["speed"] = "Speed"
instability_avg = col_mapping["instability_avg"] = "Instability (avg)"
instability_std = col_mapping["instability_std"] = "Instability (std)"


def efficiency(_r: Rewards): return f"$E_{_r[0].lower()}$"


col_mapping.update({f"pi_{r[0].lower()}": efficiency(r) for r in Rewards})


# ==============================================================================


str_root = str(args.root)
df_file = args.root.joinpath("summaries.csv")
if args.purge and df_file.exists():
    df_file.unlink()

showcase_folder = args.root.joinpath("_showcase")
if args.purge and showcase_folder.exists():
    shutil.rmtree(showcase_folder)

if df_file.exists():
    df = pd.read_csv(df_file, index_col=0)
    print("Loaded existing df:")
    print(df)

else:
    df = pd.concat(
        pd.read_csv(args.root.joinpath(r).joinpath("summary.csv"), index_col=0)
        for r in tqdm(runs, desc="Reading csvs")
    )

    df.index = df.index.map(lambda _p: _p.replace("/home/kgd/data", str(args.root.parent)))

    try:
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

    df["sub-arch"] = _arch = df.index.map(lambda _x: _x.split("/")[-2])
    df["groups"] = df.trainer + "-" + df["arch"]
    df["sub-groups"] = df.trainer + "-" + _arch.map(lambda _x: "-".join(_x.split("-")[:-1]))

    reward_to_enum_name = {
        v.name(): k.value
        for values in RewardToMonitor.values()
        for k, v in values.items()
    }

    df = df.rename(columns=reward_to_enum_name)
    df = df.T.groupby(by=df.columns).sum().transpose()
    rewards = [r.value for r in Rewards]

    for r in rewards:
        index = (df.reward == r)
        if sum(index) > 0:
            df.loc[index, normal_reward] = StandardScaler().fit_transform(np.array(df.loc[index, r]).reshape(-1, 1))
        else:
            df.loc[index, normal_reward] = np.nan

        df[f"pi_{r[0].lower()}"] = df["fitness"] / df["params"]

    print("Saving aggregated df:")
    print(df.columns)
    print(df)
    print("--------------")
    print()

    df.to_csv(df_file)

# ==============================================================================
#
# print(col_mapping)
# print(df.columns)

df.rename(inplace=True, columns=col_mapping)
df[reward] = df[reward].map(lambda _x: col_mapping[_x])


def pretty_format_groups(_x): return "".join([_t[0] for _t in _x.split("-")]).upper()


def pretty_format_arch(_x):
    tokens = _x.split("-")
    if len(tokens) > 2:
        s = f"^{{{tokens[1]}}}_{{{tokens[2]}}}"
    else:
        s = f"_{{{tokens[1]}}}"
    return f"{tokens[0][0]}${s}$"



pretty_groups = f"Arch + Trainer"
df[pretty_groups] = df[groups].map(pretty_format_groups)

pretty_sub_groups = f"Detailed Arch + Trainer"
df[pretty_sub_groups] = df[sub_groups].map(pretty_format_groups)

pretty_sub_archs = f"pretty_{sub_archs}"
df[pretty_sub_archs] = df[sub_archs].map(pretty_format_arch)

print(df.columns)

# ==============================================================================


df_gb_dg = df.groupby([groups, reward])

if args.distance_filtering > 0:
    filtered_df = df[df["dX"] >= .15]
    filtered_df_gb_dg = filtered_df.groupby([groups, reward])
    print("Filtered data:")
    print(pd.DataFrame({"Original": df_gb_dg.size(),
                       "Filtered": filtered_df_gb_dg.size(),
                        "Kept": 100 * filtered_df_gb_dg.size() / df_gb_dg.size()}))
else:
    filtered_df = df


def _groups(_detailed: bool): return pretty_sub_groups if _detailed else pretty_groups
def hue_order(_detailed): return sub_groups_list if _detailed else groups_list


groups_list = sorted(df[_groups(_detailed=False)].unique().tolist())
sub_groups_list = sorted(df[_groups(_detailed=True)].unique().tolist(),
                         key=lambda _x: (_x[:2], 0 if len(_x) < 3 else -int(_x[-1])))
legend_sub_groups_list = sorted(sub_groups_list)

envs = df[env].unique().tolist()

rewards = df[reward].unique().tolist()
rewards_hue_order = sorted(df[reward].unique().tolist())

parameter_count = {
    (t, s): p.item() for (t, s), p in
    df.groupby([trainer, sub_archs])[params].unique().items()
}

print("===")
print("=== \"Fixing\" efficiency metrics")
for r in rewards:
    index = (df[reward] == r)
    scaler = StandardScaler().fit(np.array(df.loc[index, r]).reshape(-1, 1))
    df.loc[index, normal_reward] = scaler.transform(np.array(df.loc[index, r]).reshape(-1, 1))
    df[efficiency(r)] = scaler.transform(np.array(df[r]).reshape(-1, 1)).reshape(-1) / np.log10(df[params])
print("===")
print("===")

# print()
# print(df.groupby(df.index.map(lambda _s: "/".join(_s.split('/')[1:4]))).size().to_string(max_rows=1000))
# print()

# ==============================================================================
# TABLES


def df_for_reward(_reward) -> pd.DataFrame: return df.loc[df[reward] == _reward, :]


median_champ = "Median"
median_champ_arch = "MChampArch"


def best_median_performance(_r, _x):
    return df.loc[_x.index, :].groupby(sub_archs)[_r].median().sort_values().tail(1)


def best_median_architecture(_r, _x):
    return best_median_performance(_r, _x).index


def make_summary_pivot(_r: Rewards, col: str):
    return pd.pivot_table(
        df_for_reward(_r),
        values=[col],
        index=groups,
        aggfunc={col: [
            ("Max", "max"),
            ("", lambda _x: df.loc[_x.index, :].loc[(df[col] == _x.max()), sub_archs]),
            (median_champ, lambda _x: best_median_performance(col, _x)),
            (median_champ_arch, lambda _x: best_median_architecture(col, _x)),
        ]},
        sort=False,
    ).transpose()

print()
print("Summary table")
summary_pivot = pd.concat(
    [make_summary_pivot(r, r) for r in rewards]
    +
    [make_summary_pivot(r, efficiency(r)) for r in rewards]
)

summary_pivot = summary_pivot[sorted(summary_pivot.columns)]
summary_pivot.rename(index={median_champ_arch: "" for r in rewards}, level=1).to_latex(
    args.root.joinpath("summary.tex"),
    float_format=lambda _f: f"{_f:.2f}"
)
print(summary_pivot.to_string(float_format=lambda _f: f"{_f:.2f}"))

# exit(42)


# ==============================================================================

training_curves_file = args.root.joinpath("training_curves.pdf")
if False and args.plot_training_curves and not args.synthesis and (args.purge or not training_curves_file.exists()):
    print()
    cma_time = "evals"
    cma_value = "fitness"

    ppo_time = "time/total_timesteps"
    ppo_value = "eval/mean_reward"

    max_samples = 20
    clipped_range = (0, 10000)
    min_sample_period = (clipped_range[1] - clipped_range[0]) / max_samples

    ratios = []

    training_curves_data = args.root.joinpath("progress.csv")
    if args.purge or not training_curves_data.exists():
        dfs = []
        for f in tqdm(runs, desc="Extracting training curves"):
            f = args.root.joinpath(f)
            summary_file = f.joinpath("summary.csv")
            if not summary_file.exists():
                continue

            _env = f.parts[-5]
            _trainer = f.parts[-4]
            _reward = f.parts[-3]
            _subgroup = f.parts[-2] + "-" + _trainer
            _subgroup = _subgroup[:3] + _subgroup[4:5] + _subgroup[-4:]
            if "cpg" in _subgroup:
                _subgroup = _subgroup[:3] + _subgroup[4:]
            _group = _subgroup[:3] + _subgroup[-4:]

            # print(f"{f} {_trainer=} {_reward=}, {_group=}, {_subgroup=}")

            if (file := f.joinpath("progress.csv")).exists():
                # continue
                sub_df = pd.read_csv(file)
                sub_df = sub_df[[ppo_time, ppo_value]].dropna()
                sub_df[ppo_time] /= 200
                sub_df.rename(inplace=True, columns={
                   ppo_time: "time", ppo_value: _reward
                })

            elif (file := f.joinpath("xrecentbest.dat")).exists():
                header = open(file).readline()
                headers = header[header.find('"')+1: header.rfind('"')].split(', ')
                _cols = [1, 4]
                sub_df = pd.read_csv(file, sep=' ', usecols=_cols, header=None,
                                     skiprows=1,
                                     names=[headers[c] for c in _cols])#.dropna()
                sub_df = sub_df[[cma_time, cma_value]].dropna()
                sub_df.rename(inplace=True, columns={
                   cma_time: "time", cma_value: _reward
                })
                sub_df[_reward] *= -1

            else:
                raise RuntimeError(f"No run data for {f}")

            sub_df[_reward] /= 30
            if _reward == "kernels":
                sub_df[_reward] /= 20

            raw_reward = sub_df[_reward].max()
            expected_reward = pd.read_csv(summary_file)["fitness"].iloc[-1]
            ratios.append([
                f,
                _trainer, _reward, _group,
                raw_reward / expected_reward
            ])

            if len(sub_df) > max_samples:
                t = sub_df.time
                t = t.clip(0, 10000)
                t = min_sample_period * (t // min_sample_period).astype(np.int32)
                sub_df = sub_df.groupby(t).max().drop(columns="time").reset_index(names="time")

            sub_df["run"] = f
            sub_df["reward"] = _reward
            sub_df["groups"] = _group
            sub_df["sub-groups"] = _subgroup
            dfs.append(sub_df)

        t_dfs = pd.concat(dfs)

        # print(t_dfs)
        # print(t_dfs.columns)
        t_dfs = t_dfs[["run", "time", "reward", "kernels", "gym", "speed", "groups", "sub-groups"]]
        t_dfs.to_csv(training_curves_data)

        rdf = pd.DataFrame(ratios, columns=["Run", "Trainer", "Reward", "Group", "Ratio"])
        print("Saving aggregated training curves data")
        print(rdf.groupby(["Trainer", "Reward", "Group"])["Ratio"].agg(["mean", "std"]))
        print("--------------")
        print()

    else:
        t_dfs = pd.read_csv(training_curves_data)

    t_dfs.rename(inplace=True, columns=col_mapping)
    t_dfs[reward] = t_dfs[reward].map(lambda _x: col_mapping[_x])

    t_dfs.drop(columns=["run"], inplace=True)

    # print(t_dfs)

    # for r in rewards:
    #     df.loc[df[reward] != r, r] = float("nan")

    print()
    print(df.groupby([groups])[rewards].aggregate(["mean", "std"]))
    print()
    print(df.groupby([sub_groups])[rewards].aggregate(["mean", "std"]))
    print()

    with PdfPages(training_curves_file) as pdf:
        for _detailed in [False, True]:
            for r in rewards:
                print(f"Generating lineplot(time, {r})")
                g = sns.lineplot(
                    x="time", y=r, data=t_dfs[t_dfs[reward] == r],
                    hue=_groups(_detailed), hue_order=hue_order(_detailed),
                )
                g.set_xlabel("Episodes")
                g.set_ylabel(r)
                pdf.savefig(g.figure, bbox_inches="tight")
                plt.close("all")

# =============================================================================


def showcase(_p, _out, _prefix=None):
    def name(_src: Path):
        _suffix = _src.suffix
        if not _src.is_dir():
            _src = _src.parent
        _basename = str(_src).replace(str_root + "/", "").replace("/", "_")
        if _prefix is not None:
            _basename = _prefix + "_" + _basename
        return _out.joinpath(Path(_basename).with_suffix(_suffix))

    def cp(_src):
        _suffix = _src.suffix
        _dst = name(_src).with_suffix(_suffix)
        print(_src, "->", _dst)
        shutil.copyfile(_src, _dst)

    def ln(_src):
        _dst: Path = name(_src)
        _src = Path(_src).relative_to(_dst.parent, walk_up=True)
        print(_dst, "~>", _src)
        _dst.symlink_to(_src, target_is_directory=True)

    _p = Path(str(_p))
    cp(_p.joinpath("champion.mp4"))
    cp(_p.joinpath("champion.zip"))
    ln(_p)


columns = [reward, archs, sub_archs, "neighborhood", "width", "depth", params,
           kernels_reward, speed_reward, gym_reward, normal_reward,
           groups, sub_groups, pretty_groups, pretty_sub_groups]

for e in envs:
    _df = df[df[env] == e]
    for _g, _name in [(groups, []), (sub_groups, ["detailed"])]:
        champs = _df.loc[pd.concat(
            _df[_df[reward] == r].groupby(_g, dropna=False)[r].idxmax()
            for r in rewards
        )][columns]
        champs["SUM"] = champs[speed_reward] + champs[kernels_reward]
        champs.sort_values(inplace=True, by="SUM", ascending=False)
        if args.print_paretos:
            print()
            print(" ".join(["Bests"] + [f"({_x})" for _x in _name]))
            print(champs)
            print()

        best_folder = showcase_folder.joinpath("_".join(["bests"]+_name))
        if args.purge:
            shutil.rmtree(best_folder, ignore_errors=True)
        if not best_folder.exists():
            best_folder.mkdir(parents=True)
            for p in champs.index:
                showcase(p, best_folder)
            print()


def _pareto(_points):
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
    return sorted(is_efficient, key=lambda i: math.atan2(_original_points[i][1], _original_points[i][0]))


def _showcase_pareto(__pareto_front, __name, __label):
    if args.print_paretos:
        print(__label)
        print(__pareto_front)
        print()
        print(__pareto_front.groupby([reward]).size())
        print()
        print(__pareto_front.groupby([sub_groups]).size())
        print()
        print(__pareto_front.groupby([reward, sub_groups]).size())
        print()

    __folder = showcase_folder.joinpath("pareto").joinpath(__name)
    if args.purge:
        shutil.rmtree(__folder, ignore_errors=True)
    if not __folder.exists():
        __folder.mkdir(parents=True)
        _n = len(__pareto_front)
        _digits = math.ceil(math.log10(_n))
        for i, __p in enumerate(__pareto_front.index):
            showcase(__p, __folder, f"{i:0{_digits}d}")
        print()


pp_pareto = _pareto(np.array([(a, b) for a, b in zip(df[params], df[normal_reward])]))
pp_pareto = df.iloc[pp_pareto][columns]
_showcase_pareto(pp_pareto, "parameters_performance", "Speed vs Stability pareto front")


ss_pareto = _pareto(np.array([(a, b) for a, b in zip(df[speed_reward], df[kernels_reward])]))
ss_pareto = df.iloc[ss_pareto][columns]
_showcase_pareto(ss_pareto, "speed_stability", "Speed vs Stability pareto front")


pe_pareto = df.iloc[_pareto(np.array([(a, b) for a, b in
                                      zip(1 - df["avg_d_o"], df[normal_reward])]))]
_showcase_pareto(pe_pareto, "performance_energy", "Performance vs energy pareto front")


zz_pareto = df.iloc[_pareto(np.array([(a, b) for a, b in
                                      zip(df["avg_z"], 1 - df["std_z"])]))]
_showcase_pareto(zz_pareto, "height_jumpiness", "Elevation vs jumpiness pareto front")


# ==============================================================================

trajectories_file = args.root.joinpath("trajectories.pdf")
if False and args.plot_trajectories and not args.synthesis and (args.purge or not trajectories_file.exists()):
    with PdfPages(trajectories_file) as summary_pdf:
        fig, ax = plt.subplots()
        tdfs_trajs = {}

        sns_cp = sns.color_palette()
        for f in tqdm(
                glob.glob("[!_]*/**/champion.trajectory.csv", root_dir=args.root, recursive=True),
                desc="Extracting trajectories"):
            f = args.root.joinpath(f)
            tdf = pd.read_csv(f, index_col=0)
            run = str(f.parent)
            data = df.loc[run, :]
            sns.lineplot(data=tdf, x="x", y="y", ax=ax,
                         color=sns_cp[groups_list.index(data[groups])],
                         zorder=-tdf.x.iloc[-1], lw=.1)

            tdfs_trajs[run] = tdf

        ax.legend(handles=[
            Line2D([0], [0], color=sns_cp[i], label=groups_list[i])
            for i in range(len(groups_list))
        ])

        summary_pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots()
        for r, tdf in tdfs_trajs.items():
            data = df.loc[r, :]
            sns.lineplot(data=tdf, x="x", y="y", ax=ax,
                         color=sns_cp[rewards_hue_order.index(data[reward])],
                         zorder=-tdf.x.iloc[-1], lw=.1)

        ax.legend(handles=[
            Line2D([0], [0], color=sns_cp[i], label=rewards_hue_order[i])
            for i in range(len(rewards_hue_order))
        ])

        summary_pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots()
        sns_cp = sns.color_palette(n_colors=len(champs))
        for r in champs.index:
            sns.lineplot(data=tdfs_trajs[r], x="x", y="y", ax=ax,)

        summary_pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots()
        sns_cp = sns.color_palette(n_colors=len(champs))
        for r in champs.index:
            sns.lineplot(data=tdfs_trajs[r], x="t", y="z", ax=ax,)

        summary_pdf.savefig(fig, bbox_inches="tight")
        plt.close()

# ==============================================================================

violinplot_common_args = dict(
    inner="box", cut=0, gap=.25,
    common_norm=True, density_norm="width"
)

# ==============================================================================


def __key(*__args): return "_".join([str(x) for x in __args])


synthesis = defaultdict(list)
synthesis.update({
    sns.relplot: [
        __key(params, r, True) for r in rewards
    ] + [
        __key(params, efficiency(r), True) for r in rewards
    ],
    sns.violinplot: [
        # __key(_groups(_detailed=False), r, False) for r in rewards
    ]
})
print(synthesis)


def is_synthesis(fn, _args):
    r = __key(*_args) in synthesis.get(fn, [])
    print(__key(*_args), "->", r)
    return r


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
        _g.suptitle(title)
    if not args.synthesis:
        summary_pdf.savefig(_g, bbox_inches="tight")
    plt.close()


def save_legend(_legend: Legend, order: list[str]):
    _fig, _ = plt.subplots(frameon=False)
    _handles = {_h.get_label(): _h for _h in _legend.legend_handles}
    standalone_legend = _fig.legend(
        handles=[_handles[h] for h in order],
        title=_legend.get_title().get_text(),
        ncols=len(hue_order(True)))
    bbox = standalone_legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array([-5, -5, +5, +5])))
    bbox = bbox.transformed(_fig.dpi_scale_trans.inverted())
    synthesis_pdf.savefig(_fig, bbox_inches=bbox)
    plt.close()

    return _legend


pdf_summary_file = args.root.joinpath(".summary.pdf")
pdf_synthesis_file = args.root.joinpath(".synthesis.pdf")
print("Plotting...")
with PdfPages(pdf_summary_file) as summary_pdf, PdfPages(pdf_synthesis_file) as synthesis_pdf:
    # g = sns.scatterplot(df, x=params, y="depth")
    # plt.xscale('log', base=10)
    # pdf.savefig(g.figure, bbox_inches="tight")
    # plt.close()
    #
    # g = sns.scatterplot(df, x="width", y="depth")
    # plt.xscale('log', base=2)
    # pdf.savefig(g.figure, bbox_inches="tight")
    # plt.close()

    annotator_args = dict(
        test="Mann-Whitney", verbose=0, loc="outside",
        hide_non_significant=False, text_format="star",
        comparisons_correction="bonferroni"
    )

    replot_args = dict(
        hue=_groups(True), hue_order=hue_order(True),
        col=_groups(False), col_order=hue_order(False),
        kind='line', marker='o',
        err_style="band", errorbar=("pi", 100), estimator="median",
        palette=subgroups_color_palette,
        facet_kws=dict(),
    )

    # KEEP: Relation plot parameters <-> performance
    legend = None
    for r in rewards:
        x, y, detailed = params, r, True
        if args.synthesis and is_synthesis(sns.relplot, (x, y, detailed)):
            replot_args["x"], replot_args["y"] = x, y

            g = sns.relplot(df[(df[env] == "ariel") & (df[reward] == r)], **replot_args)
            g.axes.flatten()[0].set_xscale('log', base=10)
            g.set_xlabels("Number of parameters")
            g.set_ylabels("Objective function")

            # Get best performing architecture and number of parameters
            perf_champ_params = {
                trainer: (champ_arch, parameter_count[(trainer.split("-")[0], champ_arch)])
                for trainer, champ_arch in
                summary_pivot.loc[(r, median_champ_arch), :].items()
            }
            for ax, (champ_trainer, (champ_arch, champ_params)), color in zip(
                    g.axes.flatten(), perf_champ_params.items(), groups_color_palette):
                perf = summary_pivot.loc[(r, "Median"), champ_trainer]
                line_args = dict(
                    color=color, linestyle="dotted",
                    linewidth=.5 * matplotlib.rcParams["lines.linewidth"],
                )
                text = ax.text(
                    champ_params, 1, pretty_format_arch(champ_arch),
                    horizontalalignment="center",
                    verticalalignment="top",
                    transform=transforms.blended_transform_factory(
                        ax.transData, ax.transAxes),
                    bbox=dict(facecolor='white', edgecolor='black',
                              boxstyle='round,pad=.2'),
                )
                ax.axhline(y=perf, **line_args)
                ax.axvline(
                    x=champ_params,
                    **line_args)
                # ax.text()

            if legend is None:
                legend = save_legend(g.legend, order=legend_sub_groups_list)
            g.legend.set_visible(False)

            maybe_save(g, True,
                       title="Direct performance by controller architecture and trainer",
                       cols=1, ratio=1/3)

    # KEEP: Relation plot parameters <-> parameter impact
    legend = None
    for r in rewards:
        x, y, detailed = params, efficiency(r), True
        if args.synthesis and is_synthesis(sns.relplot, (x, y, detailed)):
            replot_args["x"], replot_args["y"] = x, y

            g = sns.relplot(df[(df[env] == "ariel") & (df[reward] == r)], **replot_args)
            g.axes.flatten()[0].set_xscale('log', base=10)
            g.set_xlabels("Number of parameters")
            g.set_ylabels(f"Efficiency {y}")

            # Get best performing architecture and number of parameters
            perf_champ_params = {
                trainer: (champ_arch, parameter_count[(trainer.split("-")[0], champ_arch)])
                for trainer, champ_arch in
                summary_pivot.loc[(y, median_champ_arch), :].items()
            }
            for ax, (champ_trainer, (champ_arch, champ_params)), color in zip(
                    g.axes.flatten(), perf_champ_params.items(), groups_color_palette):
                perf = summary_pivot.loc[(y, "Median"), champ_trainer]
                line_args = dict(
                    color=color, linestyle="dotted",
                    linewidth=.5 * matplotlib.rcParams["lines.linewidth"],
                )
                text = ax.text(
                    champ_params, 1, pretty_format_arch(champ_arch),
                    horizontalalignment="center",
                    verticalalignment="top",
                    transform=transforms.blended_transform_factory(
                        ax.transData, ax.transAxes),
                    bbox=dict(facecolor='white', edgecolor='black',
                              boxstyle='round,pad=.2'),
                )
                ax.axhline(y=perf, **line_args)
                ax.axvline(
                    x=champ_params,
                    **line_args)
                # ax.text()

            if legend is None:
                legend = save_legend(g.legend, order=legend_sub_groups_list)
            g.legend.set_visible(False)

            maybe_save(g, True,
                       title="Direct performance by controller architecture and trainer",
                       cols=1, ratio=1/3)

    if args.plot_perf_violins:
        group_pairs = list(itertools.combinations(groups_list, 2))
        print(group_pairs)

        # == This we keep: it shows the raw performance
        for e in envs:
            for r in rewards:
                detailed = False
                x, y = _groups(detailed), r
                if not args.synthesis or is_synthesis(sns.violinplot, (x, y, detailed)):
                    _df = df[(df[env] == e) & (df[reward] == r)]
                    violinplot_args = dict(
                        data=_df, x=x, y=y,
                        hue=_groups(detailed), hue_order=hue_order(detailed),
                        order=hue_order(detailed),
                        **violinplot_common_args
                    )

                    g = sns.violinplot(**violinplot_args)
                    g.set_ylabel(f"{r}")

                    if not detailed:
                        annotator = Annotator(
                            ax=g.axes, pairs=group_pairs, plot='violinplot',
                            **violinplot_args)
                        annotator.configure(
                            test="Mann-Whitney", verbose=0, loc="outside",
                            hide_non_significant=False, text_format="star",
                            comparisons_correction="bonferroni")
                        _, corrected_results = annotator.apply_and_annotate()

                    maybe_save(g, not detailed, title=f"Global performance on {r} in {e}", cols=3, ratio=2)

        # == KEEP: efficiency (but only after extracting champions?)
        # for e in envs:
        #     for r in rewards:
        #         _df = df[(df[env] == e) & (df[reward] == r)]
        #         for detailed in [False, True]:
        #             violinplot_args = dict(
        #                 data=_df, x=_groups(detailed), y=param_impact,
        #                 hue=_groups(detailed), hue_order=hue_order(detailed),
        #                 order=hue_order(detailed),
        #                 **violinplot_common_args
        #             )
        #             g = sns.violinplot(**violinplot_args)
        #             g.set_ylabel(f"$E_{r[0].lower()}$")
        # 
        #             if not detailed:
        #                 annotator = Annotator(
        #                     ax=g.axes, pairs=group_pairs, plot='violinplot',
        #                     verbose=0,
        #                     **violinplot_args)
        #                 annotator.configure(
        #                     test="Mann-Whitney", verbose=0, loc="outside",
        #                     hide_non_significant=False, text_format="star",
        #                     comparisons_correction="bonferroni")
        #                 _, corrected_results = annotator.apply_and_annotate()
        # 
        #             maybe_save(g, not detailed, title=f"Global efficiency on {r} in {e}", cols=3, ratio=2)

    def relplot(_e, _x, _y, base=10, **kwargs):
        _detailed = kwargs.pop("detailed", False)
        if (_isS := kwargs.pop("synthesis", None)) is None:
            _isS = is_synthesis(sns.relplot, (x, y, _detailed))
        if not args.synthesis or _isS:
            _args = dict(x=_x, y=_y,
                         hue=_groups(_detailed), hue_order=hue_order(_detailed),
                         col=reward,
                         kind='line', marker='o',
                         err_style="bars", errorbar="ci", estimator="median")

            _args.update(kwargs)
            g = sns.relplot(df[df[env] == _e], **_args)
            if len(envs) > 1:
                g.legend.set_title(f"env = {_e}")
            plt.xscale('log', base=base)

            try:
                _i = rewards.index(_y)
                for _j, _ax in enumerate(g.axes.flatten()):
                    if _i != _j:
                        _ax.add_patch(plt.Rectangle(
                            (0, 0), 1, 1,
                            fc="#FFFFFF7F",
                            zorder=10,
                            transform=_ax.transAxes),
                        )
            except ValueError:
                pass

            maybe_save(g, _isS, title=f"relplot({_e=}, {_x=}, {_y=})")

    pareto_args = dict(
        linestyle='dashed', color="red", lw=.5,
        marker='D', markeredgecolor='k', markersize=7,
        label="Pareto front", zorder=5
    )

    if args.plot_paretos:
        for detailed in [False, True]:
            scatter_args = dict(
                style=reward, style_order=rewards_hue_order,
                hue=_groups(detailed), hue_order=hue_order(detailed),
                legend=False,
            )
            kde_args = {
                k: scatter_args[k] for k in ["hue", "hue_order", "legend"]
            } | dict(cut=0)

            for x, y, pareto_df, log_x in [
                # Normalized performance vs parameters + pareto front
                (params, normal_reward, pp_pareto, True),
                # Speed vs kernels + pareto front
                (speed_reward, kernels_reward, ss_pareto, False),
                # = Energy vs performance + pareto front
                ("avg_d_o", normal_reward, pe_pareto, False),
                # = Elevation vs jumpiness + pareto front
                ("avg_z", "std_z", zz_pareto, False),
            ]:
                print(x, y)

                isS = is_synthesis(sns.scatterplot, (x, y, detailed))
                if not args.synthesis or isS:
                    # g = sns.scatterplot(df, **scatter_args)
                    # if log_x:
                    #     plt.xscale('log', base=10)
                    # g.axes.plot(pareto_df[x], pareto_df[y], **pareto_args)
                    # sns.scatterplot(pareto_df, **scatter_args, ax=g.axes, zorder=10, legend=False)

                    g = sns.JointGrid()
                    sns.scatterplot(df, x=x, y=y, **scatter_args | dict(legend=True), ax=g.ax_joint)
                    sns.kdeplot(df, x=x, ax=g.ax_marg_x, **kde_args)
                    sns.kdeplot(df, y=y, ax=g.ax_marg_y, **kde_args)
                    if log_x:
                        g.ax_joint.set_xscale('log', base=10)
                    sns.lineplot(data=pareto_df, x=x, y=y, **pareto_args, ax=g.ax_joint)
                    sns.scatterplot(pareto_df, x=x, y=y, **scatter_args,
                                    ax=g.ax_joint, zorder=10)

                    g.ax_joint.legend().remove()
                    g.figure.legend(loc='center left', bbox_to_anchor=(1, .5))
                    maybe_save(g, isS, title=f"Pareto front {x} vs {y}")

            # ====

    if args.plot_relations:
        for e in envs:
            for detailed in [False, True]:
                for k in rewards:
                    relplot(_e=e, _x=params, _y=k, detailed=detailed, synthesis=False)
                    relplot(_e=e, _x=params, _y=k, errorbar=("pi", 100),
                            err_style="band", detailed=detailed, synthesis=False)
        for e in envs:
            for detailed in [False, True]:
                relplot(_e=e, _x=params, _y="wall-time", detailed=detailed, synthesis=False)
    print()

    tested_pairs = [
        # ((a, b), (c, d)) for ((a, b), (c, d)) in
        ((b, a), (d, c)) for ((a, b), (c, d)) in
        itertools.combinations(itertools.product(df[groups].unique(), df[reward].unique()), r=2)
        if a == c or b == d
    ]
    tests = pd.DataFrame(index=tested_pairs, columns=[])
    if len(synthesis[sns.violinplot]) > 0:
        print("Testing", len(tested_pairs), "pairs:", tested_pairs)
    annotator = None

    if args.plot_all_violins:
        # for c in ["avg_d_o", "Vx", "Vy", "Vz", "z", "dX", "dY", "cX", "cY"] + [c for c in df.columns if ("avg" in c or "std" in c)]:
        for c in rewards + [c for c in df.columns if ("avg" in c or "std" in c)]:
            for e in envs:
                relplot(_e=e, _x=params, _y=c)

            # ===
            isS = is_synthesis(sns.violinplot, (c,))
            if not args.synthesis or isS:
                violinplot_args = dict(
                    data=df, x=reward, y=c, hue=groups,
                    **violinplot_common_args
                )

                ax = sns.violinplot(**violinplot_args)

                # print(c)
                # annotator = Annotator(ax=ax, pairs=tested_pairs, plot='violinplot', **violinplot_args)
                # annotator.configure(test="Mann-Whitney", verbose=2,
                #                     hide_non_significant=True, text_format="simple",
                #                     comparisons_correction="bonferroni")
                # _, corrected_results = annotator.apply_and_annotate()
                # tests[c] = [r.data.pvalue for r in corrected_results]
                # print()
                maybe_save(ax, isS, title=f"Distribution of {c} grouped by training reward")

                violinplot_args["x"], violinplot_args["hue"] = violinplot_args["hue"], violinplot_args["x"]
                ax = sns.violinplot(**violinplot_args)

                maybe_save(ax, isS, title=f"Distribution of {c} grouped by group")

            # # ===
            # fig, ax = plt.subplots()
            # sub_violinplots_args = violinplot_args.copy()
            # sub_violinplots_args["inner"] = None
            # handles, labels = [], []
            # for (hue, group), color in zip(
            #         detailed_groups.groupby(detailed_groups, sort=False),
            #         sns.color_palette(None, detailed_groups.unique().size)):
            #     sub_violinplots_args["data"] = df.loc[group.index]
            #     sub_violinplots_args["x"] = groups[group.index]
            #     sub_violinplots_args["palette"] = (color, .5 + .5 * np.array(color))
            #     sns.violinplot(**sub_violinplots_args, alpha=.25, legend=False)
            #     handles.append(plt.Rectangle((0, 0), 0, 0,
            #                                  fc=color))
            #     labels.append(hue)
            #
            # fig.legend(
            #     labels=labels, handles=handles,
            #     title="arch-detailed",
            # )
            #
            # pdf.savefig(fig, bbox_inches="tight")
            # plt.close()
            # # ===

    if tests.size > 0:
        tests = tests.map(lambda _x: _x if x <= 0.05 else np.nan)
        g = sns.heatmap(tests, annot=True, cmap="magma_r", fmt=".2g", annot_kws=dict(size=3), norm=LogNorm())
        summary_pdf.savefig(g.figure, bbox_inches="tight")
        plt.close()

    def _process(_path):
        try:
            _path = Path(_path)
            _df = pd.read_csv(_path.joinpath("motors.csv"))
            _df["Path"] = _path
            _df["Valid"] = _df.f.map(lambda x: .1 <= x <= 10)
            return _df
        except FileNotFoundError:
            return None

for file in [pdf_summary_file, pdf_synthesis_file]:
    if file.exists():
        new_path = file.parent.joinpath(file.name[1:])
        file.rename(new_path)
        print("Generated", new_path)
