import argparse
import glob
import itertools
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from pandas import Series
from sklearn.preprocessing import StandardScaler
from tqdm.rich import tqdm

from aapets.common.misc.debug import kgd_debug
from aapets.cpg_rl.types import RewardToMonitor, Rewards

matplotlib.use("agg")

parser = argparse.ArgumentParser("Summarizes summary.csv files")
parser.add_argument("root", type=Path)
parser.add_argument("--purge", default=False, action="store_true", help="Purge old showcased files")
parser.add_argument("--synthesis", default=False, action="store_true", help="Only produce synthesis plots")
parser.add_argument("--pretty-columns", default=False, action="store_true",
                    help="Rename columns for publication")
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
runs = glob.glob("**/run-*/", root_dir=args.root, recursive=True)

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
        pd.read_csv(f, index_col=0)
        for f in args.root.glob("**/summary.csv")
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

    def _make_groups(_overview=True):
        return Series(name="arch" + ("" if _overview else "-detailed"),
                      data=(
                              df.arch
                              + ("" if _overview else df.depth.map(lambda f: str(int(f)) if not np.isnan(f) else ""))
                              + "-" + df.index.map(lambda _s: _s.split("/")[-4])
                      ))

    df["groups"] = _make_groups(_overview=True)
    df["detailed-groups"] = _make_groups(_overview=False)

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
            df.loc[index, "normalized_reward"] = StandardScaler().fit_transform(np.array(df.loc[index, r]).reshape(-1, 1))
        else:
            df.loc[index, "normalized_reward"] = np.nan

    df["pi"] = df["fitness"] / df["params"]
    df["pi_"] = df["normalized_reward"] / df["params"]

    print("Saving aggregated df:")
    print(df.columns)
    print(df)
    print("--------------")
    print()

    df.to_csv(df_file)


# ==============================================================================

col_mapping = {}
if args.pretty_columns:
    groups = col_mapping["groups"] = "Groups"
    all_groups = col_mapping["detailed-groups"] = "Detailed groups"
    params = col_mapping["params"] = "Parameters"
    reward = col_mapping["reward"] = "Reward"
    normal_reward = col_mapping["normalized_reward"] = "Normalized Reward"
    kernels_reward = col_mapping["kernels"] = "Gaussians"
    gym_reward = col_mapping["lazy"] = "Gym-Ant"
    speed_reward = col_mapping["speed"] = "Speed"
    col_mapping["distance"] = col_mapping["speed"]
    instability_avg = col_mapping["instability_avg"] = "Instability (avg)"
    instability_std = col_mapping["instability_std"] = "Instability (std)"
    df.rename(inplace=True, columns=col_mapping)

    df[reward] = df[reward].map(lambda _x: col_mapping[_x])

else:
    groups = "groups"
    all_groups = "detailed-groups"
    params = "params"
    env = "env"
    reward = "reward"
    normal_reward = "normalized_reward"
    kernels_reward = "kernels"
    gym_reward = "gym"
    speed_reward = "speed"
    instability_avg = "instability_avg"
    instability_std = "instability_std"


df_gb_dg = df.groupby([all_groups, reward])

if args.distance_filtering > 0:
    filtered_df = df[df["dX"] >= .15]
    filtered_df_gb_dg = filtered_df.groupby([all_groups, reward])
    print("Filtered data:")
    print(pd.DataFrame({"Original": df_gb_dg.size(),
                       "Filtered": filtered_df_gb_dg.size(),
                        "Kept": 100 * filtered_df_gb_dg.size() / df_gb_dg.size()}))
else:
    filtered_df = df


def _groups(_detailed): return all_groups if _detailed else groups


groups_hue_order = sorted(df[groups].unique().tolist())
detailed_groups_hue_order = sorted(df[all_groups].unique().tolist())

envs = df[env].unique().tolist()

rewards = df[reward].unique().tolist()
rewards_hue_order = sorted(df[reward].unique().tolist())


def hue_order(_detailed):
    return detailed_groups_hue_order if _detailed else groups_hue_order


# print()
# print(df.groupby(df.index.map(lambda _s: "/".join(_s.split('/')[1:4]))).size().to_string(max_rows=1000))
# print()

# ==============================================================================

training_curves_file = args.root.joinpath("training_curves.pdf")
if args.plot_training_curves and not args.synthesis and (args.purge or not training_curves_file.exists()):
    print()
    print("Extracting training curves")
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
                cols = [1, 4]
                sub_df = pd.read_csv(file, sep=' ', usecols=cols, header=None,
                                     skiprows=1,
                                     names=[headers[c] for c in cols])#.dropna()
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
            sub_df["detailed-groups"] = _subgroup
            dfs.append(sub_df)

        t_dfs = pd.concat(dfs)
        # print(t_dfs)
        # print(t_dfs.columns)
        t_dfs = t_dfs[["run", "time", "reward", *rewards, "groups", "detailed-groups"]]
        t_dfs.to_csv(training_curves_data)

        rdf = pd.DataFrame(ratios, columns=["Run", "Trainer", "Reward", "Group", "Ratio"])
        print("Saving aggregated training curves data")
        print(rdf.groupby(["Trainer", "Reward", "Group"])["Ratio"].agg(["mean", "std"]))
        print("--------------")
        print()

    else:
        t_dfs = pd.read_csv(training_curves_data)

    if args.pretty_columns:
        t_dfs.rename(inplace=True, columns=col_mapping)
        t_dfs[reward] = t_dfs[reward].map(lambda _x: col_mapping[_x])

    t_dfs.drop(columns=["run"], inplace=True)

    # print(t_dfs)

    # for r in rewards:
    #     df.loc[df[reward] != r, r] = float("nan")

    print()
    print(df.groupby([groups])[rewards].aggregate(["mean", "std"]))
    print()
    print(df.groupby([all_groups])[rewards].aggregate(["mean", "std"]))
    print()

    with PdfPages(training_curves_file) as pdf:
        for _detailed in [False, True]:
            for r in rewards:
                print(f"Generating lineplot(time, {r})")
                g = sns.lineplot(
                    x="time", y=r, data=t_dfs[t_dfs.reward == r],
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


columns = [reward, "arch", "neighborhood", "width", "depth", params,
           kernels_reward, speed_reward, gym_reward, normal_reward,
           groups, all_groups]

for e in envs:
    _df = df[df.env == e]
    for _g, _name in [(groups, []), (all_groups, ["detailed"])]:
        print()
        print(" ".join(["Bests"] + [f"({_x})" for _x in _name]))
        champs = _df.loc[pd.concat(
            _df[_df[reward] == r].groupby(_g, dropna=False)[r].idxmax()
            for r in rewards
        )][columns]
        champs["SUM"] = champs[speed_reward] + champs[kernels_reward]
        champs.sort_values(inplace=True, by="SUM", ascending=False)
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


def _showcase_pareto(__pareto_front, __name):
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

if args.print_paretos:
    print("Speed vs Stability pareto front")
    print(pp_pareto)
    print()
    print(pp_pareto.groupby([reward]).size())
    print()
    print(pp_pareto.groupby([all_groups]).size())
    print()
    print(pp_pareto.groupby([reward, all_groups]).size())
    print()
_showcase_pareto(pp_pareto, "parameters_performance")


ss_pareto = _pareto(np.array([(a, b) for a, b in zip(df[speed_reward], df[kernels_reward])]))
ss_pareto = df.iloc[ss_pareto][columns]

if args.print_paretos:
    print("Speed vs Stability pareto front")
    print(ss_pareto)
    print()
    print(ss_pareto.groupby([reward]).size())
    print()
    print(ss_pareto.groupby([all_groups]).size())
    print()
    print(ss_pareto.groupby([reward, all_groups]).size())
    print()
_showcase_pareto(ss_pareto, "speed_stability")


pe_pareto = df.iloc[_pareto(np.array([(a, b) for a, b in
                                      zip(1 - df["avg_d_o"], df[normal_reward])]))]

if args.print_paretos:
    print("Performance vs energy pareto front")
    print(pe_pareto)
    print("Pareto front:")
    print()
    print(pe_pareto.groupby([reward]).size())
    print()
    print(pe_pareto.groupby([all_groups]).size())
    print()
    print(pe_pareto.groupby([reward, all_groups]).size())
    print()
_showcase_pareto(pe_pareto, "performance_energy")


zz_pareto = df.iloc[_pareto(np.array([(a, b) for a, b in
                                      zip(df["avg_z"], 1 - df["std_z"])]))]
if args.print_paretos:
    print("Elevation vs jumpiness pareto front")
    print(zz_pareto)
    print("Pareto front:")
    print()
    print(zz_pareto.groupby([reward]).size())
    print()
    print(zz_pareto.groupby([all_groups]).size())
    print()
    print(zz_pareto.groupby([reward, all_groups]).size())
    print()
_showcase_pareto(zz_pareto, "height_jumpiness")


# ==============================================================================

trajectories_file = args.root.joinpath("trajectories.pdf")
if args.plot_trajectories and not args.synthesis and (args.purge or not trajectories_file.exists()):
    with PdfPages(trajectories_file) as summary_pdf:
        fig, ax = plt.subplots()
        tdfs_trajs = {}

        sns_cp = sns.color_palette()
        for f in tqdm(
                glob.glob("[!_]*/**/champion.trajectory.csv", root_dir=args.root, recursive=True),
                desc="Processing"):
            f = args.root.joinpath(f)
            tdf = pd.read_csv(f, index_col=0)
            run = str(f.parent)
            data = df.loc[run, :]
            sns.lineplot(data=tdf, x="x", y="y", ax=ax,
                         color=sns_cp[groups_hue_order.index(data[groups])],
                         zorder=-tdf.x.iloc[-1], lw=.1)

            tdfs_trajs[run] = tdf

        ax.legend(handles=[
            Line2D([0], [0], color=sns_cp[i], label=groups_hue_order[i])
            for i in range(len(groups_hue_order))
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
# TABLES

print(df.groupby(groups)[params].median())
# exit(42)

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
        __key(params, r, False) for r in rewards
    ]
})
print(synthesis)


def is_synthesis(fn, _args):
    r = __key(*_args) in synthesis.get(fn, [])
    print(__key(*_args), "->", r)
    return r


def maybe_save(_g, _is_synthesis):
    if not args.synthesis:
        summary_pdf.savefig(_g.figure, bbox_inches="tight")
    if _is_synthesis:
        synthesis_pdf.savefig(_g.figure, bbox_inches="tight")
    plt.close()


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

    if args.plot_perf_violins:
        for e in envs:
            for r in rewards:
                _df = df[(df[env] == e) & (df[reward] == r)]
                for detailed in [False, True]:
                    g = sns.violinplot(
                        data=_df, x=_df[_groups(detailed)], y=_df["pi"],
                        hue=_groups(detailed), hue_order=hue_order(detailed),
                        order=hue_order(detailed),
                        **violinplot_common_args
                    )
                    g.set_ylabel(f"{r} / {params}")
                    g.set_title(f"Performance / Parameters ({e})")
                    maybe_save(g, not detailed)
            g = sns.violinplot(
                data=_df, x=_df[groups], y=_df["pi_"],
                hue=_groups(detailed), hue_order=hue_order(detailed),
                col="reward",
                order=hue_order(detailed),
                **violinplot_common_args
            )
            g.set_ylabel(f"{r} / {params}")
            g.set_title(f"Performance / Parameters ({e})")
            maybe_save(g, not detailed)


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
            g.legend.set_title(f"env = {_e}")
            plt.xscale('log', base=base)
            maybe_save(g, _isS)

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
                    maybe_save(g, isS)

            # ====

    if args.plot_relations:
        for e in envs:
            for detailed in [False, True]:
                for k in rewards:
                    relplot(_e=e, _x=params, _y=k, detailed=detailed, synthesis=not detailed)
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
                maybe_save(ax, isS)

                violinplot_args["x"], violinplot_args["hue"] = violinplot_args["hue"], violinplot_args["x"]
                ax = sns.violinplot(**violinplot_args)

                maybe_save(ax, isS)

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
