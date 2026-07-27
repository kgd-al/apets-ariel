import dataclasses
import pathlib
import pickle
import typing

import ariel.body_phenotypes.robogen_lite.config
import ariel.body_phenotypes.robogen_lite.modules
import ariel.utils.renderers
import cycler
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import optuna
import optuna.visualization.matplotlib as vis_mpl
import scipy
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
import ariel.body_phenotypes.robogen_lite.modules.hinge as hinge_module

import aapets.common.config
import aapets.common.misc.config_base
import aapets.common.monitors.plotters.brain_activity
import aapets.common.mujoco.callback
import aapets.common.mujoco.state
import aapets.g_cpg.worlds

ORIGINAL_HINGE_KP = hinge_module.HINGE_KP
ORIGINAL_HINGE_KV = hinge_module.HINGE_KV


@dataclasses.dataclass
class CLAConfig(aapets.common.config.BaseConfig):
    datafile: typing.Annotated[pathlib.Path, "Location of ground-truth datafile", dict(required=True)] = None
    output: typing.Annotated[pathlib.Path, "Where to store the output"] = "./"
    render: typing.Annotated[bool, "Whether to render the robot used for simulation"] = False

    # Hardware was measured at 50Hz
    control_frequency: typing.Annotated[int, "How often to query the control for new outputs (Hz)"] = 50

    optuna: typing.Annotated[bool, "Whether to just look at values or use optuna to fine-tune"] = False


@dataclasses.dataclass
class TestConfig:
    config: CLAConfig

    hardware_data: dict[str, dict[str, float]]

def to_time(data): return np.array([i * 0.02 for i in range(len(data))])

def rmse(true, pred): return np.sqrt(np.mean((true - pred) ** 2))

def trim_initial_plateau(a, tol=1e-3, min_len=5):
    """Trim leading/trailing samples that stay within tol of the initial value."""
    moving = np.abs(np.gradient(a)) > tol

    # require min_run consecutive True to count as "started moving" (denoise)
    kernel = np.ones(min_len, dtype=int)
    run = np.convolve(moving, kernel, mode='valid') == min_len

    if not run.any():
        return a[:0]  # never leaves the plateau

    start = np.argmax(run)  # index into `moving`/`a` where the run begins
    return a[start:]

def make_simulation(kp: float = None, kv: float = None):
    core = CoreModule()

    hinge_module.HINGE_KP = kp or ORIGINAL_HINGE_KP
    hinge_module.HINGE_KV = kv or ORIGINAL_HINGE_KV
    hinge = HingeModule()
    hinge_module.HINGE_KP = ORIGINAL_HINGE_KP
    hinge_module.HINGE_KV = ORIGINAL_HINGE_KV

    core.sites[ariel.body_phenotypes.robogen_lite.config.ModuleFaces.FRONT].attach_body(hinge.body, prefix="C-")
    core.spec.body("core").quat = (np.cos(-np.pi / 4), 0, np.sin(-np.pi / 4), 0)
    world = aapets.g_cpg.worlds.default_world(core.spec, "calibrator")
    return aapets.common.mujoco.state.MjState.from_spec(world.spec)


def do_render():
    state = make_simulation()
    ariel.utils.renderers.single_frame_renderer(state.model, state.data,
                          width=640, height=480,
                          save=True, save_path="./calibrator.png")


def run_simulation(kp, kv, config: TestConfig):
    state, model, data = make_simulation(kp, kv).unpacked
    dynamics = dict()

    for f, d in config.hardware_data.items():
        duration = len(d["pos"]) * 0.02
        state.reset()

        hinge = data.actuator("calibrator1_C-servo")
        ctrl, pos = [], []
        def brain(s: aapets.common.mujoco.state.MjState):
            # TODO Why 11???
            hinge.ctrl[:] = np.sin(11 * f * s.time) * np.pi / 2
            ctrl.append(hinge.ctrl[0])
            pos.append(hinge.length[0])

        with aapets.common.mujoco.callback.MjcbCallbacks(state, [brain], dict(), config.config.where(duration=duration)):
            mujoco.mj_step(model, data, nstep=int(duration / model.opt.timestep))

        dynamics[f] = dict(pos=pos, ctrl=ctrl)

    return dynamics

def plot(data: dict[dict], suptitle, pdf, rmses=None):
    n = len(list(data.values())[0])
    fig, axes = plt.subplots(n, 2, figsize=(16, 2*n), sharex=True, sharey=True)

    legend = []
    for data_name, data_dict in data.items():
        for (f, d), ax in zip(data_dict.items(), axes):
            for i, (_n, _d) in enumerate(d.items()):
                x = to_time(_d)
                label = data_name
                if label not in legend:
                    legend.append(label)
                else:
                    label = "_" + label

                ax[i].plot(x, _d, label=label)

                title = f"${2*f}\\pi t$: {_n}"
                if rmses is not None and _n == "pos":
                    title = f"{title} [RMSE={rmses[f]:.3g}]"
                ax[i].set_title(title)

    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.legend()
    pdf.savefig(fig, bbox_inches="tight")


def rmse_for(kp, kv, config: TestConfig, pdf=None, desc=None):
    data = run_simulation(kp=kp, kv=kv, config=config)

    rmses = {f: rmse(hd["pos"], d["pos"]) for (f, hd), d in zip(config.hardware_data.items(), data.values())}
    r_vals = np.array(list(rmses.values()))
    r_min, r_max = np.quantile(r_vals, q=[0, 1])
    r_avg, r_dev = np.mean(r_vals), np.std(r_vals)

    print(f"RMSE(KP={kp}, KV={kv}) {r_min:.3g} < {r_avg:.3g} + {r_dev:3g} < {r_max:3g}")
    if pdf is not None:
        title = f"KP: {kp:.3g}, KV: {kv:.3g}"
        if desc is not None:
            title = f"{title} ({desc})"
        title = f"{title}    [RMSE: ${r_min:.3g} \\leq {r_avg:.3g} \\pm {r_dev:3g} \\leq {r_max:3g}$]"
        plot(data={"hardware": config.hardware_data, "simulation": data},
            suptitle=title, pdf=pdf, rmses=rmses)

    return rmses, dict(min=r_min, max=r_max, avg=r_avg, dev=r_dev)


def main():
    args = CLAConfig.parse_command_line_arguments(
        description="Small utility that compares ground-truth obtained from a physical robot"
                    " to simulated dynamics to try and match hinges actuation")

    with open(args.datafile, "rb") as f:
        hardware_data = pickle.load(f)

    # Normalise hardware data from [0-180] to [-1, 1]
    hardware_data = {
        f: {
            # k: trim_initial_plateau(np.deg2rad(v - 90), tol=1e-10)
            k: np.deg2rad(v - 90)
            for k, v in d.items()
        } for f, d in hardware_data.items()
    }

    td = TestConfig(config=args, hardware_data=hardware_data)

    plt.rcParams['text.usetex'] = True

    if not args.optuna:
        if args.render:
            do_render()

        with matplotlib.backends.backend_pdf.PdfPages(args.output.joinpath("calibration.pdf")) as pdf:
            plot({"hardware": hardware_data}, "Hardware ground-truth", pdf)

            rmse_for(kp=ORIGINAL_HINGE_KP, kv=ORIGINAL_HINGE_KV, config=td, pdf=pdf, desc="Apets default")
            rmse_for(kp=2, kv=0.75, config=td, pdf=pdf, desc="Ariel defaults")
            rmse_for(kp=1, kv=1, config=td, pdf=pdf, desc="Intermediate")

    else:
        study = optuna.create_study(direction="minimize")

        def optuna_trial(trial):
            kp = trial.suggest_float("kp", 1e-3, 1e3, log=True)
            kv = trial.suggest_float("kv", 1e-3, 1e3, log=True)
            return rmse_for(kp=kp, kv=kv, config=td)[1]["avg"]

        study.optimize(optuna_trial, n_trials=200)

        print(study.best_params)   # {'kp': ..., 'kv': ...}
        print(study.best_value)    # best rmse

        with matplotlib.backends.backend_pdf.PdfPages(args.output.joinpath("calibration.pdf")) as pdf:
            plot({"hardware": hardware_data}, "Hardware ground-truth", pdf)
            rmse_for(kp=study.best_params["kp"], kv=study.best_params["kv"],
                     config=td, pdf=pdf, desc="Optuna best")

            plots = {
                "optimization_history": vis_mpl.plot_optimization_history(study),
                "param_importances": vis_mpl.plot_param_importances(study),
                "contour": vis_mpl.plot_contour(study, params=["kp", "kv"]),
                "slice": vis_mpl.plot_slice(study, params=["kp", "kv"]),
            }

            for name, ax in plots.items():
                fig = ax.get_figure() if not isinstance(ax, np.ndarray) else ax.flat[0].get_figure()
                fig.suptitle(name)
                pdf.savefig(fig)

if __name__ == "__main__":
    main()
