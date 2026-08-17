"""Not actually implemented. As of 31-07-2026, hinge test data was only collected 
for a flipped robot, making comparison moot."""

import pickle
from dataclasses import dataclass
from typing import Annotated

import ariel.body_phenotypes.robogen_lite.modules.hinge as hinge_module
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import optuna
import optuna.visualization.matplotlib as vis_mpl
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
from ariel.utils.renderers import single_frame_renderer
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ....g_cpg.worlds import default_world
from ....common.config import BaseConfig
from ....common.mujoco.callback import MjcbCallbacks
from ....common.mujoco.state import MjState

ORIGINAL_HINGE_KP = hinge_module.HINGE_KP
ORIGINAL_HINGE_KV = hinge_module.HINGE_KV
ORIGINAL_HINGE_A = hinge_module.HINGE_ARMATURE


@dataclass
class CLAConfig(BaseConfig):
    datafile: Annotated[Path, "Location of ground-truth datafile", dict(required=True)] = None
    record: Annotated[Path, "Location of robot used for the measurements (controller is ignored)",
                      dict(required=True)] = None

    output: Annotated[Path, "Where to store the output"] = "./"
    render: Annotated[bool, "Whether to render the robot used for simulation"] = False

    # Hardware was measured at 20Hz
    control_frequency: Annotated[int, "How often to query the control for new outputs (Hz)"] = 20

    optuna: Annotated[bool, "Whether to just look at values or use optuna to fine-tune"] = False
    optuna_db: Annotated[str, "Where to look for a persistent optuna study database"] = "sqlite:///hinge_tuning.db"


@dataclass
class Parameters:
    kp: float = ORIGINAL_HINGE_KP
    kv: float = ORIGINAL_HINGE_KP
    a: float = ORIGINAL_HINGE_A

@dataclass
class TestConfig:
    config: CLAConfig

    hardware_data: dict[str, dict[str, float]]

    def to_time(self, data): 
        return np.array([i / self.config.control_frequency for i in range(len(data))])


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

def make_simulation(p: Parameters = None):
    p = p or Parameters()
    core = CoreModule()
    brick = BrickModule()

    hinge_module.HINGE_KP = p.kp
    hinge_module.HINGE_KV = p.kv
    hinge_module.HINGE_A = p.a
    hinge = HingeModule()
    hinge_module.HINGE_KP = ORIGINAL_HINGE_KP
    hinge_module.HINGE_KV = ORIGINAL_HINGE_KV
    hinge_module.HINGE_A = ORIGINAL_HINGE_A

    core.sites[ModuleFaces.FRONT].attach_body(hinge.body, prefix="C-")
    hinge.sites[ModuleFaces.FRONT].attach_body(brick.body, prefix="foo-")

    core.spec.body("core").quat = (np.cos(-np.pi / 4), 0, np.sin(-np.pi / 4), 0)
    world = default_world(core.spec, "calibrator")
    return MjState.from_spec(world.spec)


def do_render():
    state = make_simulation()
    single_frame_renderer(state.model, state.data,
                          width=640, height=480,
                          save=True, save_path="./calibrator.png")


def run_simulation(p: Parameters, config: TestConfig):
    state, model, data = make_simulation(p).unpacked
    dynamics = dict()

    for f, d in config.hardware_data.items():
        duration = len(d["pos"]) / config.config.control_frequency
        state.reset()

        hinge = data.actuator("calibrator1_C-servo")
        ctrl, pos = [], []
        def brain(s: MjState):
            # TODO Why 11???
            hinge.ctrl[:] = np.sin(2 * np.pi * f * s.time) * np.pi / 2
            ctrl.append(hinge.ctrl[0])
            pos.append(hinge.length[0])

        with MjcbCallbacks(state, [brain], dict(), config.config.where(duration=duration)):
            mujoco.mj_step(model, data, nstep=int(duration / model.opt.timestep))

        dynamics[f] = dict(pos=pos, ctrl=ctrl)

    return dynamics

def plot(data: dict[dict], config: TestConfig, suptitle, pdf, rmses=None):
    data = dict(hardware=config.hardware_data, **data)
    n = len(list(data.values())[0])
    fig, axes = plt.subplots(n, 2, figsize=(16, 2*n), sharex=True, sharey=True)

    legend = []
    for data_name, data_dict in data.items():
        for (f, d), ax in zip(data_dict.items(), axes):
            for i, (_n, _d) in enumerate(d.items()):
                x = config.to_time(_d)
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


def rmse_for(p: Parameters, config: TestConfig, pdf=None, desc=None):
    data = run_simulation(p, config=config)

    rmses = {f: rmse(hd["pos"], d["pos"]) for (f, hd), d in zip(config.hardware_data.items(), data.values())}
    r_vals = np.array(list(rmses.values()))
    r_min, r_max = np.quantile(r_vals, q=[0, 1])
    r_avg, r_dev = np.mean(r_vals), np.std(r_vals)

    info = ""
    if desc is not None:
        info += desc + "\t"
    info += f"RMSE(KP={p.kp}, KV={p.kv}, A={p.a}) {r_min:.3g} < {r_avg:.3g} + {r_dev:3g} < {r_max:3g}"
    print(info)

    if pdf is not None:
        title = f"KP: {p.kp:.3g}, KV: {p.kv:.3g}, A: {p.a:.3g}"
        if desc is not None:
            title = f"{title} ({desc})"
        title = f"{title}    [RMSE: ${r_min:.3g} \\leq {r_avg:.3g} \\pm {r_dev:3g} \\leq {r_max:3g}$]"
        plot(data=dict(simulation=data), config=config, suptitle=title, pdf=pdf, rmses=rmses)

    return rmses, dict(min=r_min, max=r_max, avg=r_avg, dev=r_dev)


def main():
    args = CLAConfig.parse_command_line_arguments(
        description="Small utility that compares ground-truth obtained from a physical robot"
                    " to simulated dynamics to try and match hinges actuation")

    with open(args.datafile, "rb") as f:
        hardware_data = pickle.load(f)
    print(hardware_data)
    exit(42)

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

        with PdfPages(args.output.joinpath("calibration.pdf")) as pdf:
            plot(dict(), td, "Hardware ground-truth", pdf)

            rmse_for(Parameters(), config=td, pdf=pdf, desc="Apets default")
            rmse_for(Parameters(kp=2, kv=0.75, a=0.05),
                     config=td, pdf=pdf, desc="Intermediate (ideal?)")
            rmse_for(Parameters(kp=1, kv=1), 
                     config=td, pdf=pdf, desc="Ariel defaults")

    else:
        study = optuna.create_study(
            study_name="Study",
            storage=args.optuna_db,
            load_if_exists=True,
            direction="minimize"
        )

        def optuna_trial(trial):
            kp = trial.suggest_float("kp", 1e-3, 1e3, log=True)
            kv = trial.suggest_float("kv", 1e-3, 1e3, log=True)
            a = trial.suggest_float("a", 1e-3, 1e3, log=True)
            return rmse_for(Parameters(kp=kp, kv=kv, a=a), config=td)[1]["avg"]

        study.optimize(optuna_trial, n_trials=200)

        print(study.best_params)   # {'kp': ..., 'kv': ...}
        print(study.best_value)    # best rmse

        with PdfPages(args.output.joinpath("calibration.pdf")) as pdf:
            plot(dict(), td, "Hardware ground-truth", pdf)
            rmse_for(Parameters(), config=td, pdf=pdf, desc="Apets default")
            rmse_for(Parameters(kp=1, kv=1), 
                     config=td, pdf=pdf, desc="Ariel defaults")
            rmse_for(Parameters(kp=study.best_params["kp"],
                                kv=study.best_params["kv"],
                                a=study.best_params["a"]),
                     config=td, pdf=pdf, desc="Optuna best")

            plots = {
                "optimization_history": vis_mpl.plot_optimization_history(study),
                "param_importances": vis_mpl.plot_param_importances(study),
                "contour": vis_mpl.plot_contour(study, params=["kp", "kv", "a"]),
                "slice": vis_mpl.plot_slice(study, params=["kp", "kv", "a"]),
            }

            for name, ax in plots.items():
                fig = ax.get_figure() if not isinstance(ax, np.ndarray) else ax.flat[0].get_figure()
                fig.suptitle(name)
                pdf.savefig(fig)

if __name__ == "__main__":
    main()
