import os
import time
from datetime import timedelta

import cma
import humanize
import matplotlib
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure

from aapets.common.monitors.metrics_storage import EvaluationMetrics
from .env import EvoEnvironment, GymEnvironment
from .types import Config, Architecture, Trainer, RewardToMonitor
from ..bin.rerun import Arguments as RerunArguments, main as _rerun
from ..common.config import ViewerModes
from ..common.robot_storage import RerunnableRobot


def evolve(args: Config):
    err = 0
    folder = args.data_folder
    folder_str = str(folder) + "/"

    evaluator = EvoEnvironment(args)

    # CMA-ES optimizer from the cma python package.
    initial_mean = evaluator.num_parameters * [0.5]
    initial_std = .5
    options = cma.CMAOptions()
    options.set("verb_filenameprefix", folder_str)
    # options.set("bounds", [-1.0, 1.0])
    options.set("seed", args.seed)
    options.set("tolfun", 0)
    options.set("tolflatfitness", 10)
    es = cma.CMAEvolutionStrategy(initial_mean, initial_std, options)
    # args.threads = 0

    budget = args.budget / (args.duration * args.control_frequency)
    print(f"Converted timestep budget of {args.budget} into {budget} episodes of {args.duration}"
          f" seconds with control frequency of {args.control_frequency}")
    es.optimize(evaluator.cma_evaluate, maxfun=budget, n_jobs=args.threads, verb_disp=1)
    with open(folder.joinpath("cma-es.pkl"), "wb") as f:
        f.write(es.pickle_dumps())

    res = es.result_pretty()
    matplotlib.use("agg")
    cma.plot(folder_str, abscissa=1)
    # plt.tight_layout()
    cma.s.figsave(folder.joinpath('plot.png'), bbox_inches='tight')  # save current figure
    cma.s.figsave(folder.joinpath('plot.pdf'), bbox_inches='tight')  # save current figure

    rerun_metrics = evaluator.evaluate(res.xbest, final=True)
    champion_archive = evaluator.save_champion(res.xbest, rerun_metrics)

    return err, champion_archive


def train(args):
    model_file = args.data_folder.joinpath("model.zip")
    print("Training", model_file)

    n = args.threads or os.cpu_count()
    vec_env = GymEnvironment.make_gym_vec_env(n=n, config=args)
    test_env = GymEnvironment.make_gym_vec_env(n=1, config=args)

    folder = args.data_folder

    budget = args.budget
    eval_callback = EvalCallback(
        test_env,
        best_model_save_path=folder,
        log_path=folder,
        eval_freq=max(100, budget // (10 * n)),
        verbose=1,
        n_eval_episodes=1,
        deterministic=True,
    )

    match args.arch:
        case Architecture.MLP:
            nn_layers = [args.mlp_width for _ in range(args.mlp_depth)]
            policy_kwargs = dict(net_arch=dict(pi=nn_layers, vf=nn_layers))
        case _:
            raise NotImplementedError(f"{args.arch}-based control not implemented with PPO")

    # Define and Train the agent
    model = PPO(
        "MlpPolicy", vec_env, device="cpu", seed=args.seed,
        # From https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/ppo2.yml#L201
        # Based on the ant-v0

        normalize_advantage=True,
        n_steps=256,
        batch_size=32,
        gae_lambda=0.95,
        gamma=0.99,
        learning_rate=2.5e-4,

        policy_kwargs=policy_kwargs,
    )

    model.set_logger(configure(str(folder), ["csv", "tensorboard"]))
    model.learn(
        total_timesteps=budget, progress_bar=True, callback=eval_callback,
    )

    model.save(model_file)

    # Re-evaluate once to get return
    print("== Rerunning to get final performance")

    monitors = {
        monitor.name(): monitor(robot_name=f"{args.robot_name_prefix}1", stepwise=False)
        for monitor in RewardToMonitor.values()
    }
    rerun_env = GymEnvironment(config=args, monitors=monitors)
    obs, _ = rerun_env.reset()

    while not rerun_env.done:
        action, _states = model.predict(obs, deterministic=True)
        obs, _, _, _, _ = rerun_env.step(action)

    rerun_env.close()

    performance = {m.name(): m.value for m in monitors.values()}
    print(">> final performance:", rerun_env._reward_function.value)
    print(">> final performance:", performance)

    # A tad stupid but, eh, it works
    champion_archive = rerun_env.save_champion(model.policy, performance)

    return 0, champion_archive


def rerun(args, champion_archive):
    rr = RerunnableRobot.load(champion_archive)

    if args is None:
        args = rr.config
        assert isinstance(args, Config)

    rerun_args = RerunArguments.copy_from(args)

    rerun_args.robot_archive = champion_archive

    rerun_args.movie = True
    rerun_args.viewer = ViewerModes.NONE

    rerun_args.movie = True
    rerun_args.camera = f"{args.robot_name_prefix}1_tracking-cam"
    rerun_args.camera_angle = 45
    rerun_args.camera_distance = 2
    rerun_args.camera_center = "com"

    rerun_args.plot_format = "png"
    rerun_args.plot_trajectory = True
    rerun_args.plot_brain_activity = True
    rerun_args.render_brain_genotype = False
    rerun_args.render_brain_phenotype = False
    rerun_args.record_pos = True
    rerun_args.record_joints = True

    _rerun(rerun_args)

    make_summary(args, len(rr.brain[1]), rr.metrics)


def make_summary(args, params, metrics: EvaluationMetrics):
    folder = args.data_folder

    steps_per_episode = args.duration * args.control_frequency

    summary = {
        "arch": args.arch.value,
        "trainer": args.trainer.value,
        "reward": args.reward.value,
        "budget": args.budget * steps_per_episode,
        "run": args.seed,
        "body": args.body.name.capitalize(),
        "params": params,
        "depth": args.mlp_depth or np.nan,
        "width": args.mlp_width or np.nan,
        "neighborhood": args.cpg_neighborhood or np.nan,
        "fitness": metrics.data[RewardToMonitor[args.reward].name()]
    }
    summary.update(metrics.data)

    summary = pd.DataFrame.from_dict({k: [v] for k, v in summary.items()})
    summary.index = [folder]

    summary.to_csv(folder.joinpath("summary.csv"))
    print(summary.to_string())


def main():
    args = Config.parse_command_line_arguments("Evolve/train controllers for a morphology")

    # Check validity
    assert args.data_folder is not None, "No output folder provided"
    match args.arch:
        case Architecture.CPG:
            assert args.cpg_neighborhood is not None, "CPG architecture requested without neighborhood"
        case Architecture.MLP:
            assert args.mlp_width is not None, "MLP architecture requested without width"
            assert args.mlp_depth is not None, "MLP architecture requested without depth"

    # Create filesystem
    if args.data_folder.exists() and not args.overwrite:
        raise FileExistsError(
            f"Data folder '{args.data_folder.absolute()}' already exists and overwrite was not requested")

    args.data_folder.mkdir(exist_ok=True, parents=True)
    args.write_yaml(args.data_folder.joinpath("config.yaml"))
    print()
    args.pretty_print()
    print()

    # Go!
    start = time.perf_counter()
    match args.trainer:
        case Trainer.CMA:
            err, archive = evolve(args)
        case Trainer.PPO:
            err, archive = train(args)
        case _:
            raise ValueError(f"Unknown trainer type: {args.trainer}")

    # Re-evaluate
    rerun(args, archive)

    duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start))
    print(f"Completed {args.trainer} in {duration} with exit code {err}")


if __name__ == "__main__":
    main()

