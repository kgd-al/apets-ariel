from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from mujoco import MjSpec, mj_forward, mj_step

from aapets.common import controllers
from aapets.common.config import ViewerConfig, ViewerModes
from aapets.common.monitors.abcpg_handler import ABCPGHandler
from aapets.common.monitors.plotters.record import MovieRecorder
from aapets.common.mujoco.callback import MjcbCallbacks
from aapets.common.mujoco.state import MjState
from aapets.common.robot_storage import RerunnableRobot
from aapets.bin.rerun import Arguments as RerunArguments
from aapets.common.world_builder import adjust_shoulder_camera


def compliance_summaries(champion: Path):
    champion_tasks = list(champion.parent.glob("champion_*.zip"))

    plot_merged_trajectories(champion, champion_tasks)        
    record_merged_performance(champion, champion_tasks)


def plot_merged_trajectories(champion: Path, tasks: list[Path]):
    fig, ax = plt.subplots()

    v_max = 0

    for w in tasks:
        f = w.with_suffix(".trajectory.csv")
        tag = w.stem.split("_")[1]

        df = pd.read_csv(f)
        ax.plot(df.x, df.y, label=tag)
        v_max = max(v_max, df[["x", "y"]].abs().max().max())

        tl, ta = [float(x) for x in tag.split("m")]
        ta = np.deg2rad(ta)
        tx, ty = tl * np.array([np.cos(ta), np.sin(ta)])
        ax.scatter([tx], [ty])
        v_max = max(v_max, max(tx, ty))

    if v_max > 0:
        v_max *= 1.1
        ax.set_xlim(-v_max, v_max)
        ax.set_ylim(-v_max, v_max)

    ax.legend()

    fig.savefig(champion.with_suffix(".merged_trajectories.png"))


def record_merged_performance(champion: Path, tasks: list[Path]):
    print()
    print("#"*60)
    print(f"Merging and recording {champion.stem}_*.zip")

    records = [RerunnableRobot.load(task) for task in tasks]

    robot_name, target_name = "apet1_world", "target"

    world = MjSpec.from_string(records[0].mj_spec.to_xml())
    world.delete(world.body(robot_name))
    world.delete(world.body(target_name))
    frame = world.worldbody.sites[0]

    robots, targets = [], []
    for i, record in enumerate(records):
        spec = record.mj_spec
        prefix = f"robot{i+1}_"

        for asset in list(spec.materials + spec.textures + spec.meshes + spec.skins + spec.hfields):
            spec.delete(asset)
            
        robot = spec.body(robot_name)
        joint_names = {j.name for j in robot.find_all("joint")}
        actuators_to_copy = [a for a in spec.actuators if a.target in joint_names]

        target = frame.attach_body(spec.body(target_name), prefix, "")
        target.name = f"target{i+1}"
        targets.append(target.name)

        robot = frame.attach_body(robot, prefix, "")
        robot.name = f"{prefix}world"
        robots.append(prefix[:-1])

        bit = 1 << i
        for g in robot.find_all("geom") + target.find_all("geom"):
            g.contype = bit
            g.conaffinity = bit

        for a in actuators_to_copy:
            new_act = world.add_actuator(
                trntype=a.trntype,
                target=a.target,
                gaintype=a.gaintype, gainprm=a.gainprm,
                biastype=a.biastype, biasprm=a.biasprm,
                dyntype=a.dyntype, dynprm=a.dynprm,
                gear=a.gear,
                ctrlrange=a.ctrlrange, ctrllimited=a.ctrllimited,
                forcerange=a.forcerange, forcelimited=a.forcelimited,
            )
            new_act.name = f"r{i+1}_{a.name}" if a.name else ""

    floor = world.geom("floor")
    floor.contype = 0
    floor.conaffinity = (1 << len(records)) - 1

    world.visual.global_.offwidth = 1920
    world.visual.global_.offheight = 1080    

    args = RerunArguments.copy_from(records[0].config)
    args.movie = True
    args.viewer = ViewerModes.NONE

    args.movie = "mp4"
    args.movie_width = 1920
    args.movie_height = 1080
    args.camera = "pretty-cam"
    args.camera_angle = 30
    args.camera_distance = 4
    args.camera_center = "com"

    args.plot_format = "png"
    args.plot_trajectory = True
    args.plot_brain_activity = True
    args.plot_rewards = True
    args.render_brain_genotype = False
    args.render_brain_phenotype = False
    args.record_position = True
    args.record_joints = True

    adjust_shoulder_camera(world, args, None, None)

    state, model, data = MjState.from_spec(world).unpacked
    mj_forward(model, data)

    monitors = dict()

    movie_file = champion.with_suffix(f".merged.{args.movie}")
    monitors["movie_recorder"] = MovieRecorder(
        args.movie_framerate, args.movie_width, args.movie_height,
        movie_file,
        camera=args.camera, shadows=True
    )

    # print(world.to_xml())
    # with open("world.xml", "wt") as f:
    #     f.write(world.to_xml())

    brains = []
    for record, robot, target in zip(records, robots, targets):
        brains.append(brain := controllers.get(record.brain[0])(
            weights=record.brain[2], state=state, name=robot, **record.brain[1] 
        ))    
        monitors[f"ab_handler_{robot}"] = ABCPGHandler(
            brain, robot, target
        )

    with MjcbCallbacks(state, brains, monitors, args):
        mj_step(model, data, nstep=int(args.duration / model.opt.timestep))

    print("Generated", movie_file)
