from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import mujoco as mj
from tqdm.rich import tqdm

from aapets.common.misc.config_base import IntrospectiveAbstractConfig
from aapets.common.robot_storage import RerunnableRobot

from aapets.zoo.evolve import Arguments as ZooArguments
from aapets.bin.rerun import Arguments as RerunArguments, main as rerun
from common.config import ViewerModes, ViewerConfig, BaseConfig


@dataclass
class Config(BaseConfig, ViewerConfig):
    @classmethod
    def yaml_tag(cls): return "ZooFlipConfig"

    root: Annotated[Path, "Where to look for robots to creepify", dict(required=True)] = None


def main(config: Config):
    root = config.root
    if root.name != "__champions":
        root = root.joinpath("__champions")

    rerun_args = RerunArguments.copy_from(config)

    # rerun_args.robot_archive = champion_archive

    rerun_args.movie = True
    rerun_args.viewer = ViewerModes.NONE

    rerun_args.movie = "mp4"
    rerun_args.camera = f"{config.robot_name_prefix}1_tracking-cam"
    rerun_args.camera_angle = 45
    rerun_args.camera_distance = 2
    rerun_args.camera_center = "com"

    rerun_args.plot_format = "png"
    rerun_args.plot_trajectory = False
    rerun_args.plot_brain_activity = False
    rerun_args.plot_rewards = False
    rerun_args.render_brain_genotype = False
    rerun_args.render_brain_phenotype = False
    rerun_args.record_position = False
    rerun_args.record_joints = False

    for champ in tqdm(list(root.glob("**/champion.zip"))):
        print(champ)
        robot = RerunnableRobot.load(champ)
        spec = robot.mj_spec
        # print(spec.to_xml())
        # print(type(spec))
        world = spec.body("apet1_world")
        world.quat = [0, 1, 0, 0]
        # print(world)
        free_joint = None
        for joint in world.joints:
            if joint.type == mj.mjtJoint.mjJNT_FREE:
                free_joint = joint
                break
        spec.delete(free_joint)
        # print(spec.to_xml())

        flipped_archive = champ.with_name("champion_flipped.zip")
        robot.save(flipped_archive)

        for a in [45, 90]:
            rerun_args.camera_angle = a
            rerun_args.robot_archive = flipped_archive
            rerun(rerun_args)
        break


if __name__ == "__main__":
    main(Config.parse_command_line_arguments(
        "Little utility to flip robots on their back and pin them in place"))
