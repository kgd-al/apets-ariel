import math
import pprint
from collections import namedtuple
from typing import ClassVar, Optional, Callable, Tuple

import mujoco
from mujoco import MjData, MjSpec, MjsBody, viewer, MjModel

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.simulation.environments import SimpleFlatWorld, BaseWorld
from ariel.utils.runners import simple_runner
from . import metrics, canonical_bodies
from .config import ExperimentType, CommonConfig
from .evaluation_result import EvaluationResult
from .genotype import Genotype
from .misc.debug import kgd_debug
from .mj_callback import ControlAndTrack
from .phenotype import decode_body, decode_brain

ScenarioData = namedtuple(
    "ScenarioData",
    field_names=[
        "fitness_name", "fitness_bounds",
        "descriptor_names", "descriptor_bounds"
    ]
)


class Evaluator:
    config: ClassVar[Optional[CommonConfig]] = None

    experiment_to_fitness = {
        ExperimentType.LOCOMOTION: "speed",
        ExperimentType.DIRECTED_LOCOMOTION: "xspeed",
        ExperimentType.TARGETED_LOCOMOTION: "proximity",
        ExperimentType.TRACKING: "proximity"
    }

    @classmethod
    def initialize(cls, config: CommonConfig) -> ScenarioData:
        cls.config = config

        cls._log = getattr(config, "logger", None)
        if cls._log and config.verbosity > 1:
            cls._log.info(f"Configuration:\n"
                          f"{pprint.pformat(cls.config)}\n")

        if config.experiment is None:
            config.experiment = ExperimentType.LOCOMOTION

        fitness = cls.experiment_to_fitness[config.experiment]
        config.fitness = fitness

        fitness_fn = getattr(metrics, f"get_{fitness.lower()}")
        assert isinstance(fitness_fn, Callable)
        cls.fitness = fitness_fn

        descriptor_fns = [
            getattr(metrics, f"get_{desc.lower()}") for desc in config.descriptors
        ]
        descriptor_bounds = [fn.bounds for fn in descriptor_fns]
        cls.descriptors = [
            (d, fn) for d, fn in zip(config.descriptors, descriptor_fns)
        ]

        return ScenarioData(
            fitness_name=fitness, fitness_bounds=fitness_fn.bounds,
            descriptor_names=config.descriptors, descriptor_bounds=descriptor_bounds,
        )

    @classmethod
    def evaluate(cls, genotype: Genotype) -> EvaluationResult:
        """
        Evaluate a *single* robot.

        Fitness is the distance traveled on the xy plane.

        :param genotype: The genotype to develop into a robot and then simulate.
        :returns: Fitness of the robot.
        """

        config = cls.config
        kgd_debug(f"{config.duration} seconds")

        world = SimpleFlatWorld()

        cls.add_defaults(world.spec)
        cls.add_ball(world.spec, pos=(1, 0, 0), radius=.05)

        kgd_debug("ping")
        cls.add_robot_body(genotype, world, config)
        kgd_debug("pong")

        # print(world.spec.to_xml())

        model = world.spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        geoms = world.spec.worldbody.find_all("geom")
        names_to_bind = ["core"]
        to_track = {
            geom.name: data.bind(geom)
            for geom in geoms
            if any(name in geom.name for name in names_to_bind)
        }
        # print(to_track)

        # single_frame_renderer(model, data, save=True, save_path="foo.png",
        #                       width=640, height=480,
        #                       # cam_fovy=8,
        #                       cam_pos=(-1, 0, 1))

        brain = cls.add_robot_brain(genotype, model, data, config)
        if config.plot_brain_genotype:
            genotype.brain.to_dot(
                Genotype.Data(config=config, seed=config.seed).brain,
                config.output_folder.joinpath("brain_cppn.png")
            )
        if config.plot_brain_phenotype:
            brain.plot_as_network(config.output_folder.joinpath("brain.png"))

        callback_object = ControlAndTrack(brain, to_track, config)

        mujoco.set_mjcb_control(callback_object.mjcb_callback)

        # == Running ==
        cls.run(model, data, config)
        # =============

        # Final call for rounder measurements
        callback_object.mjcb_callback(model, data)

        if config.plot_brain_activity:
            callback_object.plot_brain_activity(config.output_folder.joinpath("brain_activity.pdf"))

        if config.plot_trajectory:
            callback_object.plot_trajectory(config.output_folder.joinpath("trajectory.pdf"))

        # <<<<<<
        #
        # if options.rerun:
        #     scene.add_camera(
        #         name="tracking-camera",
        #         mode="targetbody",
        #         target=f"mbs{len(robots)+len(objects)}/",
        #         pos=camera_position(config.experiment)
        #     )
        #
        # >>>>>>>

        fitness = cls.fitness(model, data, to_track)
        if math.isnan(fitness) or math.isinf(fitness):
            raise RuntimeError(f"non-finite {fitness=}")

        descriptors = {k: d(model, data, to_track) for k, d in cls.descriptors}
        for k, d in descriptors.items():
            if math.isnan(d) or math.isinf(d):
                raise RuntimeError(f"non-finite descriptor {k}={d}")

        return EvaluationResult(
            fitnesses={config.fitness: fitness},
            infos=dict(descriptors=descriptors)
        )

    @classmethod
    def run(cls, model, data, config: CommonConfig):
        if config.viewer:
            mujoco.viewer.launch(
                model=model,
                data=data,
            )

        else:
            simple_runner(
                model,
                data,
                duration=config.duration,
                steps_per_loop=1
            )

            # # This records a video of the simulation
            # path_to_video_folder = str("tmp/videos")
            # video_recorder = VideoRecorder(output_folder=path_to_video_folder)
            #
            # # Render with video recorder
            # cam_quat = np.zeros(4)
            # mujoco.mju_euler2Quat(cam_quat, np.deg2rad([30, 0, 0]), "XYZ")
            # video_renderer(
            #     model,
            #     data,
            #     duration=config.duration,
            #     video_recorder=video_recorder,
            #     cam_fovy=7,
            #     cam_pos=(2., -1., 2.),
            #     cam_quat=cam_quat,
            # )

    @staticmethod
    def add_defaults(spec: MjSpec):
        spec.worldbody.add_light(diffuse=[.5, .5, .5], pos=[0, 0, 3], dir=[0, 0, -1])

        # TODO: does not work
        # spec.option.integrator = mjtIntegrator.mjINT_RK4

    @staticmethod
    def add_robot_body(genotype: Genotype, world: BaseWorld, config: CommonConfig) -> MjSpec:
        print(canonical_bodies.get_all())
        if (body_name := config.fixed_body) is None:
            kgd_debug("ping")
            body = decode_body(genotype.body, config)
            kgd_debug("pong")
        else:
            body = canonical_bodies.get(body_name)

        if not isinstance(body, CoreModule):
            robot = construct_mjspec_from_graph(body)
        else:
            robot = body

        return world.spawn(
            robot.spec, spawn_prefix=config.robot_name_prefix,
            correct_collision_with_floor=True,
            validate_no_collisions=True
        )

    @staticmethod
    def add_robot_brain(genotype: Genotype, model: MjModel, data: MjData, config: CommonConfig):
        return decode_brain(genotype.brain, model, data, config)

    @staticmethod
    def add_ball(world: MjSpec, pos: Tuple[float, float, float], radius: float) -> MjsBody:
        pos = (*pos[:-1], pos[-1] + radius)
        ball = world.worldbody.add_body(
            name="ball",
            pos=pos,
        )
        ball.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(radius, 0, 0),
            rgba=(0.8, 0.2, 0.2, 1.0),
            name="ball_geom"
        )
        ball.add_freejoint()
        return ball
