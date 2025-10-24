import math
import pprint
from collections import namedtuple
from typing import ClassVar, Optional, Callable, Tuple

import mujoco
import numpy as np
from mujoco import MjData, MjSpec, MjsBody, viewer, MjModel

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.environments import SimpleFlatWorld, BaseWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder
from .config import ExperimentType, CommonConfig
from .evaluation_result import EvaluationResult
from .genotype import Genotype
from .phenotype import decode_body, decode_brain

# class MultiCameraOverlay:
#     class Mode(StrEnum):  # Inset configuration
#         NONE = auto()
#         PRETTY_RGB = auto()  # rgb coloring and comfortable size
#         RGB = auto()         # rgb coloring and accurate size
#         PRETTY = auto()      # color-mapped vision and comfortable size
#         ACCURATE = auto()    # color-mapped vision and accurate size (WYSWYG)
#
#     PRETTY_CAMERA_WIDTH_RATIO = .25
#
#     def __init__(self, vision, fd_cm, inv_cm):
#         self.viewer = None
#         self.cameras = None
#         self.vision = vision
#         self.mode = self.Mode.ACCURATE
#
#         self.vopt, self.scene, self.ctx = None, None, None
#
#         self.camera_buffer = np.zeros(shape=(vision[1], vision[0], 3), dtype=np.uint8)
#         self.window_buffer = np.zeros(0)
#
#         self.fd_cm, self.inv_cm = fd_cm, inv_cm
#
#     def start(self, model: MjModel, data: MjData, viewer: CustomMujocoViewer):
#         self.viewer = viewer
#         self.cameras = []
#
#         for name_index in model.name_camadr:
#             terminator = model.names.find(b'\x00', name_index)
#             name = model.names[name_index:terminator].decode("ascii")
#             if "mbs" in name:
#                 camera = mujoco.MjvCamera()
#                 camera.fixedcamid = model.camera(name).id
#                 camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
#                 self.cameras.append(camera)
#
#         if len(self.cameras) > 0:
#             # create options, camera, scene, context
#             self.vopt = mujoco.MjvOption()
#             self.scene = mujoco.MjvScene(model, maxgeom=10000)
#
#             self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
#
#             viewer._viewer_backend.add_callback(
#                 mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
#                 "Camera inset",
#                 glfw.KEY_PAGE_UP,
#                 lambda: self.mode.name.capitalize().replace("_", " "),
#                 self.next_mode
#             )
#
#             viewer._viewer_backend.add_callback(None, None, glfw.KEY_PAGE_DOWN, None, self.prev_mode)
#
#     def next_mode(self, step=1):
#         modes = list(self.Mode)
#         index = (modes.index(self.mode) + step) % len(modes)
#         print(f"{self.mode} -> {modes}[{index}]")
#         self.mode = modes[index]
#         print(f">>", self.mode)
#
#     def prev_mode(self):
#         self.next_mode(-1)
#
#     def process(self, model: MjModel, data: MjData, viewer: _MujocoViewerBackend):
#         if not viewer.is_alive or self.mode is self.Mode.NONE:
#             return
#
#         start_0 = time.time()
#
#         vw, vh = self.vision
#         width, height = viewer.viewport.width, viewer.viewport.height
#         for camera in self.cameras:
#             inset_width = int(width * self.PRETTY_CAMERA_WIDTH_RATIO)
#             inset_height = int(inset_width * vh / vw)
#             if self.mode in [self.Mode.PRETTY, self.Mode.PRETTY_RGB]:
#                 camera_width, camera_height = inset_width, inset_height
#             else:
#                 camera_width, camera_height = self.vision
#
#             mujoco.mjv_updateScene(model, data, self.vopt, None, camera, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
#
#             viewport_inset = mujoco.MjrRect(width - inset_width, height - inset_height, inset_width, inset_height)
#
#             mujoco.mjr_rectangle(mujoco.MjrRect(viewport_inset.left - 1, viewport_inset.bottom - 1,
#                                                 viewport_inset.width + 2, viewport_inset.height + 2),
#                                  1, 0, 0, 1)
#
#             if self.mode is self.Mode.PRETTY_RGB:
#                 viewport_render = mujoco.MjrRect(width - camera_width, height - camera_height,
#                                                  camera_width, camera_height)
#                 mujoco.mjr_render(viewport_render, self.scene, self.ctx)
#
#             else:
#                 viewport_render = mujoco.MjrRect(0, 0, camera_width, camera_height)
#                 # View has been rendered, now grab and tweak it
#
#                 rescale = (self.mode in [self.Mode.RGB, self.Mode.ACCURATE])
#
#                 if rescale:
#                     buffer = self.camera_buffer
#                 else:
#                     size = inset_width * inset_height * 3
#                     if len(self.window_buffer) != size:
#                         self.window_buffer = np.ones(shape=(inset_height, inset_width, 3), dtype=np.uint8)
#                     buffer = self.window_buffer
#
#                 start_rn = time.time()
#                 mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.ctx)
#                 mujoco.mjr_render(viewport_render, self.scene, self.ctx)
#                 mujoco.mjr_readPixels(buffer, None, viewport_render, self.ctx)
#                 mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)
#                 print(" grab:", time.time() - start_rn)
#
#                 if self.mode in [self.Mode.PRETTY, self.Mode.ACCURATE]:
#                     start_cm = time.time()
#                     buffer = self.inv_cm(buffer)
#                     print(" cmap:", time.time() - start_cm)
#
#                 if rescale:
#                     start_rs = time.time()
#                     buffer = cv2.resize(buffer, (inset_width, inset_height),
#                                         interpolation=cv2.INTER_NEAREST)
#                     print("scale:", time.time() - start_rs)
#
#                 start_dp = time.time()
#                 mujoco.mjr_drawPixels(rgb=buffer.flatten(), depth=None, viewport=viewport_inset, con=self.ctx)
#                 print(" draw:", time.time() - start_dp)
#
#         print(f"[{self.mode}] Rendering in", time.time() - start_0)
#
#
# class PersistentViewerOptions:
#     @classmethod
#     def persistent_storage(cls, viewer):
#         path = cls.backend(viewer).CONFIG_PATH
#         path.parent.mkdir(parents=True, exist_ok=True)
#         return path
#
#     @classmethod
#     def backend(cls, viewer) -> MujocoViewer:
#         backend = viewer._viewer_backend
#         assert isinstance(backend, MujocoViewer)
#         return backend
#
#     @classmethod
#     def window(cls, viewer):
#         return cls.backend(viewer).window
#
#     @classmethod
#     def start(cls, model: MjModel, data: MjData, viewer: CustomMujocoViewer):
#         backend = cls.backend(viewer)
#
#         backend.original_close = backend.close
#         def monkey_patch():
#             cls.end(viewer)
#             backend.original_close()
#         backend.close = monkey_patch
#
#         try:
#             with open(cls.persistent_storage(viewer), "r") as f:
#                 if (config := yaml.safe_load(f)) is not None:
#                     # print("[kgd-debug] restored viewer config: ", config)
#                     window = PersistentViewerOptions.window(viewer)
#                     glfw.restore_window(window)
#                     glfw.set_window_pos(window, *config["pos"])
#                     glfw.set_window_size(window, *config["size"])
#
#         except FileNotFoundError:
#             pass
#
#     @classmethod
#     def end(cls, viewer: CustomMujocoViewer):
#         if not cls.backend(viewer).is_alive:
#             return
#         with open(cls.persistent_storage(viewer), "w") as f:
#             window = cls.window(viewer)
#             yaml.safe_dump(
#                 dict(
#                     pos=glfw.get_window_pos(window),
#                     size=glfw.get_window_size(window),
#                 ),
#                 f
#             )


ScenarioData = namedtuple(
    "ScenarioData",
    field_names=[
        "fitness_name", "fitness_bounds",
        "descriptor_names", "descriptor_bounds"
    ]
)


def bounds(l, u):
    def wrapper(f):
        f.bounds = (l, u)
        return f
    return wrapper


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

        fitness_fn = getattr(cls, f"get_{fitness.lower()}")
        assert isinstance(fitness_fn, Callable)
        cls.fitness = fitness_fn

        descriptor_fns = [getattr(cls, f"get_{desc.lower()}") for desc in config.descriptors]
        descriptor_bounds = [fn.bounds for fn in descriptor_fns]
        cls.descriptors = [(d, fn) for d, fn in zip(config.descriptors, descriptor_fns)]

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

        world = SimpleFlatWorld()

        cls.add_defaults(world.spec)
        cls.add_ball(world.spec, pos=(1, 0, 0), size=.05)
        cls.add_robot_body(genotype, world, config)

        print(world.spec.to_xml())

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

        mujoco.set_mjcb_control(brain.control)

        cls.run(model, data, "launcher", config)

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
    def run(cls, model, data, mode, config: CommonConfig):
        match mode:
            case "simple":
                # This disables visualisation (fastest option)
                simple_runner(
                    model,
                    data,
                    duration=config.duration,
                )

            case "frame":
                # Render a single frame (for debugging)
                save_path = str("tmp/robot.png")
                single_frame_renderer(model, data, save=True, save_path=save_path)

            case "video":
                # This records a video of the simulation
                path_to_video_folder = str("tmp/videos")
                video_recorder = VideoRecorder(output_folder=path_to_video_folder)

                # Render with video recorder
                cam_quat = np.zeros(4)
                mujoco.mju_euler2Quat(cam_quat, np.deg2rad([30, 0, 0]), "XYZ")
                video_renderer(
                    model,
                    data,
                    duration=config.duration,
                    video_recorder=video_recorder,
                    cam_fovy=7,
                    cam_pos=(2., -1., 2.),
                    cam_quat=cam_quat,
                )
            case "launcher":
                # This opens a liver viewer of the simulation
                viewer.launch(
                    model=model,
                    data=data,
                )
            case "no_control":
                # If mj.set_mjcb_control(None), you can control the limbs manually.
                mujoco.set_mjcb_control(None)
                viewer.launch(
                    model=model,
                    data=data,
                )

    @staticmethod
    def add_defaults(spec: MjSpec):
        spec.worldbody.add_light(diffuse=[.5, .5, .5], pos=[0, 0, 3], dir=[0, 0, -1])

    @staticmethod
    def add_robot_body(genotype: Genotype, world: BaseWorld, config: CommonConfig) -> MjSpec:
        body = decode_body(genotype.body, config)
        robot = construct_mjspec_from_graph(body)
        # print(robot.spec.to_xml())
        # for site in robot.spec.sites:
        #     robot.spec.delete(site)
        return world.spawn(
            robot.spec, spawn_prefix="apet",
            correct_collision_with_floor=True,
            validate_no_collisions=True
        )

    @staticmethod
    def add_robot_brain(genotype: Genotype, model: MjModel, data: MjData, config: CommonConfig):
        joint_data = {
            name: data.joint(i).xanchor for i in range(model.njnt)
            if len((name := data.joint(i).name)) > 0
        }
        return decode_brain(genotype.brain, joint_data, data, config)

    @staticmethod
    def add_ball(world: MjSpec, pos: Tuple[float, float, float], size: float) -> MjsBody:
        pos = (*pos[:-1], pos[-1] - .5 * size)
        ball = world.worldbody.add_body(
            name="ball",
            pos=pos,
        )
        ball.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(size, size, size),
            rgba=(0.8, 0.2, 0.2, 1.0),
        )
        ball.add_freejoint()
        return ball

    @staticmethod
    @bounds(0, 5)
    def get_speed(model: MjModel, data: MjData, tracked: dict) -> float:
        return (
            float(np.sqrt(sum(v**2 for v in tracked["apet1_core"].xpos))) / data.time
            if data.time > 0 else 0
        )

    @staticmethod
    @bounds(0, 4)
    def get_weight(model: MjModel, data: MjData, tracked: dict) -> float:
        return float(model.body("apet1_core").subtreemass[0])
