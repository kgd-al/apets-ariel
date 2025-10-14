import pprint
from dataclasses import dataclass, field
from typing import ClassVar

import mujoco

from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.video_recorder import VideoRecorder
from config import SimuConfig, ExperimentType, CommonConfig
from genotype import Genotype
from src.aapets.evaluation_result import EvaluationResult


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


class Evaluator:
    config: ClassVar[CommonConfig] = None

    @classmethod
    def initialize(cls, config: CommonConfig, verbose=True):
        cls.config = config

        cls._log = getattr(config, "logger", None)
        if cls._log and verbose:
            cls._log.info(f"Configuration:\n"
                          f"{pprint.pformat(cls.config)}\n")

        cls._fitness = getattr(cls, f"fitness_{config.experiment.lower()}")

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

        body = mujoco.MjSpec()
        ball = body.worldbody.add_body(name="ball")
        ball.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.1, 0.1, 0.1),
            rgba=(0.8, 0.2, 0.2, 1.0),
        )

        world.spawn(body)

        model = world.spec.compile()
        data = mujoco.MjData(model)

        # Non-default VideoRecorder options
        # video_recorder = VideoRecorder(output_folder=DATA)

        # Reset state and time of simulation
        mujoco.mj_resetData(model, data)

        # Define action specification and set policy
        # data.ctrl = RNG.normal(scale=0.1, size=model.nu)

        steps_per_loop = config.simu.control / model.opt.timestep
        while data.time < config.simu.duration:
            mujoco.mj_step(model, data, nstep=steps_per_loop)

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

        fd.update(states=scene_states)
        fitness = cls._fitness(fd)
        try:
            assert not math.isnan(fitness) and not math.isinf(fitness), f"{fitness=}"
        except Exception as e:
            raise RuntimeError(f"{fitness=}") from e
        return fitness

