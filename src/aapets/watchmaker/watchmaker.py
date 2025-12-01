import functools
import time
from collections import defaultdict
from concurrent.futures import as_completed, wait
from concurrent.futures.process import ProcessPoolExecutor
from typing import Dict, List, Optional, Callable

import mujoco
import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, QCoreApplication
from PyQt6.QtWidgets import QProgressDialog, QMessageBox
from mujoco import mj_step, mj_forward, MjvOption, mjv_connector, mjv_initGeom, mjtGeom, mjtRndFlag, \
    MjvScene, MjSpec

from ariel.utils.renderers import single_frame_renderer
from .config import WatchmakerConfig
from .types import Genotype, Individual
from .window import WatchmakerWindow
from ..common.controllers import RevolveCPG
from ..common.monitors import XYSpeedMonitor
from ..common.monitors.metrics_storage import EvaluationMetrics
from ..common.mujoco.state import MjState
from ..common.robot_storage import RerunnableRobot
from ..common.world_builder import make_world


class Watchmaker:
    def __init__(
            self,
            config: WatchmakerConfig,
            window: Optional[WatchmakerWindow] = None):

        self.window = window
        if self.window is not None:
            self.window.set_interactive_callback(lambda ix: self.next_generation(ix))

        self.config = config

        n_joints = len(config.body_spec.worldbody.find_all("joint"))

        self.genetic_data = Genotype.Data(
            size=RevolveCPG.compute_dimensionality(n_joints),
            rng=np.random.default_rng(config.seed),
            scale=config.mutation_scale,
            range=config.mutation_range
        )

        self.population: list[Individual] = []
        self.generation, self.evaluations = 0, 0

        self.robot_records_file = self.config.data_folder.joinpath("evolution.csv")
        self.human_records_file = self.config.data_folder.joinpath("interactions.csv")

        self._pool = ProcessPoolExecutor(max_workers=self.config.population_size - 1)

        self._world = make_world(
            config.body_spec.copy(),
            camera_zoom=.75, camera_angle=config.camera_angle,
            show_start=config.show_start
        )

        self.start_time = None

    def reset(self):
        self.generation = 0
        self.evaluations = 0
        self.population = [
            Individual(Genotype.random(self.genetic_data))
            for _ in range(self.config.population_size)
        ]

        self.start_time = time.time()

        with open(self.robot_records_file, "w") as f:
            f.write("GenID IndID ParID Speed "
                    + " ".join([f"Gene{i}" for i in range(self.genetic_data.size)])
                    + "\n")

        with open(self.human_records_file, "w") as f:
            f.write("GenID Time Precision_abs Precision_rel")

        self.evaluate(ignore_parent=False)
        self.on_new_generation()

    def on_new_generation(self):
        if self.window is not None:
            evals = f"{self.evaluations}"
            if (m := self.config.max_evaluations) is not None:
                evals += f" / {m}"
            evals += " Evaluations"
            self.window.setWindowTitle(f"CPG-Watchmaker - {self.config.body}")
            self.window.statusBar().showMessage(
                " - ".join([
                    f"Generation {self.generation}",
                    evals,
                    f"Videos at {self.config.speed_up}X speed, {self.config.duration}s"
                ]))
            self.window.on_new_generation()

        self.generation_start = time.time()

    def next_generation(self, ix) -> bool:
        # print("[kgd-debug] selection time:", time.time() - self.generation_start)

        if (me := self.config.max_evaluations) is not None and self.evaluations >= me:
            if self.window is not None:
                QMessageBox.information(
                    self.window,
                    "Evolution finished",
                    "Evaluation budget exhausted. Thanks for your work!",
                    QMessageBox.StandardButton.Ok,
                    QMessageBox.StandardButton.Ok,
                )
                exit(0)
            return False

        new_parent = (ix != 0)
        parent = self.population[ix]

        if new_parent:
            self.population[0] = parent
            self.save_champion()

        for i, ind in enumerate(self.population[1:], start=1):
            self.population[i] = parent.mutated(self.genetic_data)

        if self.generation == 0 or ix != 0:
            self.generation += 1

        if self.window is not None and new_parent:
            self.window.update_fields(parent, 0, self.generation)

        self.evaluate(ignore_parent=True)

        self.on_new_generation()

        return True

    def evaluate(self, ignore_parent=False):
        worker = functools.partial(
            self._evaluate_one,
            world_xml=self._world.spec.to_xml(),
            generation=self.generation,
            config=self.config,
            make_gif=self.window is not None
        )
        args = dict(worker=worker, ignore_parent=ignore_parent)
        if self.window is not None:
            self._evaluate_interactive(**args)
        else:
            self._evaluate_population(**args)

        with open(self.robot_records_file, "a") as f:
            for individual in self.population:
                f.write(f"{self.generation} {individual.to_string()}\n")

    def re_evaluate_champion(self, gif=True):
        _, video, fitness = self._evaluate_one(
            weights=self.population[0].genotype.data, index=0,
            world_xml=self._world.spec.to_xml(),
            generation=self.generation,
            config=self.config,
            make_gif=gif
        )
        print("Champion's performance:", fitness, "stored at", video)

    def _evaluate_population(self, worker: Callable, ignore_parent):
        for future in [
            self._pool.submit(worker, ind.genotype.data, i)
            for i, ind in enumerate(self.population[ignore_parent:])
        ]:
            ix, video, fitness = future.result()
            self.evaluations += 1

            individual = self.population[ix]

            individual.video = video
            individual.fitness = fitness

    def _evaluate_interactive(self, worker: Callable, ignore_parent):
        self.window.setEnabled(False)

        offset = int(ignore_parent)
        n = len(list(filter(None, self.population))) - offset
        progress = QProgressDialog(f"Evaluating generation {self.generation}", None, 0, n)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        progress.show()
        QCoreApplication.processEvents()

        # Parallel
        futures = [
            self._pool.submit(worker, ind.genotype.data, i)
            for i, ind in enumerate(self.population[offset:], start=offset)
        ]
        for i, future in enumerate(as_completed(futures)):
            ix, video, fitness = future.result()
            # ===

        # Sequential
        # for i, (ix, video, fitness) in enumerate([
        #         worker(ind.genotype.data, i)
        #         for i, ind in enumerate(self.population[offset:], start=offset)]):
        #     # ===

            self.evaluations += 1

            individual = self.population[ix]

            individual.video = video
            individual.fitness = fitness

            self.window.update_fields(individual, ix, self.generation)
            progress.setValue(i + 1)

            progress.show()
            QCoreApplication.processEvents()

        self.window.setEnabled(True)

    def save_champion(self):
        champion = self.population[0]
        RerunnableRobot(
            mj_spec=self._world.spec,
            brain=("RevolveCPG", champion.genotype.data),
            metrics=EvaluationMetrics(dict(xyspeed=champion.fitness)),
            misc=dict(),
            config=self.config
        ).save(self.config.data_folder.joinpath("champion.zip"))

    @classmethod
    def _evaluate_one(
            cls,
            weights: np.ndarray, index: int,
            world_xml: str, generation: int,
            make_gif: bool,
            config: WatchmakerConfig,
            visuals: MjvOption = None):

        state = MjState.from_spec(MjSpec.from_string(world_xml))
        cpg = RevolveCPG(weights, state)

        timer = Timer()
        timer.start("eval")

        visuals = visuals or cls.visual_options()

        state, model, data = state.unpacked
        mujoco.set_mjcb_control(lambda m, d: cpg(state))

        monitor = XYSpeedMonitor(f"{config.robot_name_prefix}1")
        monitor.start(state)

        if config.debug_fast:
            timer.start("step")
            mj_step(model, data, int(config.duration / model.opt.timestep))
            timer.stop("step")

            timer.start("render")
            path = config.data_folder.joinpath(f"{generation}_{index}.png")
            single_frame_renderer(
                model, data, width=config.video_size, height=config.video_size,
                camera=config.camera, save_path=path, save=True
            )
            timer.stop("render")

        else:
            path = config.data_folder.joinpath(f"{generation}_{index}.gif")
            video_framerate = 25
            frames: list[Image.Image] = []

            camera = model.camera(config.camera).id

            trajectory = [monitor.current_position.copy()] if config.show_trajectory else None

            with mujoco.Renderer(
                model,
                width=config.video_size,
                height=config.video_size,
            ) as renderer:
                substeps = int(config.speed_up / (model.opt.timestep * video_framerate))

                scene = renderer.scene
                cls.rendering_options(scene)

                mj_forward(model, data)
                renderer.update_scene(data, scene_option=visuals, camera=camera)

                timer.start("render")
                if make_gif:
                    frames.append(Image.fromarray(renderer.render()))
                timer.stop("render")

                while data.time < config.duration:
                    timer.start("step")
                    mj_step(model, data, substeps)
                    timer.stop("step")

                    timer.start("render")
                    renderer.update_scene(data, scene_option=visuals, camera=camera)

                    if trajectory is not None:
                        trajectory.append(monitor.current_position.copy())
                        for i in range(1, len(trajectory)):
                            scene.ngeom += 1
                            mjv_initGeom(scene.geoms[scene.ngeom-1], mjtGeom.mjGEOM_LINE,
                                         np.zeros(3), np.zeros(3), np.zeros(9),
                                         [1, 1, 1, 1])
                            mjv_connector(scene.geoms[scene.ngeom-1], mjtGeom.mjGEOM_LINE,
                                          .02, trajectory[i-1], trajectory[i])

                    frames.append(Image.fromarray(renderer.render()))
                    timer.stop("render")

            if make_gif:
                timer.start("gif")
                frames[0].save(
                    path, append_images=frames[1:],
                    duration=1000 / video_framerate, loop=0,
                    optimize=False
                )
                timer.stop("gif")

        # ==========

        timer.stop("eval")
        if config.timing:
            print(f"Evaluating {index=} from {generation=}")
            timer.show()
            print()

        mujoco.set_mjcb_control(None)
        monitor.stop(state)

        return index, path, monitor.value

    @staticmethod
    def visual_options():
        viz_options = MjvOption()
        viz_options.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
        viz_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        viz_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
        viz_options.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = False

        return viz_options

    @staticmethod
    def rendering_options(scene: MjvScene):
        scene.flags[mjtRndFlag.mjRND_SHADOW] = True


class Timer:
    def __init__(self):
        self.data: Dict[str, List[float]] = defaultdict(list)

    def start(self, name: str): self.data[name].append(time.time())
    def stop(self, name: str): self.data[name][-1] = (time.time() - self.data[name][-1])

    def show(self):
        for name, data in self.data.items():
            if (n := len(data)) > 1:
                avg = sum(data)
                value = f"{avg}s total ({avg / n}s average)"
            else:
                value = f"{data[0]:g}s"
            print(f"{name.capitalize():>10s}: {value}")
