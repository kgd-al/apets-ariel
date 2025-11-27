import functools
import time
from collections import defaultdict
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, List

import mujoco
import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, QCoreApplication
from PyQt6.QtWidgets import QProgressDialog, QMessageBox
from mujoco import mj_step, mj_forward, MjvOption, mjv_connector, mjv_initGeom, mjtGeom, mjtRndFlag, \
    MjvScene, MjSpec

from ariel.utils.renderers import single_frame_renderer
from .config import WatchmakerConfig
from .window import MainWindow, GridCell
from ..common.controllers import RevolveCPG
from ..common.monitors import XSpeedMonitor
from ..common.monitors.metrics_storage import EvaluationMetrics
from ..common.mujoco.state import MjState
from ..common.robot_storage import RerunnableRobot
from ..common.world_builder import make_world


class Genotype:
    @dataclass
    class Data:
        size: int
        rng: np.random.Generator
        scale: float

    def __init__(self, data: np.ndarray):
        self.data = data.copy()

    @classmethod
    def random(cls, data: Data) -> "Genotype":
        # return cls(.5 * np.ones(data.size))
        return cls(data.rng.uniform(-1, 1, data.size))

    def mutated(self, data: Data) -> "Genotype":
        clone = self.__class__(self.data)
        clone.data += data.rng.normal(0, data.scale, data.size)
        return clone


class Individual:
    __next_id = 0

    def __init__(self, genotype: Genotype, parent: int = -1):
        self.genotype = genotype
        self.fitness = float("nan")
        self.video = None

        self.id = self.__next_id + 1
        self.__class__.__next_id += 1

        self.parent = parent

    def __repr__(self): return f"R{self.id}"

    def mutated(self, data: Genotype.Data):
        return self.__class__(self.genotype.mutated(data), parent=self.id)

    def to_string(self):
        return (f"{self.id} {self.parent} {100*self.fitness}"
                " " + " ".join([f"{x:g}" for x in self.genotype.data]))


class Watchmaker:
    def __init__(self, window: MainWindow, config: WatchmakerConfig):
        self.window = window
        self.config = config

        n_joints = len(config.body_spec.worldbody.find_all("joint"))

        self.genetic_data = Genotype.Data(
            size=RevolveCPG.compute_dimensionality(n_joints),
            rng=np.random.default_rng(config.seed),
            scale=1
        )

        self.population: list[Individual] = []
        self.generation, self.evaluations = 0, 0

        self.robot_records_file = self.config.data_folder.joinpath("robot_records.dat")
        self.human_records_file = self.config.data_folder.joinpath("human_records.dat")

        self._pool = None
        if config.parallelism:
            self._pool = ProcessPoolExecutor(max_workers=self.config.grid_spec.population_size - 1)

        self._world = make_world(
            config.body_spec,
            camera_zoom=.75, camera_angle=config.camera_angle,
            show_start=config.show_start
        )

        self.start_time = None

        for ix, cell in enumerate(self.window.cells):
            if cell is not None:
                cell.clicked.connect(lambda _, _ix=ix: self.next_generation(_ix))

    def reset(self):
        self.generation = 0
        self.evaluations = 0
        self.population = [
            Individual(Genotype.random(self.genetic_data))
            if not self.config.grid_spec.empty_cell(i, j) else
            None
            for i, j in self.config.grid_spec.all_cells()
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

    def next_generation(self, ix):
        print(time.time() - self.generation_start)

        if (me := self.config.max_evaluations) is not None and self.evaluations >= me:
            QMessageBox.information(
                self.window,
                "Evolution finished",
                "Evaluation budget exhausted. Thanks for your work!",
                QMessageBox.StandardButton.Ok,
                QMessageBox.StandardButton.Ok,
            )
            exit(0)

        gs = self.config.grid_spec

        new_parent = (ix != gs.parent_ix)
        parent = self.population[ix]
        self.save_champion()

        if new_parent:
            self.population[gs.parent_ix] = parent

        for i, ind in enumerate(self.population):
            if ind is not None and i != gs.parent_ix:
                self.population[i] = parent.mutated(self.genetic_data)

        if self.generation == 0 or ix != gs.parent_ix:
            self.generation += 1

        if new_parent:
            self.update_fields(parent, self.window.cells[gs.parent_ix], gs.parent_ix)

        self.evaluate(ignore_parent=True)

        self.on_new_generation()

    def evaluate(self, ignore_parent=False):
        self.window.setEnabled(False)

        offset = int(ignore_parent)
        n = len(list(filter(None, self.population))) - offset
        progress = QProgressDialog(f"Evaluating generation {self.generation}", None, 0, n)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        progress.show()
        QCoreApplication.processEvents()

        camera = "apet1_tracking-cam"

        gs = self.config.grid_spec
        ind_cells = [
            (ind, cell) for (i, j), ind, cell
            in zip(gs.all_cells(), self.population, self.window.cells)
            if not gs.empty_cell(i, j) and (
                gs.ix(i, j) != gs.parent_ix or not ignore_parent
            )
        ]

        caller = functools.partial(
            self._evaluate,
            world_xml=self._world.spec.to_xml(),
            generation=self.generation,
            camera=camera,
            config=self.config
        )

        if self._pool is not None:
            futures = [
                self._pool.submit(caller, ind.genotype.data, i)
                for i, (ind, _) in enumerate(ind_cells)
            ]
            for i, future in enumerate(as_completed(futures)):
                ix, video, fitness = future.result()

                self.evaluations += 1

                individual, cell = ind_cells[ix]

                individual.video = video
                individual.fitness = fitness

                self.update_fields(individual, cell, ix)
                progress.setValue(i + 1)

                progress.show()
                QCoreApplication.processEvents()

        else:
            for i, (individual, cell) in enumerate(ind_cells):
                _, individual.video, fitness = caller(individual.genotype.data, i)
                self.evaluations += 1

                individual.fitness = fitness

                self.update_fields(individual, cell, i)
                progress.setValue(i + 1)

        self.window.setEnabled(True)

        with open(self.robot_records_file, "a") as f:
            for individual, _ in ind_cells:
                f.write(f"{self.generation} {individual.to_string()}\n")

    def save_champion(self):
        champion = self.population[self.config.grid_spec.parent_ix]
        RerunnableRobot(
            mj_spec=self._world.spec,
            brain=("RevolveCPG", champion.genotype.data),
            metrics=EvaluationMetrics(dict(xspeed=champion.fitness)),
            misc=dict(),
            config=self.config
        ).save(self.config.data_folder.joinpath("champion.zip"))

    def update_fields(self, ind: Individual, cell: GridCell, ix: int):
        items = []

        if self.config.debug_show_id:
            items.append(f"[R{ind.id}]")

        if ix == self.config.grid_spec.parent_ix and self.generation > 0:
            items.append("Parent")
        elif self.generation == 0:
            items.append(f"Robot {ix+1}")
        else:
            items.append(f"Child {ix+1}")

        if self.config.show_xspeed:
            items.append(f"{100*ind.fitness:.2f} cm/s")

        cell.update_fields(video=ind.video, desc=" ".join(items))

    @classmethod
    def _evaluate(cls,
                  weights: np.ndarray, index: int,
                  world_xml: str, generation: int,
                  camera: str,
                  config: WatchmakerConfig,
                  visuals: MjvOption = None):

        state = MjState.from_spec(MjSpec.from_string(world_xml))
        cpg = RevolveCPG(weights, state)

        timer = Timer()
        timer.start("eval")

        visuals = visuals or cls.visual_options()

        state, model, data = state.unpacked
        mujoco.set_mjcb_control(lambda m, d: cpg(state))

        monitor = XSpeedMonitor(f"{config.robot_name_prefix}1")
        monitor.start(state)

        if config.debug_fast:
            timer.start("step")
            mj_step(model, data, int(config.duration / model.opt.timestep))
            timer.stop("step")

            timer.start("render")
            path = config.data_folder.joinpath(f"{generation}_{index}.png")
            single_frame_renderer(
                model, data, width=config.video_size, height=config.video_size,
                camera=camera, save_path=path, save=True
            )
            timer.stop("render")

        else:
            path = config.data_folder.joinpath(f"{generation}_{index}.gif")
            video_framerate = 25
            frames: list[Image.Image] = []

            camera = model.camera(camera).id

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
