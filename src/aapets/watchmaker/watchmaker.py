import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import mujoco
import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, QCoreApplication
from PyQt6.QtWidgets import QProgressDialog, QMessageBox
from mujoco import mj_step, mj_forward, MjModel, MjData, MjvOption, mjv_connector, mjv_initGeom, mjtGeom, mjtRndFlag, \
    MjvScene

from aapets.common import RevolveCPG
from aapets.common import RerunnableRobot
from aapets.watchmaker.config import WatchmakerConfig
from aapets.common import make_world, compile_world
from aapets.watchmaker.window import MainWindow, GridCell
from ariel.utils.renderers import single_frame_renderer


class Genotype:
    @dataclass
    class Data:
        size: int
        rng: np.random.Generator

    def __init__(self, data: np.ndarray):
        self.data = data.copy()

    @classmethod
    def random(cls, data: Data) -> "Genotype":
        # return cls(.5 * np.ones(data.size))
        return cls(data.rng.uniform(-1, 1, data.size))

    def mutated(self, data: Data) -> "Genotype":
        clone = self.__class__(self.data)
        clone.data += data.rng.normal(0, 1, data.size)
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
            rng=np.random.default_rng(config.seed)
        )

        self.population: list[Individual] = []
        self.generation, self.evaluations = 0, 0

        self.records_file = self.config.data_folder.joinpath("records.dat")

        self._world = make_world(
            config.body_spec,
            camera_zoom=.75, camera_angle=config.camera_angle,
            show_start=config.show_start
        )

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

        with open(self.records_file, "w") as f:
            f.write("GenID IndID ParID Speed "
                    + " ".join([f"Gene{i}" for i in range(self.genetic_data.size)])
                    + "\n")

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
                f"Videos at {self.config.speed_up}X speed"
            ]))
        self.window.on_new_generation()

    def next_generation(self, ix):
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
        ## TODO REMOVE
        print(self.config.grid_spec)
        print(self.population)
        self.save_champion()

        self.window.setEnabled(False)

        offset = int(ignore_parent)
        n = len(list(filter(None, self.population))) - offset
        progress = QProgressDialog(f"Evaluating generation {self.generation}", None, 0, n)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        QCoreApplication.processEvents()

        camera = "apet1_tracking-cam"

        duration = self.config.duration

        gs = self.config.grid_spec
        ind_cells = [
            (ind, cell) for (i, j), ind, cell
            in zip(gs.all_cells(), self.population, self.window.cells)
            if not gs.empty_cell(i, j) and (
                gs.ix(i, j) != gs.parent_ix or not ignore_parent
            )
        ]

        for i, (individual, cell) in enumerate(ind_cells):
            model, data = compile_world(self._world)
            cpg = RevolveCPG(individual.genotype.data, model, data)

            individual.video, delta_pos = self._evaluate(
                model, data,
                cpg, self.generation, i,
                camera, self.config
            )
            self.evaluations += 1

            individual.fitness = float(delta_pos[0] / duration)

            self.update_fields(individual, cell, i)
            progress.setValue(i+1)

        self.window.setEnabled(True)

        with open(self.records_file, "a") as f:
            for individual, _ in ind_cells:
                f.write(f"{self.generation} {individual.to_string()}\n")

    def save_champion(self):
        champion = self.population[self.config.grid_spec.parent_ix]
        RerunnableRobot(
            mj_spec=self._world.spec,
            brain=champion.genotype.data,
            fitness=dict(xspeed=champion.fitness),
            infos=dict(),
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
    def _evaluate(cls, model: MjModel, data: MjData, cpg: RevolveCPG,
                  generation: int, index: int,
                  camera: str,
                  config: WatchmakerConfig,
                  visuals: MjvOption = None):

        timer = Timer()
        timer.start("eval")

        visuals = visuals or cls.visual_options()

        mujoco.set_mjcb_control(cpg.control)

        p = data.body("apet1_core").xpos
        p0 = p.copy()

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

            trajectory = [p0] if config.show_trajectory else None

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
                        trajectory.append(p.copy())
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

        return path, data.body("apet1_core").xpos - p0

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
