import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageSequence
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QProgressDialog
from mujoco import mj_step, mj_forward, MjModel, MjData, MjvOption, MjvGeom, mjv_connector, mjv_initGeom, mjtGeom

from aapets.common.phenotypes.cpg import RevolveCPG
from aapets.watchmaker.config import WatchmakerConfig
from aapets.watchmaker.evaluation import make_world, compile_world
from aapets.watchmaker.window import MainWindow
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
    def __init__(self, genotype: Genotype):
        self.genotype = genotype
        self.fitness = float("nan")
        self.video = None

    def mutated(self, data: Genotype.Data):
        return self.__class__(self.genotype.mutated(data))

    def fields(self):
        return dict(video=self.video, fitness=self.fitness)


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
        self.generation = 0

        self._world = make_world(config.body_spec, camera_zoom=.75)

        for ix, cell in enumerate(self.window.cells):
            cell.clicked.connect(lambda _, _ix=ix: self.next_generation(_ix))

    def reset(self):
        self.population = [
            Individual(Genotype.random(self.genetic_data))
            for _ in range(self.config.population_size)
        ]
        self.evaluate()
        self.on_new_generation()

    def on_new_generation(self):
        self.window.setWindowTitle(f"CPG-Watchmaker - {self.config.body} - Generation {self.generation}")
        self.window.on_new_generation()

    def next_generation(self, ix):
        parent = self.population[ix]

        self.window.cells[0].update_fields(**parent.fields())

        self.population = [
            parent
        ] + [
            parent.mutated(self.genetic_data)
            for _ in range(1, self.config.population_size)
        ]

        if self.generation == 0 or ix != 0:  # After first generation, upper right corner is parent
            self.generation += 1

        self.evaluate(offset=1)

        self.on_new_generation()

    def evaluate(self, offset=0):
        self.window.setEnabled(False)

        n = len(self.population) - offset
        progress = QProgressDialog(f"Evaluating generation {self.generation}", None, 0, n)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(offset)

        camera = "apet1_tracking-cam"

        duration = self.config.duration

        # Enable joint visualization option:
        viz_options = MjvOption()
        viz_options.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
        viz_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        viz_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
        viz_options.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = False

        for i, (individual, cell) in enumerate(zip(self.population[offset:],
                                                   self.window.cells[offset:])):
            model, data = compile_world(self._world)
            cpg = RevolveCPG(individual.genotype.data, model, data)

            individual.video, delta_pos = self._evaluate(
                model, data,
                cpg, self.generation, i,
                camera, viz_options, self.config
            )

            individual.fitness = delta_pos[0] / duration

            cell.update_fields(**individual.fields())
            progress.setValue(i+1)

        self.window.setEnabled(True)

    @classmethod
    def _evaluate(cls, model: MjModel, data: MjData, cpg: RevolveCPG,
                  generation: int, index: int,
                  camera: str, visuals: MjvOption,
                  config: WatchmakerConfig):

        mujoco.set_mjcb_control(cpg.control)

        # ==========
        # mj_step(model, data, int(config.duration / model.opt.timestep))
        # path = config.data_folder.joinpath(f"{generation}_{index}.png")
        # single_frame_renderer(
        #     model, data, width=config.video_size, height=config.video_size,
        #     camera=camera, save_path=path, save=True
        # )
        # ==========
        path = config.data_folder.joinpath(f"{generation}_{index}.gif")
        video_framerate = 25
        frames: list[Image.Image] = []

        camera = model.camera(camera).id

        p = data.body("apet1_core").xpos
        p0 = p.copy()
        trajectory = [p0]

        with mujoco.Renderer(
            model,
            width=config.video_size,
            height=config.video_size,
        ) as renderer:
            substeps = int(1 / (model.opt.timestep * video_framerate))

            scene = renderer.scene

            mj_forward(model, data)
            renderer.update_scene(data, scene_option=visuals, camera=camera)

            frames.append(Image.fromarray(renderer.render()))

            while data.time < config.duration:
                mj_step(model, data, substeps)
                renderer.update_scene(data, scene_option=visuals, camera=camera)

                trajectory.append(p.copy())
                for i in range(1, len(trajectory)):
                    scene.ngeom += 1
                    mjv_initGeom(scene.geoms[scene.ngeom-1], mjtGeom.mjGEOM_LINE,
                                 np.zeros(3), np.zeros(3), np.zeros(9),
                                 [1, 1, 1, 1])
                    mjv_connector(scene.geoms[scene.ngeom-1], mjtGeom.mjGEOM_LINE,
                                  .02, trajectory[i-1], trajectory[i])

                frames.append(Image.fromarray(renderer.render()))

        frames[0].save(
            path, append_images=frames[1:],
            duration=1000 / video_framerate, loop=0,
            optimize=False
        )
        # ==========

        mujoco.set_mjcb_control(None)

        return path, data.body("apet1_core").xpos - p0
