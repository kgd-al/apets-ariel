from dataclasses import dataclass

import mujoco
import numpy as np
from PIL import Image
from PIL.ImageQt import QPixmap, ImageQt
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QMovie
from PyQt6.QtWidgets import QProgressDialog
from mujoco import mj_step

from aapets.common.phenotypes.cpg import RevolveCPG
from aapets.watchmaker.config import WatchmakerConfig
from aapets.watchmaker.evaluation import make_world, compile_world
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
    def __init__(self, genotype: Genotype):
        self.genotype = genotype
        self.fitness = float("nan")
        self.video = None

    def mutated(self, data: Genotype.Data):
        return self.__class__(self.genotype.mutated(data))


class Watchmaker:
    def __init__(self, window: MainWindow, config: WatchmakerConfig):
        self.window = window
        self.config = config

        n_joints = len(config.body_spec.worldbody.find_all("joint"))

        self.genetic_data = Genotype.Data(
            size=RevolveCPG.compute_dimensionality(n_joints),
            rng=np.random.default_rng(config.seed)
        )

        self.population = None
        self.generation = 0

        self._world = make_world(config.body_spec)

        for ix, cell in enumerate(self.window.cells):
            cell.clicked.connect(lambda _, _ix=ix: self.next_generation(_ix))

    def reset(self):
        self.population = [
            Individual(Genotype.random(self.genetic_data))
            for _ in range(self.config.population_size)
        ]
        self.evaluate()
        self.update_window_name()

    def update_window_name(self):
        self.window.setWindowTitle(f"CPG-Watchmaker - {self.config.body} - Generation {self.generation}")

    def next_generation(self, ix):
        print(f"cell[0] <- Individual[{ix}] with fitness {self.population[ix].fitness}")
        self.set_cell(self.population[ix], self.window.cells[0])

        parent = self.population[ix]
        self.population = [
            parent
        ] + [
            parent.mutated(self.genetic_data)
            for _ in range(1, self.config.population_size)
        ]

        self.evaluate(1)

        if self.generation == 0 or ix != 0:  # After first generation, upper right corner is parent
            self.generation += 1

        self.update_window_name()
        self.window.setEnabled(True)

    @staticmethod
    def set_cell(individual: Individual, cell: GridCell):
        # cell.img = individual.video
        # cell.qt_img = ImageQt(cell.img)
        # cell.qt_pix = QPixmap.fromImage(cell.qt_img)
        # cell.viewer.setPixmap(cell.qt_pix)

        # cell.viewer.setPixmap(QPixmap.fromImage(ImageQt(img)))

        cell.viewer.setMovie(individual.video)
        cell.label.setText(str(individual.fitness))
        individual.video.start()

    def evaluate(self, offset=0):
        n = len(self.population) - offset
        progress = QProgressDialog("Evaluating new generation", "Abort", 0, n)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(200)

        camera = "apet1_tracking-cam"

        viz_options = mujoco.MjvOption()
        # viz_options.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
        # viz_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        # viz_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = False
        # viz_options.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = False

        for i, (individual, cell) in enumerate(zip(self.population[offset:], self.window.cells[offset:])):
            model, data = compile_world(self._world)
            cpg = RevolveCPG(individual.genotype.data, model, data)

            p0 = data.body("apet1_core").xpos.copy()

            mujoco.set_mjcb_control(cpg.control)
            if False:
                mj_step(model, data, int(self.config.duration / model.opt.timestep))

                individual.video = single_frame_renderer(
                    model, data, width=self.config.video_size, height=self.config.video_size,
                    camera=camera,
                )
            else:
                video_framerate = 25
                frames = []

                camera = model.camera(camera).id

                with mujoco.Renderer(
                        model,
                        width=self.config.video_size,
                        height=self.config.video_size,
                ) as renderer:
                    step = int(1 / (model.opt.timestep * video_framerate))
                    renderer.update_scene(
                        data,
                        scene_option=viz_options,
                        camera=camera,
                    )

                    while data.time < self.config.duration:
                        mj_step(model, data, step)

                        # Save frame
                        frames.append(Image.fromarray(renderer.render()))

                path = self.config.data_folder.joinpath(f"{self.generation}_{i}.gif")
                frames[0].save(
                    fp=path, format="GIF", append_images=frames[1:],
                    save_all=True, duration=self.config.duration, loop=1
                )
                individual.video = QMovie(str(path))
                print(len(frames), individual.video.frameCount())

            mujoco.set_mjcb_control(None)

            d = data.body("apet1_core").xpos - p0

            individual.fitness = np.sqrt(np.sum(d[:2]**2))

            self.set_cell(individual, cell)
            progress.setValue(i)

        progress.setValue(n)
