from dataclasses import dataclass

import mujoco
import numpy as np
from PIL import Image
from PIL.PdfParser import IndirectObjectDef
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QProgressDialog
from mujoco import mj_step, mj_forward, MjModel, MjData, MjvOption, mjv_connector, mjv_initGeom, mjtGeom, mjtRndFlag, \
    MjvScene

from aapets.common.phenotypes.cpg import RevolveCPG
from aapets.common.robot_storage import RerunnableRobot
from aapets.watchmaker.config import WatchmakerConfig
from aapets.common.world_builder import make_world, compile_world
from aapets.watchmaker.window import MainWindow, GridCell


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
        self.generation = 0
        self.parent_ix = None

        self.records_file = self.config.data_folder.joinpath("records.dat")

        self._world = make_world(config.body_spec, camera_zoom=.75, camera_angle=config.camera_angle)

        for ix, cell in enumerate(self.window.cells):
            cell.clicked.connect(lambda _, _ix=ix: self.next_generation(_ix))

    def reset(self):
        self.generation = 0
        self.population = [
            Individual(Genotype.random(self.genetic_data))
            for _ in range(self.config.population_size)
        ]

        with open(self.records_file, "w") as f:
            f.write("GenID IndID ParID Speed "
                    + " ".join([f"Gene{i}" for i in range(self.genetic_data.size)])
                    + "\n")

        self.evaluate()
        self.on_new_generation()

    def on_new_generation(self):
        self.window.setWindowTitle(f"CPG-Watchmaker - {self.config.body} - Generation {self.generation}")
        self.window.on_new_generation()

    def next_generation(self, ix):
        parent = self.population[ix]
        self.save_champion()

        self.update_fields(parent, self.window.cells[0], 0)

        self.population = [
            parent
        ] + [
            parent.mutated(self.genetic_data)
            for _ in range(1, self.config.population_size)
        ]

        if self.generation == 0:
            self.parent_ix = 0

        if self.generation == 0 or ix != self.parent_ix:  # After first generation, upper right corner is parent
            self.generation += 1

        self.evaluate(offset=1)

        self.on_new_generation()

    def evaluate(self, offset=0):
        ## TODO REMOVE
        self.save_champion()

        self.window.setEnabled(False)

        n = len(self.population) - offset
        progress = QProgressDialog(f"Evaluating generation {self.generation}", None, 0, n)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(offset)

        camera = "apet1_tracking-cam"

        duration = self.config.duration

        for i, (individual, cell) in enumerate(zip(self.population[offset:],
                                                   self.window.cells[offset:])):
            model, data = compile_world(self._world)
            cpg = RevolveCPG(individual.genotype.data, model, data)

            individual.video, delta_pos = self._evaluate(
                model, data,
                cpg, self.generation, i,
                camera, self.config
            )

            individual.fitness = float(delta_pos[0] / duration)

            self.update_fields(individual, cell, i)
            progress.setValue(i+1)

        self.window.setEnabled(True)

        with open(self.records_file, "a") as f:
            for individual in self.population[offset:]:
                f.write(f"{self.generation} {individual.to_string()}\n")

    def save_champion(self):
        champion = self.population[0]
        RerunnableRobot(
            mj_spec=self._world.spec,
            brain=champion.genotype.data,
            fitness=dict(xspeed=champion.fitness),
            infos=dict(),
            config=self.config
        ).save(self.config.data_folder.joinpath("champion.zip"))

    def update_fields(self, ind: Individual, cell: GridCell, ix: int):
        desc = f"[R{ind.id}] "
        if ix == self.parent_ix:
            desc += "Parent"
        elif self.generation == 0:
            desc += f"Robot {ix}"
        else:
            desc += f"Child {ix}"
        # desc += f"{100*ind.fitness:.2f} cm/s"
        cell.update_fields(video=ind.video, desc=desc)

    @classmethod
    def _evaluate(cls, model: MjModel, data: MjData, cpg: RevolveCPG,
                  generation: int, index: int,
                  camera: str,
                  config: WatchmakerConfig,
                  visuals: MjvOption = None):

        visuals = visuals or cls.visual_options()

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
            cls.rendering_options(scene)

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
