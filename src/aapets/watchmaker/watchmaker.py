import functools
import time
from collections import defaultdict
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Callable

import graphviz
import mujoco
import numpy as np
import pandas as pd
from PIL import Image
from PyQt6.QtCore import Qt, QCoreApplication
from PyQt6.QtWidgets import QProgressDialog, QMessageBox
from matplotlib import pyplot as plt
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
        self.step, self.generation, self.evaluations = 0, 0, 0

        self.selection_time = None

        self.robot_records_file = self.evolution_file(self.config)
        self.human_records_file = self.interaction_file(self.config)

        self._pool = ProcessPoolExecutor(max_workers=self.config.population_size - 1)

        self._world = make_world(
            config.body_spec.copy(),
            camera_zoom=.75, camera_angle=config.camera_angle,
            show_start=config.show_start
        )

        self.start_time = None

    @staticmethod
    def champion_file(config: WatchmakerConfig): return config.data_folder.joinpath("champion.zip")

    @staticmethod
    def evolution_file(config: WatchmakerConfig): return config.data_folder.joinpath("evolution.csv")

    @staticmethod
    def interaction_file(config: WatchmakerConfig): return config.data_folder.joinpath("interactions.csv")

    def reset(self):
        self.step, self.generation, self.evaluations = 0, 0, 0
        self.population = [
            Individual(Genotype.random(self.genetic_data))
            for _ in range(self.config.population_size)
        ]

        self.start_time = time.time()

        self.write_evolution_data(header=True)
        self.write_interaction_data(None, header=True)

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

        self.selection_time = time.time()

    def next_generation(self, ix) -> bool:
        # print("[kgd-debug] selection time:", time.time() - self.generation_start)

        self.write_interaction_data(ix)

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
        self.step += 1

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

        self.write_evolution_data()

    def re_evaluate_champion(self, gif=True):
        _, video, fitness = self._evaluate_one(
            weights=self.population[0].genotype.data, index=0,
            world_xml=self._world.spec.to_xml(),
            generation=self.generation,
            config=self.config,
            make_gif=gif
        )

        return self.population[0]

    def _evaluate_population(self, worker: Callable, ignore_parent):
        def process(_ix, _video, _fitness):
            self.evaluations += 1

            individual = self.population[_ix]

            individual.video = _video
            individual.fitness = _fitness

        offset = int(ignore_parent)

        if self.config.parallelism:
            for future in [
                self._pool.submit(worker, ind.genotype.data, i)
                for i, ind in enumerate(self.population[offset:], start=offset)
            ]:
                process(*future.result())

        else:
            for ix, video, fitness in [
                    worker(ind.genotype.data, i)
                    for i, ind in enumerate(self.population[offset:], start=offset)]:
                process(ix, video, fitness)

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

        def process(_i, _ix, _video, _fitness):
            self.evaluations += 1

            individual = self.population[_ix]

            individual.video = _video
            individual.fitness = _fitness

            self.window.update_fields(individual, _ix, self.generation)
            progress.setValue(_i + 1)

            progress.show()
            QCoreApplication.processEvents()

        if self.config.parallelism:
            # Parallel
            futures = [
                self._pool.submit(worker, ind.genotype.data, i)
                for i, ind in enumerate(self.population[offset:], start=offset)
            ]
            for i, future in enumerate(as_completed(futures)):
                process(i, *future.result())

        else:
            for i, (ix, video, fitness) in enumerate([
                    worker(ind.genotype.data, i)
                    for i, ind in enumerate(self.population[offset:], start=offset)]):

                process(i, ix, video, fitness)

        self.window.setEnabled(True)

    def save_champion(self):
        champion = self.population[0]
        RerunnableRobot(
            mj_spec=self._world.spec,
            brain=("RevolveCPG", champion.genotype.data),
            metrics=EvaluationMetrics(dict(xyspeed=champion.fitness)),
            misc=dict(),
            config=self.config
        ).save(self.champion_file(self.config))

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

                if make_gif:
                    timer.start("render")
                    frames.append(Image.fromarray(renderer.render()))
                    timer.stop("render")

                while data.time < config.duration:
                    timer.start("step")
                    mj_step(model, data, substeps)
                    timer.stop("step")

                    if make_gif or trajectory is not None:
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

    def write_evolution_data(self, header=False):
        with open(self.robot_records_file, "w" if header else "a") as f:
            if header:
                f.write("GenID IndID ParID Speed "
                        + " ".join([f"Gene{i}" for i in range(self.genetic_data.size)])
                        + "\n")
            else:
                for individual in self.population:
                    f.write(f"{self.generation} {individual.to_string()}\n")

    def write_interaction_data(self, selected_index, header=False):
        with open(self.human_records_file, "w" if header else "a") as f:
            if header:
                f.write("GenID Selection_time Selected_ix Selected_value"
                        " Max_value Min_value Precision_abs Precision_rel\n")
            else:
                f_select = self.population[selected_index].fitness

                f_min, f_max = np.quantile([i.fitness for i in self.population], [0, 1])
                precision_rel = (f_select - f_min) / (f_max - f_min)
                precision_abs = f_max - f_select

                f.write(" ".join(str(x) for x in [
                    self.generation,
                    time.time() - self.selection_time,
                    selected_index, f_select,
                    f_min, f_max,
                    precision_abs, precision_rel,
                    "\n"
                ]))

    @classmethod
    def do_final_plots(cls, config: WatchmakerConfig):
        e_df = pd.read_csv(cls.evolution_file(config), sep=" ")
        i_df = pd.read_csv(cls.interaction_file(config), sep=" ", index_col=False)
        # out = config.data_folder
        out = Path(".")
        ext = config.plot_extension

        print(e_df)
        print(i_df)

        cls.do_fitness_plot(e_df, i_df, out, config)
        cls.do_genealogy_plot(e_df, out, config)

    @staticmethod
    def do_fitness_plot(e_df: pd.DataFrame, i_df: pd.DataFrame, folder: Path, config: WatchmakerConfig):
        fig, ax = plt.subplots()

        ax.set_xlabel("Evaluations")
        ax.set_ylabel("Fitness")

        n = config.population_size
        index = e_df.index
        print(list(index))
        index = list((n - 1) * (1 + ((index-1) / n).astype(int)) + 1)
        # index[0] = 0
        ax.plot((n - 1) * (i_df.index+1) + 1, i_df.Selected_value, 'k--', linewidth=.2)
        artist = ax.scatter(index, e_df.Speed, c=e_df.IndID, s=2)
        cb = fig.colorbar(artist)
        cb.set_label("Recency")

        print(index)
        print(i_df.index)
        print(list((n - 1) * (i_df.index+1) + 1))

        fig.savefig(folder.joinpath(f"fitness.{config.plot_extension}"), bbox_inches="tight")

    @staticmethod
    def do_genealogy_plot(e_df: pd.DataFrame, folder: Path, config: WatchmakerConfig):
        dot = graphviz.Digraph()

        n = 9
        dot.attr('node', colorscheme=f"ylorrd{n}", style="filled")

        # TODO merge duplicate edges
        # TODO colormap: ylorrd9

        f_min, f_max = e_df.Speed.agg(["min", "max"])
        f_range = f_max - f_min

        for gen_id, ind_id, par_id, f in e_df[["GenID", "IndID", "ParID", "Speed"]].itertuples(index=False):
            dot.node(name=f"{ind_id}",
                     fillcolor=f"{int((n - 1) * (f - f_min) / f_range)+1}")
            if par_id >= 0:
                dot.edge(f"{par_id}", f"{ind_id}")

        dot.render(outfile=folder.joinpath(f"genealogy.{config.plot_extension}"), cleanup=True)


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
