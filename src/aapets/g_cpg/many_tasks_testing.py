
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import dataclass
import functools
import os
from pathlib import Path
import sys
from tabnanny import check
import time
from typing import Annotated, Callable, Literal, Optional, Tuple

from mujoco import (mj_forward, mj_step, mjv_initGeom, mjtGeom, mjv_connector,
                    mju_euler2Quat, mju_rotVecQuat)
import numpy as np
import pandas as pd
import rich
from rich.progress import Progress

from aapets.common import controllers
from aapets.common.config import BaseConfig
from aapets.common.controllers import ABCpg
from aapets.common.metrics_storage import BAD, GOOD, RESET
from aapets.common.monitors._monitor import MonitorBase
from aapets.common.monitors.abcpg_handler import ABCPGHandler, compute_angle
from aapets.common.monitors.plotters.record import MovieRecorder
from aapets.common.mujoco.callback import MjcbCallbacks
from aapets.common.mujoco.state import MjState
from aapets.common.mujoco.viewer import passive_viewer
from aapets.common.robot_storage import RerunnableRobot

from aapets.g_cpg.config import Config


# Current tasks:
# - (WIP) Drone pathing
# - (?) Different durations (e.g. 60s evaluation, do they go further or just exploit the simulation?)
# - (?) Ball catching (very visual, too many parameters though)
# - (?) Obstacle avoidance


@dataclass
class Arguments(BaseConfig):
    base_duration: Annotated[
        float, "Maximum base duration of a trial." 
        " May be adjusted by an appropriate ratio for shorter/longer tasks"
    ] = 600
    base_length: Annotated[
        float, "Base unit length for the paths: circle's radius, shuttlerun's half length, slalom length"
    ] = 2

    movie: Annotated[bool, "Whether to generate videos of the various performances"] = True
    movie_speed: Annotated[float, "Speed factor for the recorded movie"] = 20

    debug_viewer: Annotated[bool, "Whether to use a viewer for debugging purposes"] = False
    debug_draw: Annotated[bool, "Whether to draw additional debugging information"] = False


class _PathTask:
    def __init__(self, name: str, args: Arguments,
                 n_checkpoints: int = 10, n_subpaths: int = None, time_scale=1):
        self.name = name
        self.args, self.time_scale = args, time_scale
        self.base_length = args.base_length

        self.n_checkpoints = n_checkpoints
        self.n_subpaths = n_subpaths or n_checkpoints

        self.checkpoints = self._sample(self.n_checkpoints)
        self.subpaths = self._sample(self.n_subpaths)

    def _sample(self, n):
        return [
            np.array([*a, 0]) if len(a) == 2 else np.array(a)
            for i in range(n)
            if (a := self.path((i+1)/n)) is not None
        ]

    @staticmethod
    def _signed(sign, name): return f"{sign:+}"[0] + name

    def __call__(self, champion: Path):
        start = time.perf_counter()
        record = RerunnableRobot.load(champion)

        robot = "apet1"

        movie_size = 960
        camera_name = "pretty-cam"

        camera = record.mj_spec.camera(camera_name)
        camera.pos[:] = [0, 0, 3 * self.args.base_length]
        mju_euler2Quat(camera.quat, [0, 0, 0], "xyz")

        state, model, data = MjState.from_spec(record.mj_spec).unpacked
        mj_forward(model, data)

        monitors = dict()

        if self.args.movie or self.args.debug_viewer:
            overlay = _PathOverlay(self)

        if self.args.movie:
            drawers = None
            if not self.args.debug_viewer:
                drawers = [functools.partial(overlay._draw_path, clear=False)]

            movie_file = champion.with_suffix(f".eval.{self.name}.mp4")
            monitors["movie-recorder"] = MovieRecorder(
                25, movie_size, movie_size,
                movie_file,
                speed_up=self.args.movie_speed,
                camera=camera_name, shadows=True,
                drawings=drawers
            )

        else:
            overlay = None

        brain = controllers.get(record.brain[0])(
            weights=record.brain[2], state=state, name=robot, **record.brain[1] 
        )    
        monitors["path-follower"] = pather = _PathFollower(
            brain, self, overlay, debug_draw=self.args.debug_draw)

        data.qpos[0] = -self.args.base_length

        self.args.duration = self.args.base_duration * self.time_scale

        with MjcbCallbacks(state, [brain], monitors, self.args):
            if self.args.debug_viewer:
                passive_viewer(state, self.args, overlays=[overlay])
            else:
                for _ in range(int(self.args.duration / model.opt.timestep)):
                    mj_step(model, data)
                    if pather.complete:
                        break

        if self.args.movie:
            if movie_file.exists():
                print(f"{GOOD}Generated {movie_file}{RESET}")
            else:
                print(f"{BAD}Failed to generate {movie_file}{RESET}")

        score = 100 * (1 - pather.result / self.args.duration)
        print(f"Evaluated {champion}: {self.name}"
            f" (score={score:.2f}%; time={state.time}s; wall time={time.perf_counter() - start:.3}s)")
        return champion, self.name, score


class _PathOverlay:
    @dataclass
    class DebugDrawData:
        pos: np.array
        quat: np.array
        alpha: float
        beta: float

    def __init__(self, task: _PathTask):
        self.task = task

        self.checkpoints = []
        self.current_checkpoint = 0

        self.default_color = [0.5, 0.5, 0.5, 1.0]
        self.highlight_color = [1.0, 0.0, 0.0, 1.0]

        self.debug_draw_data = None

    def start(self, viewer, state: MjState):
        self._draw_path(viewer.user_scn, clear=True)
    def render(self, viewer, state: MjState): pass
    def stop(self, viewer, state: MjState): pass

    def set_current_checkpoint(self, i: int):
        if len(self.checkpoints) > 0:  # If checkpoints are stored, change color.
            self.checkpoints[self.current_checkpoint].rgba = self.default_color
        self.current_checkpoint = i
        if len(self.checkpoints) > 0:
            self.checkpoints[self.current_checkpoint].rgba = self.highlight_color

    def set_debug_draw(self, data: DebugDrawData):
        self.debug_draw_data = data

    def _draw_path(self, scene, clear):
        scene.ngeom = 0 if clear else scene.ngeom
        i = scene.ngeom

        for j, p in enumerate(self.task.checkpoints):
            mjv_initGeom(
                scene.geoms[i],
                type=mjtGeom.mjGEOM_SPHERE,
                size=[0.1, 0, 0],
                pos=p,
                mat=np.eye(3).flatten(),
                rgba=self.highlight_color if (not clear and j == self.current_checkpoint) else self.default_color,
            )
            if not clear:
                self.checkpoints.append(scene.geoms[i])
            i += 1

        for p0, p1 in zip(self.task.subpaths[:-1], self.task.subpaths[1:]):
            mjv_initGeom(
                scene.geoms[i],
                type=mjtGeom.mjGEOM_LINE,
                size=np.zeros(3),
                pos=np.zeros(3), mat=np.eye(3).flatten(), rgba=self.default_color,
            )
            mjv_connector(
                scene.geoms[i], mjtGeom.mjGEOM_LINE, 3.0,
                p0, p1,
            )
            i += 1

        if self.debug_draw_data is not None:
            pos, quat = self.debug_draw_data.pos, self.debug_draw_data.quat
            a, b = self.debug_draw_data.alpha, self.debug_draw_data.beta
            h = np.array([0, 0, .1])
            d = b * np.array([np.cos(a), np.sin(a), 0])
            mju_rotVecQuat(d, d, quat)

            mjv_initGeom(
                scene.geoms[i], mjtGeom.mjGEOM_ARROW,
                np.zeros(3), np.zeros(3), np.zeros(9),
                [1, 1, 1, 1])
            
            mjv_connector(scene.geoms[i],
                          mjtGeom.mjGEOM_ARROW, .01,
                          pos + h, pos + h + d)
            i += 1

        scene.ngeom = i


class _PathFollower(MonitorBase):
    def __init__(self, controller: ABCpg, task: _PathTask, overlay: _PathOverlay, debug_draw: bool = False):
        super().__init__(frequency=20)
        self.task = task
        self.controller = controller
        self.overlay = overlay
        self._debug_draw = debug_draw

        self._current_checkpoint, self._next_checkpoint = self._set_checkpoint(0)

        self.proximity_threshold = .1
        self._finish_line = None

        self.half_vision = np.deg2rad(62.2) / 2

    def _set_checkpoint(self, i: int):
        self._current_checkpoint = i
        self._next_checkpoint = self.task.checkpoints[i]
        if self.overlay is not None:
            self.overlay.set_current_checkpoint(i)
        return self._current_checkpoint, self._next_checkpoint

    def next_checkpoint(self, time):
        if self._current_checkpoint < len(self.task.checkpoints) - 1:
            self._set_checkpoint(self._current_checkpoint + 1)
        elif self._finish_line is None:
            self._finish_line = time

    @property
    def complete(self): return self._finish_line is not None

    @property
    def result(self): return self._finish_line if self.complete else np.inf

    def start(self, state: MjState):
        super().start(state)

        self.robot = state.data.body("apet1_world")

    def _step(self, state: MjState):
        super()._step(state)

        if not self.complete:
            target = self.task.checkpoints[self._current_checkpoint]
            if np.linalg.norm(target - np.array([*self.robot.xpos[:2], 0])) < self.proximity_threshold:
                self.next_checkpoint(state.time)

            _, _, angle = compute_angle(self.robot, target)
        
            alpha = float(np.clip(angle / self.half_vision, -1, 1))
            beta = 1.0

        else:
            alpha, beta = 1, 0

        if self.overlay is not None and self._debug_draw:
            self.overlay.set_debug_draw(
                self.overlay.DebugDrawData(
                    self.robot.xpos, self.robot.xquat, 
                    alpha=alpha, beta=beta))

        self.controller.set(alpha=alpha, beta=beta)



class CircleTask(_PathTask):
    def __init__(self, args: Arguments, sign: Literal[-1, 1]):
        self.sign = sign
        super().__init__(name=self._signed(sign, "circle"), args=args,
                         n_checkpoints=10, n_subpaths=100, time_scale=1)

    def path(self, u):
        a = self.sign * 2 * np.pi * (u + .5)
        return self.base_length * np.array([np.cos(a), np.sin(a)])


class SlalomTask(_PathTask):
    def __init__(self, args: Arguments, sign: Literal[-1, 1]):
        self.sign = sign
        super().__init__(name=self._signed(sign, "slalom"), args=args,
                         n_checkpoints=10, n_subpaths=100, time_scale=1)
        
    def path(self, u):
        return np.array([(2 * u - 1) * self.base_length,
                         .5 * self.base_length * np.sin(self.sign * 2 * np.pi * u)])


class Figure8Task(_PathTask):
    def __init__(self, args: Arguments, sign: Literal[-1, 1]):
        self.sign = sign
        super().__init__(name=self._signed(sign, "figure8"), args=args,
                         n_checkpoints=10, n_subpaths=100, time_scale=1)
        
    def path(self, u):
        if u < .5:
            x = 4 * u - 1
        else:
            x = 1 - 4 * (u - .5)
        return np.array([x * self.base_length, .5 * self.base_length * np.sin(self.sign * 4 * np.pi * u)])


class ShuttlerunTask(_PathTask):
    def __init__(self, args: Arguments):
        super().__init__(name="shuttlerun", args=args, n_checkpoints=2, n_subpaths=2, time_scale=1)
        
    def path(self, u):
        return np.array([2 * self.base_length * ((2 * u if u <= .5 else 2 * (1 - u)) - .5), 0])


def prepare_tasks(args: Arguments):
    tasks = []
    for t in [CircleTask, SlalomTask, Figure8Task]:
        for sign in [-1, +1]:
            tasks.append(t(args=args, sign=sign))
    for t in [ShuttlerunTask]:
        tasks.append(t(args=args))
    return tasks


def persistent_data(champion: Path): return champion.with_suffix(".evaluation.csv")


if __name__ == "__main__":
    parser = ArgumentParser(description="Performs a suite of test on a number of robots"
                                        " to test their polyvalence")
    parser.add_argument("file", nargs="+", type=Path)
    Arguments.populate_argparser(parser)
    cli_args = parser.parse_args(namespace=Arguments())
    cli_args.pretty_print()

    n_files = len(cli_args.file)
    
    tasks = prepare_tasks(cli_args)
    n_tasks = len(tasks)

    progress_args = (
        rich.progress.SpinnerColumn(),
        *Progress.get_default_columns(),
        rich.progress.TimeElapsedColumn(),
        rich.progress.MofNCompleteColumn(),
    )
    progress_kwargs = dict(
        redirect_stdout=(cli_args.verbosity > 0),
    )
    start_time = time.perf_counter()
    with Progress(*progress_args, **progress_kwargs) as progress, \
         ProcessPoolExecutor(max_workers=os.cpu_count()-1) as executor:
        
        taskbar = progress.add_task("Evaluating...", total=n_files * n_tasks)
        futures = []
        series, needs_write = dict(), set()
        already_completed = 0

        for champion in cli_args.file:
            df_path = persistent_data(champion)
            if not df_path.exists():
                s = pd.Series(dtype=float, name="score")
                s.index.name = "name"
            else:
                s = pd.read_csv(df_path, index_col=0)
            print(s)
            series[champion] = s

            for task in tasks:
                if task.name not in s.index:
                    futures.append(executor.submit(task, champion))
                    needs_write.add(champion)
                else:
                    already_completed += 1

        progress.update(taskbar, advance=already_completed, description=f"Skipping existing {already_completed}")

        for future in as_completed(futures):
            champion, task, score = future.result()
            series[champion].loc[task] = score
            progress.update(taskbar, advance=1, description=f"{champion} / {task}: {score:.2f}%")

        progress.update(
            taskbar,
            description=f"\n{GOOD}Evaluated {n_files} champions on {n_tasks} tasks"
                        f" in {time.perf_counter() - start_time:.3f} seconds{RESET}")

        for champion in needs_write:
            series[champion].to_csv(persistent_data(champion))
