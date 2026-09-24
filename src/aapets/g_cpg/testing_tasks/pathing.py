
from dataclasses import dataclass
import functools
from pathlib import Path
from typing import Literal

import numpy as np
from mujoco import mj_step, mjv_initGeom, mjtGeom, mjv_connector, mju_rotVecQuat

from aapets.common import controllers

from ...common.controllers.ABCpg import ABCpg
from ...common.monitors._monitor import MonitorBase
from ...common.monitors.abcpg_handler import compute_angle
from ...common.mujoco.callback import MjcbCallbacks
from ...common.mujoco.state import MjState
from ...common.mujoco.viewer import passive_viewer
from ...common.robot_storage import RerunnableRobot
from .config import TestingConfig
from .task import TestTask


class _PathTask(TestTask):
    def __init__(self, name: str, config: TestingConfig,
                 n_checkpoints: int = 10, n_subpaths: int = None, time_scale=1):
        super().__init__(name=name, config=config)
        
        self.time_scale = time_scale
        self.config.duration = self.config.base_duration * self.time_scale

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

    def _process(self, state: MjState, record: RerunnableRobot, champion: Path):
        monitors = dict()

        if self.config.movie or self.config.debug_viewer:
            overlay = _PathOverlay(self)

        if self.config.movie:
            drawers = None
            if not self.config.debug_viewer:
                drawers = [functools.partial(overlay._draw_path, clear=False)]
            monitors["movie-recorder"] = self._movie_recorder(champion, drawers)

        else:
            overlay = None

        brain = controllers.get(record.brain[0])(
            weights=record.brain[2], state=state, name=f"{self.config.robot_name_prefix}1",
            **record.brain[1] 
        )    
        monitors["path-follower"] = pather = _PathFollower(
            brain, self, overlay, debug_draw=self.config.debug_draw)

        state, model, data = state.unpacked
        with MjcbCallbacks(state, [brain], monitors, self.config):
            if self.config.debug_viewer:
                passive_viewer(state, self.config, overlays=[overlay])
            else:
                for _ in range(int(self.config.duration / model.opt.timestep)):
                    mj_step(model, data)
                    if pather.complete:
                        break

        return 100 * (1 - pather.result / self.config.duration)


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
        self._draw_path(viewer.user_scn, state, clear=True)
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

    def _draw_path(self, scene, state, clear):
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
            alpha, beta = .0, 0.0

        if self.overlay is not None and self._debug_draw:
            self.overlay.set_debug_draw(
                self.overlay.DebugDrawData(
                    self.robot.xpos, self.robot.xquat, 
                    alpha=alpha, beta=beta))

        self.controller.set(alpha=alpha, beta=beta)



class CircleTask(_PathTask):
    def __init__(self, config: TestingConfig, sign: Literal[-1, 1]):
        self.sign = sign
        super().__init__(name=self._signed(sign, "circle"), config=config,
                         n_checkpoints=10, n_subpaths=100, time_scale=1)

    def path(self, u):
        a = self.sign * 2 * np.pi * (u + .5)
        return self.base_length * np.array([np.cos(a), np.sin(a)])


class Figure8Task(_PathTask):
    def __init__(self, config: TestingConfig, sign: Literal[-1, 1]):
        self.sign = sign
        super().__init__(name=self._signed(sign, "figure8"), config=config,
                         n_checkpoints=10, n_subpaths=100, time_scale=1)
        
    def path(self, u):
        if u < .5:
            x = 4 * u - 1
        else:
            x = 1 - 4 * (u - .5)
        return np.array([x * self.base_length, .5 * self.base_length * np.sin(self.sign * 4 * np.pi * u)])


class ShuttlerunTask(_PathTask):
    def __init__(self, config: TestingConfig):
        super().__init__(name="shuttlerun", config=config, n_checkpoints=2, n_subpaths=2, time_scale=1)
        
    def path(self, u):
        return np.array([2 * self.base_length * ((2 * u if u <= .5 else 2 * (1 - u)) - .5), 0])


def prepare_tasks(config: TestingConfig):
    tasks = []
    for t in [CircleTask, Figure8Task]:
        for sign in [-1, +1]:
            tasks.append(t(config=config, sign=sign))
    for t in [ShuttlerunTask]:
        tasks.append(t(config=config))
    return tasks
