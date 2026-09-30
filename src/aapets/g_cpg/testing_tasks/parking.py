
from dataclasses import dataclass
import functools
from pathlib import Path
from typing import Literal

import numpy as np
from mujoco import mj_step, mjv_initGeom, mjtGeom, mjv_connector, mju_rotVecQuat

from aapets.common import controllers

from ...common.controllers.ABCpg import ABCpg
from ...common.monitors._monitor import MonitorBase
from ...common.monitors.abcpg_handler import compute_angle, cross2d
from ...common.mujoco.callback import MjcbCallbacks
from ...common.mujoco.state import MjState
from ...common.mujoco.viewer import passive_viewer
from ...common.robot_storage import RerunnableRobot
from .config import TestingConfig
from .task import TestTask


class _ParkingTask(TestTask):
    def __init__(self, name: str, config: TestingConfig, n_places: int = 4):
        super().__init__(name=name, config=config)
        
        self.config.duration = self.config.base_duration

        self.places = n_places
        self.parking_spots = n_places * 2
        self.start_position = np.array([0, 0])

        bl = self.base_length
        self.final_targets = [
            np.array([(2*(i/n_places - .375)) * bl, j * .75 * bl])
            for j in [-1, 1] for i in range(n_places)
        ]
        # self.intermediate_targets = [.5*(self.start_position+finish) for finish in self.final_targets]

    def _modify_specs(self, specs, config: TestingConfig):
        super()._modify_specs(specs, config)
        specs.body(self.robot_name).pos[0] = 0

    def _process(self, state: MjState, record: RerunnableRobot, champion: Path):
        scores = []

        for i in range(1):#self.positions):
            monitors = dict()

            if self.config.movie or self.config.debug_viewer:
                overlay = _ParkingOverlay(self, ix=i)
                overlays = [overlay]

            if self.config.movie:
                drawers = [functools.partial(overlay._draw_parking_lot, clear=False)]
                monitors["movie-recorder"] = self._movie_recorder(champion, drawers)

            else:
                overlay = None

            brain = controllers.get(record.brain[0])(
                weights=record.brain[2], state=state, name=f"{self.config.robot_name_prefix}1",
                **record.brain[1] 
            )    
            monitors["path-follower"] = pather = _Parker(
                brain, self, overlay, debug_draw=self.config.debug_draw, ix=i)

            state, model, data = state.unpacked
            with MjcbCallbacks(state, [brain], monitors, self.config):
                if self.config.debug_viewer:
                    self.passive_viewer(state, overlays=overlays)
                else:
                    for _ in range(int(self.config.duration / model.opt.timestep)):
                        mj_step(model, data)
                        if pather.complete:
                            break

            scores.append(100 * (1 - pather.result / self.config.duration))

        return np.average(scores)


class _ParkingOverlay:
    @dataclass
    class DebugDrawData:
        pos: np.array
        quat: np.array
        alpha: float
        beta: float

    def __init__(self, task: _ParkingTask, ix: int):
        self.task = task
        self.debug_draw_data = None
        self.default_color = [0.5, 0.5, 0.5, 1.0]
        self.highlight_color = [1.0, 0.0, 0.0, 1.0]
        self.ix = ix

    def start(self, viewer, state: MjState):
        self._draw_parking_lot(viewer.user_scn, state, clear=True)
    def render(self, viewer, state: MjState): pass
    def stop(self, viewer, state: MjState): pass

    def set_debug_draw(self, data: DebugDrawData):
        self.debug_draw_data = data

    def _draw_parking_lot(self, scene, state, clear):
        ix = self.ix
        scene.ngeom = 0 if clear else scene.ngeom
        i = scene.ngeom

        def color(highlight): return self.highlight_color if highlight else self.default_color

        hn, bl = self.task.places, self.task.base_length
        ij, ik = ix % hn, -1 if (ix // hn == 0) else 1
        for j in range(hn+1):
            for k in [-1, 1]:
                mjv_initGeom(scene.geoms[i], mjtGeom.mjGEOM_LINE,
                             np.zeros(3), np.zeros(3), np.zeros(9),
                             color((j == ij or j == ij+1) and k == ik))
                mjv_connector(scene.geoms[i], mjtGeom.mjGEOM_LINE, .05,
                              [(2 * (j / hn) - 1) * bl, k*bl, 0],
                              [(2 * (j / hn) - 1) * bl, .5*k*bl, 0])
                i += 1

        for j in range(hn):
            for k in [-1, 1]:
                mjv_initGeom(scene.geoms[i], mjtGeom.mjGEOM_LINE,
                             np.zeros(3), np.zeros(3), np.zeros(9),
                             color(j == ij and k == ik))
                mjv_connector(scene.geoms[i], mjtGeom.mjGEOM_LINE, .05,
                              [(2 * (j / hn) - 1) * bl, k*bl, 0],
                              [(2 * ((j+1) / hn) - 1) * bl, k*bl, 0])
                i += 1

        for j, p in enumerate(self.task.final_targets):
            mjv_initGeom(
                scene.geoms[i],
                type=mjtGeom.mjGEOM_SPHERE,
                size=[0.05, 0, 0],
                pos=[*p, 0],
                mat=np.eye(3).flatten(),
                rgba=color(j == ix),
            )
            i += 1

        if self.debug_draw_data is not None:
            pos, back, alpha = self.debug_draw_data
        #     pos, quat = self.debug_draw_data.pos, self.debug_draw_data.quat
        #     a, b = self.debug_draw_data.alpha, self.debug_draw_data.beta
        #     h = np.array([0, 0, .1])
        #     d = b * np.array([np.cos(a), np.sin(a), 0])
        #     mju_rotVecQuat(d, d, quat)

            mjv_initGeom(
                scene.geoms[i], mjtGeom.mjGEOM_ARROW,
                np.zeros(3), np.zeros(3), np.zeros(9),
                [1, 1, 1, 1])
            
            mjv_connector(scene.geoms[i],
                          mjtGeom.mjGEOM_ARROW, .01,
                          pos, pos + back)
            i += 1

        scene.ngeom = i


class _Parker(MonitorBase):
    def __init__(self, controller: ABCpg, task: _ParkingTask, overlay: _ParkingOverlay, ix: int,
                 debug_draw: bool = False):
        super().__init__(frequency=20)
        self.task = task
        self.controller = controller
        self.overlay = overlay
        self._debug_draw = debug_draw

        self.target = self.task.final_targets[ix]

        self.proximity_threshold = .1
        self._finish_line = None

        self.half_vision = np.deg2rad(62.2) / 2

    @property
    def complete(self): return self._finish_line is not None

    @property
    def result(self): return self._finish_line if self.complete else np.inf

    def start(self, state: MjState):
        super().start(state)

        self.robot = state.data.body(self.task.robot_name)

    def _step(self, state: MjState):
        super()._step(state)

        back = np.array([0., 0., 0.])
        mju_rotVecQuat(back, np.array([-1., 0., 0.]), self.robot.xquat)

        if not self.complete:
            if np.linalg.norm(self.target - self.robot.xpos[:2]) < self.proximity_threshold:
                self._finish_line = state.time

            tgt = np.array([0., 0., 0.])
            tgt[:2] = (self.target - self.robot.xpos[:2])
            tgt[:2] /= (tgt[:2] ** 2).sum() ** .5

            angle = np.arccos(np.clip(np.dot(back[:2], tgt[:2]), -1.0, 1.0))
            if cross2d(back[:2], tgt[:2]) < 0:
                angle *= -1

            alpha = float(np.clip(angle / self.half_vision, -1, 1))
            beta = -1.0

        else:
            alpha, beta = .0, 0.0

        if self.overlay is not None and self._debug_draw:
            self.overlay.debug_draw_data = (
                self.robot.xpos, back, alpha)

        self.controller.set(alpha=alpha, beta=beta)


def prepare_tasks(config: TestingConfig):
    return [_ParkingTask(name="parking", config=config)]
