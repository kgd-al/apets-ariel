
from dataclasses import dataclass
import functools
import itertools
from pathlib import Path

import numpy as np
from mujoco import mj_step, mjv_initGeom, mjtGeom, mjv_connector, mju_rotVecQuat

from aapets.common import controllers
from aapets.common.monitors.plotters.record import MovieRecorder

from ...common.controllers.ABCpg import ABCpg
from ...common.monitors._monitor import MonitorBase
from ...common.monitors.abcpg_handler import cross2d
from ...common.mujoco.callback import MjcbCallbacks
from ...common.mujoco.state import MjState
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
        self.intermediate_targets = [
            np.array([
                1.25 * finish[0],
                (2 * self.start_position[1] + finish[1]) / 3
            ])
            for finish in self.final_targets
        ]

    def _modify_specs(self, specs, config: TestingConfig):
        super()._modify_specs(specs, config)
        specs.body(self.robot_name).pos[0] = 0

    def _process(self, state: MjState, record: RerunnableRobot, champion: Path):
        scores = []
        movies = []

        for i in range(self.parking_spots):
            state.reset()
            monitors = dict()

            if self.config.movie or self.config.debug_viewer:
                overlay = _ParkingOverlay(self, ix=i)
                overlays = [overlay]

            if self.config.movie:
                drawers = [functools.partial(overlay._draw_parking_lot, clear=False)]
                monitors["movie-recorder"] = movie = self._movie_recorder(champion, drawers)
                movies.append(movie)

            else:
                overlay = None

            brain = controllers.get(record.brain[0])(
                weights=record.brain[2], state=state, name=f"{self.config.robot_name_prefix}1",
                **record.brain[1] 
            )    
            monitors["path-follower"] = parker = _Parker(
                brain, self, overlay, debug_draw=self.config.debug_draw, ix=i)

            state, model, data = state.unpacked
            with MjcbCallbacks(state, [brain], monitors, self.config):
                if self.config.debug_viewer:
                    self.passive_viewer(state, overlays=overlays)
                else:
                    for _ in range(int(self.config.duration / model.opt.timestep)):
                        mj_step(model, data)
                        if parker.complete:
                            break

            scores.append(parker.result)

        if movies:
            MovieRecorder.write(
                self._movie_file(champion),
                list(itertools.chain(*[m.images for m in movies])),
                25, (self.config.movie_size, self.config.movie_size),
                self.config.movie_speed
            )

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

        c = 5
        hn, bl = self.task.places, self.task.base_length
        ij, ik = ix % hn, -1 if (ix // hn == 0) else 1
        for j in range(hn+1):
            for k in [-1, 1]:
                mjv_initGeom(scene.geoms[i], mjtGeom.mjGEOM_LINE,
                             np.zeros(3), np.zeros(3), np.zeros(9),
                             color((j == ij or j == ij+1) and k == ik))
                mjv_connector(scene.geoms[i], mjtGeom.mjGEOM_LINE, c,
                              [(2 * (j / hn) - 1) * bl, k*bl, 0],
                              [(2 * (j / hn) - 1) * bl, .5*k*bl, 0])
                i += 1

        for j in range(hn):
            for k in [-1, 1]:
                mjv_initGeom(scene.geoms[i], mjtGeom.mjGEOM_LINE,
                             np.zeros(3), np.zeros(3), np.zeros(9),
                             color(j == ij and k == ik))
                mjv_connector(scene.geoms[i], mjtGeom.mjGEOM_LINE, c,
                              [(2 * (j / hn) - 1) * bl, k*bl, 0],
                              [(2 * ((j+1) / hn) - 1) * bl, k*bl, 0])
                i += 1

        if self.debug_draw_data is not None:
            pos, vec, dir, alpha = self.debug_draw_data

            for j, p in enumerate(self.task.final_targets):
                mjv_initGeom(
                    scene.geoms[i],
                    type=mjtGeom.mjGEOM_SPHERE,
                    size=[0.05, 0, 0],
                    pos=[*p, 0],
                    mat=np.eye(3).flatten(),
                    rgba=color(dir == -1 and j == ix),
                )
                i += 1

            for j, p in enumerate(self.task.intermediate_targets):
                mjv_initGeom(
                    scene.geoms[i],
                    type=mjtGeom.mjGEOM_SPHERE,
                    size=[0.05, 0, 0],
                    pos=[*p, 0],
                    mat=np.eye(3).flatten(),
                    rgba=color(dir == 1 and j == ix),
                )
                i += 1

            mjv_initGeom(
                scene.geoms[i], mjtGeom.mjGEOM_ARROW,
                np.zeros(3), np.zeros(3), np.zeros(9),
                [1, 1, 1, 1])
            
            mjv_connector(scene.geoms[i],
                          mjtGeom.mjGEOM_ARROW, .01,
                          pos, pos + vec)
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

        self.ix = ix
        self.target = self.task.intermediate_targets[ix]
        self.dir = 1.0

        self.proximity_threshold, self.max_angle = .1, np.deg2rad(90)
        self._finish_line, self._final_angle = None, None

        self.half_vision = np.deg2rad(62.2) / 2

    @property
    def complete(self): return self._finish_line is not None

    @property
    def result(self):
        if not self.complete or self._final_angle is None:
            return -np.inf
        else:
            return 100 * (1 - np.clip(abs(self._final_angle) / self.max_angle, 0, 1))

    def start(self, state: MjState):
        super().start(state)

        self.robot = state.data.body(self.task.robot_name)

    def _step(self, state: MjState):
        super()._step(state)
        alpha, beta = 0.0, 0.0

        if not self.complete:
            vec = np.array([0., 0., 0.])
            tgt = np.array([0., 0., 0.])

            xy = self.robot.xpos[:2]
            if any(abs(v) >= 1.1 * self.task.base_length for v in xy):
                self._finish_line = state.time

            elif np.linalg.norm(self.target - xy) < self.proximity_threshold:
                if self.dir == 1.0:
                    self.target = self.task.final_targets[self.ix]
                    self.dir = -1.0
                else:
                    self._finish_line = state.time

                    mju_rotVecQuat(vec, np.array([self.dir, 0., 0.]), self.robot.xquat)
                    tgt = np.array([0, np.sign(self.target[1]), 0])
                    
                    self._final_angle = np.arccos(np.clip(np.dot(vec[:2], tgt[:2]), -1.0, 1.0))
                    # print(f"final angle: cos-1({vec[:2]} . {tgt[:2]}) = {np.rad2deg(self._final_angle)}")

            else:
                mju_rotVecQuat(vec, np.array([self.dir, 0., 0.]), self.robot.xquat)

                tgt[:2] = (self.target - xy)
                tgt[:2] /= (tgt[:2] ** 2).sum() ** .5

                angle = np.arccos(np.clip(np.dot(vec[:2], tgt[:2]), -1.0, 1.0))
                if cross2d(vec[:2], tgt[:2]) < 0:
                    angle *= -1

                alpha = float(np.clip(angle / self.half_vision, -1, 1))
                beta = self.dir

                if self.overlay is not None and self._debug_draw:
                    self.overlay.debug_draw_data = (
                        self.robot.xpos, vec, self.dir, alpha)

        self.controller.set(alpha=alpha, beta=beta)


def prepare_tasks(config: TestingConfig):
    return [_ParkingTask(name="parking", config=config)]
