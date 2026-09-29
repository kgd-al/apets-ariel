
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mujoco import mj_step, mjv_initGeom, mjtGeom, mjv_connector, mju_rotVecQuat, MjSpec

from ...common import controllers
from ...common.controllers.ABCpg import ABCpg
from ...common.monitors._monitor import MonitorBase
from ...common.monitors.abcpg_handler import compute_angle
from ...common.mujoco.callback import MjcbCallbacks
from ...common.mujoco.state import MjState
from ...common.robot_storage import RerunnableRobot
from .config import TestingConfig
from .task import TestTask


class _CarryingTask(TestTask):
    BOX_NAME = "box"

    def __init__(self, name: str, config: TestingConfig):
        super().__init__(name=name, config=config)
        
        self.config.duration = self.config.base_duration
        self.target = [self.base_length, 0]
        self.proximity_threshold = .1

        self.minimum_height = None

    def _modify_specs(self, specs: MjSpec, config: TestingConfig):
        super()._modify_specs(specs, config)

        box_size = .075
        box_height = .15

        self.minimum_height = box_height

        box = specs.worldbody.add_body(
            name=self.BOX_NAME,
            pos=[-self.base_length, 0, box_height + box_size + 0.001],
        )
        box.add_geom(
            name=self.BOX_NAME,
            type=mjtGeom.mjGEOM_BOX,
            size=(box_size, box_size, box_size),
            rgba=[1, 1, 1, .5],
            density=1
        )
        box.add_freejoint()

        specs.worldbody.add_geom(
            name="target", pos=[*self.target, 0], type=mjtGeom.mjGEOM_CYLINDER,
            size=[self.proximity_threshold, 0.001, 0],   # radius, half-height, unused
            rgba=[1, 1, 0, 1],
        )

    def _process(self, state: MjState, record: RerunnableRobot, champion: Path):
        monitors = dict()

        overlay, overlays = None, []
        if self.config.debug_viewer:
            overlay = _CarryingOverlay(self)
            overlays = [overlay]

        if self.config.movie:
            monitors["movie-recorder"] = self._movie_recorder(champion)

        brain = controllers.get(record.brain[0])(
            weights=record.brain[2], state=state, name=f"{self.config.robot_name_prefix}1",
            **record.brain[1] 
        )    
        monitors["carrier"] = carrier = _Carrier(
            brain, self, overlay, debug_draw=self.config.debug_draw)

        state, model, data = state.unpacked
        with MjcbCallbacks(state, [brain], monitors, self.config):
            if self.config.debug_viewer:
                self.passive_viewer(state, overlays=overlays)
            else:
                for _ in range(int(self.config.duration / model.opt.timestep)):
                    mj_step(model, data)
                    if carrier.complete:
                        break

        return 100 * (1 - carrier.result)  # Invert normalized distance and re-scale


class _CarryingOverlay:
    @dataclass
    class DebugDrawData:
        pos: np.array
        quat: np.array
        alpha: float
        beta: float

    def __init__(self, task: _CarryingTask):
        self.task = task

        self.debug_draw_data = None

    def start(self, viewer, state: MjState): pass
    def stop(self, viewer, state: MjState): pass

    def render(self, viewer, state: MjState):
        scene = viewer.user_scn
        scene.ngeom = 0
        i = scene.ngeom

        if self.debug_draw_data is not None:
            body_pos, body_up, box_pos, box_up, angle, beta = self.debug_draw_data
            body_end, box_end = body_pos + body_up, box_pos + box_up
            
            mjv_initGeom(scene.geoms[i],
                         mjtGeom.mjGEOM_ARROW,
                         np.zeros(3), np.zeros(3), np.zeros(9),
                         [1, 1, 0, 1])
            mjv_connector(scene.geoms[i],
                          mjtGeom.mjGEOM_ARROW, .005,
                          body_pos, body_end)
            i += 1

            mjv_initGeom(scene.geoms[i],
                         mjtGeom.mjGEOM_ARROW,
                         np.zeros(3), np.zeros(3), np.zeros(9),
                         [0, 1, 1, 1])
            mjv_connector(scene.geoms[i],
                          mjtGeom.mjGEOM_ARROW, .005,
                          box_pos, box_end)
            i += 1

            if not np.isclose(angle, np.pi / 2):
                body_end, box_end = body_pos + .5 * body_up, box_pos + .5 * box_up
                offset = .5 * (box_end - body_end)
                mjv_initGeom(scene.geoms[i],
                             mjtGeom.mjGEOM_ARROW2,
                             np.zeros(3), np.zeros(3), np.zeros(9),
                             [1, 0, 1, 1])
                mjv_connector(scene.geoms[i],
                              mjtGeom.mjGEOM_ARROW2, .005,
                              body_end + offset,
                              box_end + offset)
                scene.geoms[i].label = f"angle: {np.rad2deg(angle):.3g}\nbeta: {beta:.3g}"
            i += 1

            scene.ngeom = i


class _Carrier(MonitorBase):
    def __init__(self, controller: ABCpg, task: _CarryingTask, overlay: _CarryingOverlay, debug_draw: bool = False):
        super().__init__(frequency=20)
        self.task = task
        self.controller = controller
        self.overlay = overlay
        self._debug_draw = debug_draw

        self._finish_line, self._dropped = None, None

        self.target = np.array([self.task.config.base_length, 0])
        self.half_vision = np.deg2rad(62.2) / 2
        self.max_v_angle = np.deg2rad(15)

    @property
    def complete(self): return (self._finish_line is not None or self._dropped is not None)

    @property
    def result(self):
        """ Returns the normalized box-target distance"""
        if self._finish_line is not None:
            print("Perfect score")
            return 0
        else:
            print("Distance score:", np.linalg.norm(self.target - self.box.xpos[:2]), (2 * self.task.base_length))
            return min(1, np.linalg.norm(self.target - self.box.xpos[:2]) / (2 * self.task.base_length))

    def start(self, state: MjState):
        super().start(state)

        self.robot = state.data.body("apet1_world")
        self.box = state.data.body(self.task.BOX_NAME)

    def _step(self, state: MjState):
        super()._step(state)

        if not self.complete:
            if self.box.xpos[2] < self.task.minimum_height:
                self._dropped = state.time

            elif np.linalg.norm(self.target - self.box.xpos[:2]) < self.task.proximity_threshold:
                self._finish_line = state.time

            up = np.array([0., 0., 1.])
            robot_up, box_up = up.copy(), up.copy()
            mju_rotVecQuat(robot_up, up, self.robot.xquat)
            mju_rotVecQuat(box_up, up, self.box.xquat)
            v_angle = max(angle_between(robot_up, box_up), angle_between(box_up, up))

            _, _, angle = compute_angle(self.robot, self.target)
        
            alpha = float(np.clip(angle / self.half_vision, -1, 1))
            beta = 1 - float(np.clip(v_angle / self.max_v_angle, 0, 1))

        else:
            alpha, beta = .0, 0.0

        if not self.complete and self.overlay is not None and self._debug_draw:
            self.overlay.debug_draw_data = (
                self.robot.xpos, robot_up, self.box.xpos, box_up, v_angle, beta)

        self.controller.set(alpha=alpha, beta=beta)


def angle_between(lhs, rhs):
    lhs, rhs = np.asarray(lhs), np.asarray(rhs)
    return np.arctan2(np.linalg.norm(np.cross(lhs, rhs)), np.dot(lhs, rhs))


def prepare_tasks(config: TestingConfig):
    return [_CarryingTask(name="carrying", config=config)]
