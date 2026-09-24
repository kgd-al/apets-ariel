from abc import ABC
from pathlib import Path
from turtle import back

from mujoco import MjSpec, mjtGeom, mj_step, mj_geomDistance, mjv_initGeom, mjv_connector
import numpy as np

from aapets.common import controllers
from aapets.common.controllers import ABCpg

from aapets.common.mujoco.viewer import passive_viewer
from ariel.simulation.environments import BaseWorld
from ...common.monitors._monitor import MonitorBase
from ...common.monitors.abcpg_handler import compute_angle, compute_forward, cross2d
from ...common.mujoco.callback import MjcbCallbacks
from ...common.mujoco.state import MjState
from ...common.robot_storage import RerunnableRobot
from .config import TestingConfig
from .task import TestTask


class _AvoidanceTask(TestTask):
    def __init__(self, name: str, config: TestingConfig):
        super().__init__(name=name, config=config.where(base_length=2*config.base_length))
        self.target = np.array([self.config.base_length, 0])
        self.proximity_threshold = .2

        # self.config.duration = 10
        # self.config.movie_speed = 1

    @staticmethod
    def _robot_size(specs: MjSpec, name):
        aabb = BaseWorld.get_aabb(specs, name)
        center = (aabb[0] + aabb[1]) / 2
        return np.linalg.norm(aabb[1] - center)

    def _modify_specs(self, specs: MjSpec):
        bl = self.base_length
        extent = .25 * bl
        height, depth = .5, .1
        for i, (x, y) in enumerate([(-.5*bl, 0), (0, -.5*bl), (0, +.5*bl), (.5*bl, 0)]):
            specs.worldbody.add_geom(
                name=f"wall_{i}",
                pos=(x, y, height),
                type=mjtGeom.mjGEOM_BOX,
                size=(extent, depth, height),
                rgba=[1, 1, 1, 1],
                quat=[0.7071068, 0, 0, 0.7071068],
            )

        specs.worldbody.add_geom(
            name="target", pos=[*self.target, 0], type=mjtGeom.mjGEOM_CYLINDER,
            size=[self.proximity_threshold, 0.001, 0],   # radius, half-height, unused
            rgba=[1, 1, 0, 1],
        )

        self.robot_size = self._robot_size(specs, self.config.robot_name_prefix)

    def _process(self, state: MjState, record: RerunnableRobot, champion: Path):
        monitors = dict()

        brain = controllers.get(record.brain[0])(
            weights=record.brain[2], state=state, name=f"{self.config.robot_name_prefix}1",
            **record.brain[1] 
        )    

        if self.config.debug_draw:
            overlay = _AvoidanceOverlay(self, brain)
            overlays = [overlay]
            drawers = [overlay]
        else:
            drawers = overlay = overlays = None

        if self.config.movie:
            monitors["movie-recorder"] = self._movie_recorder(champion, drawers)

        monitors["avoider-dynamics"] = dynamics = _Avoider(
            robot_name=self.robot_name, robot_size=self.robot_size,
            brain=brain, overlay=overlay, task=self)

        state, model, data = state.unpacked
        with MjcbCallbacks(state, [brain], monitors, self.config):
            if self.config.debug_viewer:
                self.config.auto_start = True
                self.config.auto_quit = True
                self.config.camera = "pretty-cam"
                self.config.settings_restore = True
                self.config.settings_save = True
                passive_viewer(state, self.config, overlays=overlays)
            else:
                for _ in range(int(self.config.duration / model.opt.timestep)):
                    mj_step(model, data)
                    if dynamics.complete:
                        break

        return 100 * (1 - dynamics.result / self.config.duration)


class _AvoidanceOverlay:
    def __init__(self, task: _AvoidanceTask, brain: ABCpg):
        self.task = task
        self.brain = brain

        self.debug_draw_data = None

    def start(self, *args, **kwargs): pass
    def stop(self, *args, **kwargs): pass

    def render(self, viewer, state):
        self(viewer.user_scn, state, clear=True)

    def __call__(self, scene, state, clear=False):
        if self.debug_draw_data is not None:
            rays, ranges, body_pos, fwd, alpha, beta = self.debug_draw_data
            i = 0 if clear else scene.ngeom
            for points in rays:
                mjv_initGeom(
                    scene.geoms[i],
                    type=mjtGeom.mjGEOM_CAPSULE,
                    size=[0.005, 0, 0],       # radius; length/pos set by connector below
                    pos=np.zeros(3), mat=np.eye(3).flatten(),
                    rgba=[1, 0, 0, 1],
                )
                mjv_connector(
                    scene.geoms[i],
                    mjtGeom.mjGEOM_CAPSULE,
                    0.005,                    # width
                    points[:3], points[3:],
                )
                i += 1

            for (radius, status) in ranges:
                mjv_initGeom(
                    scene.geoms[i],
                    type=mjtGeom.mjGEOM_CYLINDER,
                    size=[radius, 0.001, 0],   # radius, half-height, unused
                    pos=[*body_pos[:2], 0],      # (x, y, z) — same z as ground/robot plane
                    mat=np.eye(3).flatten(),   # identity: cylinder axis along world z
                    rgba=[status, 1-status, 0, .1],
                )
                i+=1

            mjv_initGeom(scene.geoms[i],
                         mjtGeom.mjGEOM_ARROW,
                         np.zeros(3), np.zeros(3), np.zeros(9),
                         [1, 1, 0, 1])
            mjv_connector(scene.geoms[i],
                          mjtGeom.mjGEOM_ARROW, .005,
                          body_pos, body_pos + fwd)
            i += 1

            mjv_initGeom(scene.geoms[i],
                        mjtGeom.mjGEOM_ARROW,
                        np.zeros(3), np.zeros(3), np.zeros(9),
                        [0, 1, 1, 1])
            orth = np.array([-fwd[1], fwd[0], 0])
            mjv_connector(scene.geoms[i],
                          mjtGeom.mjGEOM_ARROW, .005,
                          body_pos + .5 * fwd, body_pos + .5 * fwd + orth * alpha)
            i += 1

            scene.ngeom = i


class _Avoider(MonitorBase):
    def __init__(self, robot_name: str, robot_size: float, brain: ABCpg,
                 overlay: _AvoidanceOverlay, task: _AvoidanceTask):
        super().__init__(frequency=20)
        self.brain = brain
        self.overlay = overlay
        self.robot_name = robot_name
        self.robot_size = robot_size
        self.task = task

        self.target = task.target
        self.max_angle = np.deg2rad(45)

        self.evasion_threshold = 2 * self.robot_size
        self.backing_threshold = 0 * self.robot_size

        self._finish_line = None

    @property
    def complete(self): return self._finish_line is not None

    @property
    def result(self): return self._finish_line if self.complete else np.inf

    def start(self, state: MjState):
        super().start(state)

        self.robot = state.data.body(self.robot_name)
        self.walls = [g for i in range(state.model.ngeom)
                      if (g := state.data.geom(i)).name[:4] == "wall"]
        self.core = state.data.geom(self.robot_name.replace("world", "core"))
        self._complete = False

    def _step(self, state: MjState):
        super()._step(state)
        # print("_step")

        if self.overlay is not None:
            fwd = None
            too_close, backtrack = False, False
            self.overlay.debug_draw_data = None

        if not self.complete:

            alpha = 0.0
            beta = 1.0
            
            if np.linalg.norm(self.robot.xpos[:2] - self.target) < self.task.proximity_threshold:
                self._finish_line = state.time

            else:
                distmax = 5
                dists, all_points = [], []
                for wall in self.walls:
                    points = np.zeros(6)
                    d = mj_geomDistance(
                        state.model, state.data,
                        self.core.id, wall.id,
                        distmax, points
                    )
                    dists.append(d)
                    all_points.append(points)
    
                closest = np.argmin(dists)
                tgt = all_points[closest][3:5] - self.robot.xpos[:2]
                dist = np.linalg.norm(tgt)
                too_close = (dist < self.evasion_threshold)
                if too_close:
                    fwd = compute_forward(self.robot)
                    tgt /= dist

                    angle = np.arccos(np.clip(np.dot(fwd[:2], tgt[:2]), -1.0, 1.0))
                    if cross2d(fwd[:2], tgt[:2]) < 0:
                        angle *= -1
            
                    backtrack = (dist < self.backing_threshold)
                    if backtrack:  # Too close -> back, back, back
                        alpha = angle
                        beta = -1.0
                    elif abs(angle) < np.pi / 2:
                        alpha = -1 * np.sign(angle)
                        beta = 1.0
                    elif abs(angle) < 3 * np.pi / 4:
                        alpha = 0.0
                        beta = 1.0
                    else:
                        alpha = angle
                        beta = 1.0
                
                else:
                    _, _, angle = compute_angle(self.robot, self.target)
                    alpha = float(np.clip(angle / self.max_angle, -1, 1))
                    beta = 1.0
    
                if self.overlay is not None:
                    # print(f"t={state.time} d={dist}")
                    # # print(f" > robot: {self.robot.xpos[:2]}")
                    # # print(f" > wall[{closest}]: {self.walls[closest].xpos[:2]}")
                    # # print(f"   > at: {all_points[closest][:2]}")
                    # print(f" < {self.evasion_threshold}? {too_close}")
                    # print(f" < {self.backing_threshold}? {backtrack}")

                    if fwd is None:
                        fwd = compute_forward(self.robot)
                    self.overlay.debug_draw_data = (
                        all_points,
                        [(self.evasion_threshold, too_close), (self.backing_threshold, backtrack)],
                        self.robot.xpos, fwd, alpha, beta)
    
        else:
            alpha, beta = 0.0, 0.0

        self.brain.set(alpha=alpha, beta=beta)
        # print("_end_step")


def prepare_tasks(config: TestingConfig):
    return [_AvoidanceTask("obstacles", config)]
