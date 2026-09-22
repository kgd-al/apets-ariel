from pathlib import Path

from mujoco import MjSpec, mjtGeom, mj_step, mj_geomDistance, mjv_initGeom, mjv_connector
import numpy as np

from aapets.common import controllers
from aapets.common.controllers import ABCpg

from ariel.simulation.environments import BaseWorld
from ...common.monitors._monitor import MonitorBase
from ...common.monitors.abcpg_handler import compute_angle, compute_forward, cross2d
from ...common.mujoco.callback import MjcbCallbacks
from ...common.mujoco.state import MjState
from ...common.robot_storage import RerunnableRobot
from .config import TestingConfig
from .task import TestTask


class _AvoidanceTask(TestTask):
    BOUNDING_SPHERE_NAME = "bounding_sphere"

    def __init__(self, name: str, config: TestingConfig):
        super().__init__(name=name, config=config)

        self.target = np.array([self.config.base_length, 0])
        self.config.duration = 60
        self.config.movie_speed = 1

    def _modify_specs(self, specs: MjSpec):
        bl = self.base_length
        extent = .1 * bl
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

        aabb = BaseWorld.get_aabb(specs, self.config.robot_name_prefix)
        size = np.abs(np.array(aabb)[:,:2]).max()
        robot = specs.body(self.robot_name)
        robot.add_geom(
            name=self.BOUNDING_SPHERE_NAME,
            type=mjtGeom.mjGEOM_SPHERE,
            size=(size, 0, 0),
            density=0, 
            contype=0, conaffinity=0
        )

    def _process(self, state: MjState, record: RerunnableRobot, champion: Path):
        monitors = dict()

        if self.config.debug_draw:
            overlay = _AvoidanceOverlay(self)
            drawers = [overlay]
        else:
            drawers = overlay = None

        if self.config.movie:
            monitors["movie-recorder"] = self._movie_recorder(champion, drawers)

        brain = controllers.get(record.brain[0])(
            weights=record.brain[2], state=state, name=f"{self.config.robot_name_prefix}1",
            **record.brain[1] 
        )    
        monitors["avoider-dynamics"] = dynamics = _Avoider(
            robot_name=self.robot_name, brain=brain, overlay=overlay, task=self)

        state, model, data = state.unpacked
        with MjcbCallbacks(state, [brain], monitors, self.config):
            for _ in range(int(self.config.duration / model.opt.timestep)):
                mj_step(model, data)
                if dynamics.complete:
                    break

        return -np.inf if dynamics.complete else 100 * dynamics.result


class _AvoidanceOverlay:
    def __init__(self, task: _AvoidanceTask):
        self.task = task

        self.debug_draw_data = None

    def __call__(self, scene):
        if self.debug_draw_data is not None:
            i = scene.ngeom
            for points in self.debug_draw_data:
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
            scene.ngeom = i


class _Avoider(MonitorBase):
    def __init__(self, robot_name: str, brain: ABCpg,
                 overlay: _AvoidanceOverlay, task: _AvoidanceTask):
        super().__init__(frequency=20)
        self.brain = brain
        self.overlay = overlay
        self.robot_name = robot_name
        self.task = task

        self.proximity_threshold = .1
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
        self.sphere = state.data.geom(_AvoidanceTask.BOUNDING_SPHERE_NAME)
        self._complete = False

        if self.task.config.debug_draw:
            state.model.geom(_AvoidanceTask.BOUNDING_SPHERE_NAME).rgba = (1.0, 1.0, 1.0, 0.1)

    def _step(self, state: MjState):
        super()._step(state)

        print("_step")

        if not self.complete:

            alpha = 0.0
            beta = 1.0
            
            if np.linalg.norm(self.robot.xpos[:2] - [self.task.target]) < self.proximity_threshold:
                self._finish_line = state.time

            else:
                distmax = 5
                dists, all_points = [], []
                for wall in self.walls:
                    points = np.zeros(6)
                    d = mj_geomDistance(
                        state.model, state.data,
                        self.sphere.id, wall.id,
                        distmax, points
                    )
                    dists.append(d)
                    all_points.append(points)
    
                min_dist = min(dists)
                print(dists, min_dist)
                if min_dist < self.proximity_threshold * 10:
                    closest = np.argmin(dists)
                    fwd = compute_forward(self.robot)
                    tgt = self.walls[closest].xpos - self.robot.xpos
                    tgt /= np.linalg.norm(tgt)

                    angle = np.arccos(np.clip(np.dot(fwd[:2], tgt[:2]), -1.0, 1.0))
                    if cross2d(fwd[:2], tgt[:2]) < 0:
                        angle *= -1
            
                    alpha = 0.0
                    beta = -1.0
            
                else:
                    print("hi")
                    _, _, angle = compute_angle(self.robot, self.target)
                    print("bye")
        
                    alpha = float(np.clip(angle / self.half_vision, -1, 1))
                    beta = 1.0
    
            if self.overlay is not None:
                self.overlay.debug_draw_data = all_points
    
        else:
            alpha, beta = 0.0, 0.0

        self.brain.set(alpha=alpha, beta=beta)
        print("_end of _step")


def prepare_tasks(config: TestingConfig):
    return [_AvoidanceTask("obstacles", config)]
