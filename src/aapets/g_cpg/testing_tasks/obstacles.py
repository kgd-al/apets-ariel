from pathlib import Path

from mujoco import MjSpec, mjtGeom, mj_step
import numpy as np

from aapets.common import controllers
from aapets.common.controllers import ABCpg

from ...common.monitors._monitor import MonitorBase
from ...common.mujoco.callback import MjcbCallbacks
from ...common.mujoco.state import MjState
from ...common.robot_storage import RerunnableRobot
from ...fetch.dynamics.base import add_ball, add_eyes, add_mouth
from ...fetch.sm_fetcher import FetcherCPG
from ...fetch.types import FetchTaskObjects
from .config import TestingConfig
from .task import TestTask


class _AvoidanceTask(TestTask):
    def __init__(self, name: str, config: TestingConfig):
        super().__init__(name=name, config=config)

    def _modify_specs(self, specs: MjSpec):
        bl = self.base_length
        extent = .5 * bl
        height, depth = .5, .1
        for i, (x, y) in enumerate([(-.5*bl, 0), (0, -.5*bl), (0, +.5*bl), (.5*bl, 0)]):
            specs.worldbody.add_geom(
                name=f"wall_{i}",
                pos=(extent * x, extent * y, height),
                type=mjtGeom.mjGEOM_BOX,
                size=(extent, depth, height),
                rgba=[1, 1, 1, 1],
            )

    def _process(self, state: MjState, record: RerunnableRobot, champion: Path):
        monitors = dict()

        if self.config.movie:
            drawers = None
            monitors["movie-recorder"] = self._movie_recorder(champion, drawers)

        robot = f"{self.config.robot_name_prefix}1_world"

        brain_class = FetcherCPG
        brain_class.__bases__ = (controllers.get(record.brain[0]),)
        brain = brain_class(weights=record.brain[2], state=state, name=robot, **record.brain[1])    
        monitors["avoider-dynamics"] = dynamics = _Avoider(
            state=state, robot=robot, brain=brain, config=self.config)

        state, model, data = state.unpacked
        with MjcbCallbacks(state, [brain], monitors, self.config):
            for _ in range(int(self.config.duration / model.opt.timestep)):
                mj_step(model, data)
                if dynamics.complete:
                    break

        return -np.inf if dynamics.failure else 100 * dynamics.result


class _AvoidanceOverlay:
    def __init__(self, task: _AvoidanceTask):
        self.task = task

        self.obstacles = []

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

    def _draw(self, scene):
        return
        scene.ngeom = 0
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


class _Avoider(MonitorBase):
    def __init__(self, controller: ABCpg, overlay: _AvoidanceOverlay):
        super().__init__(frequency=20)
        self.controller = controller
        self.overlay = overlay

        self.proximity_threshold = .1

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

            alpha = 1
            beta = 1.0

        else:
            alpha, beta = 1, 0

        if self.overlay is not None and self._debug_draw:
            self.overlay.set_debug_draw(
                self.overlay.DebugDrawData(
                    self.robot.xpos, self.robot.xquat, 
                    alpha=alpha, beta=beta))

        self.controller.set(alpha=alpha, beta=beta)


def prepare_tasks(config: TestingConfig):
    return [_AvoidanceTask("obstacles", config)]
