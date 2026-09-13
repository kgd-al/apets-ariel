from pathlib import Path

from mujoco import MjSpec, mjtGeom, mj_step, mjtMeshInertia
import numpy as np

from aapets.common import controllers

from ...common.monitors._monitor import MonitorBase
from ...common.mujoco.callback import MjcbCallbacks
from ...common.mujoco.state import MjState
from ...common.robot_storage import RerunnableRobot
from ...fetch.dynamics.base import add_ball, add_eyes, add_mouth
from ...fetch.sm_fetcher import FetcherCPG
from ...fetch.types import FetchTaskObjects
from .config import TestingConfig
from .task import TestTask


class _FetchTask(TestTask):
    def __init__(self, name: str, config: TestingConfig):
        super().__init__(name=name, config=config)

        # self.config.duration = 10

    def _modify_specs(self, specs: MjSpec):
        FetchDynamics.adjust_world(specs, self.config)

    def _process(self, state: MjState, record: RerunnableRobot, champion: Path):
        scores = []
        for i in range(5):
            monitors = dict()

            if self.config.movie and i == 0:
                drawers = None
                monitors["movie-recorder"] = self._movie_recorder(champion, drawers)

            robot = f"{self.config.robot_name_prefix}1_world"

            brain_class = FetcherCPG
            brain_class.__bases__ = (controllers.get(record.brain[0]),)
            brain = brain_class(weights=record.brain[2], state=state, name=robot, **record.brain[1])    
            monitors["fetch-dynamics"] = dynamics = FetchDynamics(
                state=state,
                robot=robot, ball=FetchTaskObjects.BALL.value, human=None,
                brain=brain, config=self.config)

            state, model, data = state.unpacked
            with MjcbCallbacks(state, [brain], monitors, self.config):
                for _ in range(int(self.config.duration / model.opt.timestep)):
                    mj_step(model, data)
                    if dynamics.complete:
                        break

            scores.append(-np.inf if dynamics.failure else 100 * dynamics.result)

        valid_scores = [s for s in scores if np.isfinite(s)]
        if len(valid_scores) > 0:
            return np.mean(valid_scores)
        else:
            return -np.inf

class FetchDynamics(MonitorBase):
    target_name = "target_ring"
    target_radius = 0.5

    def __init__(self, state: MjState, 
                 robot: str, ball: str, human: str,
                 brain: FetcherCPG,
                 config: TestingConfig,
                 seed=None
    ):

        super().__init__(frequency=50)

        self.state = state
        self.brain = brain

        self.robot = state.data.body(robot)
        self.ball = state.data.body(ball)
        self.target = state.data.geom(self.target_name)

        pos_rng = np.random.default_rng(0)
        self.positions, self.current = [(0, 0)], 0
        self.total_length = self.dist(self.positions[0])
        while len(self.positions) < 10:
            target_pos = pos_rng.uniform(-config.base_length, config.base_length, size=2)
            if (dist := self.dist(target_pos)) >= 1.5 * self.target_radius:
                self.positions.append(target_pos)
                self.total_length += dist

        shuffle_rng = np.random.default_rng(1)#seed or config.seed)
        shuffle_rng.shuffle(self.positions[1:])

        self.config = config

        self.current_length = 0

    @property
    def result(self): return self.current_length / self.total_length

    @property
    def complete(self): return self.current == len(self.positions)

    @property
    def failure(self): return self.current == 0

    def dist(self, p): return np.linalg.norm(p - self.target.xpos[:2])

    @classmethod
    def adjust_world(cls, specs: MjSpec, config: TestingConfig):
        target_pos = [config.base_length, 0, 0]
        specs.worldbody.add_body(name=FetchTaskObjects.HAND, pos=target_pos)

        add_ring_mesh(specs, cls.target_name, target_pos, radius=cls.target_radius, width=.1)

        add_ball(specs, (0, 0, .05))

        robot_name = f"{config.robot_name_prefix}1"
        add_mouth(specs, robot_name, adhesion_strength=1000)
        add_eyes(specs, robot_name)

    def _step(self, state: MjState):
        super()._step(state)
        dist = np.linalg.norm(self.ball.xpos - self.target.xpos)
        if dist < self.target_radius:
            self.current_length += self.dist(self.positions[self.current])
            self.current += 1

            self.brain.release_ball(.5)

            if not self.complete:
                body_id = self.ball.id
                qpos_adr = state.model.jnt_qposadr[state.model.body_jntadr[body_id]]
                state.data.qpos[qpos_adr:qpos_adr + 2] = self.positions[self.current]


def add_ring_mesh(spec, name, pos, radius, width, n=32,
                   rgba=(1.0, 0.4, 0.0, 1.0)):
    
    outer_r, inner_r = radius + width, radius - width
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    outer = np.stack([outer_r * np.cos(angles), outer_r * np.sin(angles),
                       np.zeros(n)], axis=1)
    inner = np.stack([inner_r * np.cos(angles), inner_r * np.sin(angles),
                       np.zeros(n)], axis=1)
    verts = np.vstack([outer, inner])

    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i]]  # two tris per segment

    mesh = spec.add_mesh(name=f"{name}_mesh")
    mesh.uservert = verts.flatten().tolist()
    mesh.userface = np.array(faces).flatten().tolist()
    mesh.inertia = mjtMeshInertia.mjMESH_INERTIA_SHELL

    spec.worldbody.add_geom(
        name=name,
        type=mjtGeom.mjGEOM_MESH,
        meshname=f"{name}_mesh",
        pos=[pos[0], pos[1], 0.001],  # tiny z-offset avoids z-fighting w/ floor
        rgba=rgba,
        contype=0,       # visual only — no collisions
        conaffinity=0,
    )


def prepare_tasks(config: TestingConfig):
    return [_FetchTask("fetch", config)]
