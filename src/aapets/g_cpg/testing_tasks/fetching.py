from pathlib import Path

from mujoco import MjSpec, mjtTexture, mjtBuiltin, mjtGeom, mj_step, mjtMark, mjtMeshInertia
import numpy as np

from aapets.common import controllers

from ...common.monitors._monitor import MonitorBase
from ...common.mujoco.callback import MjcbCallbacks
from ...common.mujoco.state import MjState
from ...common.mujoco.viewer import passive_viewer
from ...common.robot_storage import RerunnableRobot
from ...fetch.dynamics.base import add_ball, add_eyes, add_mouth
from ...fetch.sm_fetcher import FetcherCPG
from ...fetch.types import FetchTaskObjects
from .config import TestingConfig
from .task import TestTask

import mujoco_menagerie as mm


class _FetchTask(TestTask):
    def __init__(self, name: str, config: TestingConfig):
        super().__init__(name=name, config=config)

        # self.config.duration = 10

    def _modify_specs(self, specs: MjSpec):
        FetchDynamics.adjust_world(specs, self.config)

    def _process(self, state: MjState, record: RerunnableRobot, champion: Path):
        monitors = dict()

        if self.config.movie:
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
            # mj_step(model, data, nstep=int(self.config.duration / model.opt.timestep))
            self.config.auto_start = False
            self.config.camera = "pretty-cam"
            self.config.settings_restore = True
            self.config.settings_save = True
            passive_viewer(state, self.config)
 
        return 100 * (1 - dynamics.result / self.config.duration)


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
        self.rng = np.random.default_rng(1)#)seed or config.seed)

        self.config = config

        self.result = 0

    @classmethod
    def adjust_world(cls, specs: MjSpec, config: TestingConfig):
        target_pos = [config.base_length, 0, 0]
        specs.worldbody.add_body(name=FetchTaskObjects.HAND, pos=target_pos)

        add_ring_mesh(specs, cls.target_name, target_pos, radius=cls.target_radius, width=.1)

        add_ball(specs, (0, 0, .05))
        # add_walls(specs, extent=arena_extent)
        for texture in specs.textures:
            if texture.type == mjtTexture.mjTEXTURE_SKYBOX:
                specs.delete(texture)
        specs.add_texture(builtin=mjtBuiltin.mjBUILTIN_FLAT,
                          rgb1=[0, 0, 0], rgb2=[0, 0, 0],
                          width=1024, height=1024,
                          random=.01, mark=mjtMark.mjMARK_RANDOM, markrgb=[1, 1, 1],
                          type=mjtTexture.mjTEXTURE_SKYBOX, name="skybox")

        robot_name = f"{config.robot_name_prefix}1"
        add_mouth(specs, robot_name, adhesion_strength=1000)
        add_eyes(specs, robot_name)

    def _step(self, state: MjState):
        super()._step(state)
        dist = np.linalg.norm(self.ball.xpos - self.target.xpos)
        if dist < self.target_radius:
            r = .1 * self.config.base_length
            next_target_pos = self.rng.uniform(-r, r, size=2)
            print(self.config.base_length, next_target_pos)

            body_id = self.ball.id
            dof_adr = state.model.body_dofadr[body_id]   # start index into qvel for this body's joint

            self.brain.release_ball()
            state.data.qvel[dof_adr:dof_adr + 3] = compute_launch_velocity_2d(
                state.model, self.ball.xpos, next_target_pos)
            state.data.qvel[dof_adr + 3:dof_adr + 6] = 0.0 


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

def compute_launch_velocity_2d(model, start_pos, target_pos, flight_time=None):
    target_pos = [0, 0]
    delta = np.asarray(target_pos[:2]) - np.asarray(start_pos[:2])
    az = model.opt.gravity[2]

    if flight_time is None:
        flight_time = np.linalg.norm(target_pos[:2] - start_pos[:2])

    vx, vy = delta / flight_time
    vz = -0.5 * az * flight_time

    return np.array([vx, vy, vz])

def prepare_tasks(config: TestingConfig):
    return [_FetchTask("fetch", config)]
