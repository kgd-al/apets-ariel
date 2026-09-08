
from abc import ABC, abstractmethod
from pathlib import Path
import time

from mujoco import MjSpec, mj_forward, mju_euler2Quat

from ...common.monitors.plotters.record import MovieRecorder
from ...common.mujoco.state import MjState
from ...common.robot_storage import RerunnableRobot
from .config import TestingConfig


class TestTask(ABC):
    def __init__(self, name: str, config: TestingConfig):
        self.name = name
        self.config = config
        self.base_length = config.base_length
        self.config.duration = self.config.base_duration

    def _prepare(self, champion: Path):
        start = time.perf_counter()
        record = RerunnableRobot.load(champion)

        camera = record.mj_spec.camera(self.config.movie_camera)
        camera.pos[:] = [0, 0, 3 * self.config.base_length]
        mju_euler2Quat(camera.quat, [0, 0, 0], "xyz")

        self._modify_specs(record.mj_spec)

        state, model, data = MjState.from_spec(record.mj_spec).unpacked
        mj_forward(model, data)

        data.qpos[0] = -self.config.base_length

        return start, state, record

    def _modify_specs(self, specs: MjSpec): pass

    @abstractmethod
    def _process(state: MjState): ...

    def __call__(self, champion: Path):
        start, state, record = self._prepare(champion)
        score = self._process(state, record, champion)

        print(f"Evaluated {champion}: {self.name}"
              f" (score={score:.2f}%; time={state.time:.3g}s; wall time={time.perf_counter() - start:.3}s)")
        return champion, self.name, score

    def _movie_recorder(self, champion: Path, drawers=None):
        movie_file = champion.with_suffix(f".eval.{self.name}.mp4")
        return MovieRecorder(
            25, self.config.movie_size, self.config.movie_size,
            movie_file,
            speed_up=self.config.movie_speed,
            camera=self.config.movie_camera, shadows=True,
            drawings=drawers
        )
        return 