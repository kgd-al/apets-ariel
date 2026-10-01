
from abc import ABC, abstractmethod
from pathlib import Path
import time

from mujoco import MjSpec, mj_forward, mju_euler2Quat

from ...common.mujoco.viewer import passive_viewer
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

        self._modify_specs(record.mj_spec, self.config)

        state, model, data = MjState.from_spec(record.mj_spec).unpacked
        mj_forward(model, data)

        return start, state, record

    @property
    def robot_name(self): return f"{self.config.robot_name_prefix}1_world"

    def _modify_specs(self, specs: MjSpec, config: TestingConfig):
        specs.body(self.robot_name).pos[0] -= self.config.base_length

    @abstractmethod
    def _process(state: MjState): ...

    def __call__(self, champion: Path):
        try:
            start, state, record = self._prepare(champion)
            score = self._process(state, record, champion)

            print(f"Evaluated {champion}: {self.name:10s}"
                f" (score={score:.2f}%; time={state.time:.3g}s; wall time={time.perf_counter() - start:.3}s)")
            return champion, self.name, score
        except Exception as e:
            print(f"Evaluating {champion} failed with {type(e)}: {e}")
            if self.config.do_raise:
                raise e
            else:
                return champion, self.name, float("nan")

    def passive_viewer(self, state, overlays):
        self.config.auto_start = False
        self.config.auto_quit = True
        self.config.camera = "pretty-cam"
        self.config.settings_restore = True
        self.config.settings_save = True
        passive_viewer(state, self.config, overlays=overlays)

    def _movie_file(self, champion: Path):
        return champion.with_suffix(f".eval.{self.name}.mp4")

    def _movie_recorder(self, champion: Path, drawers=None):
        
        return MovieRecorder(
            25, self.config.movie_size, self.config.movie_size,
            self._movie_file(champion),
            speed_up=self.config.movie_speed,
            camera=self.config.movie_camera, shadows=True,
            drawings=drawers
        )
        return 