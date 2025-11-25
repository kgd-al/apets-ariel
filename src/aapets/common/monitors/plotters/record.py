from pathlib import Path

import cv2
from mujoco import Renderer

from .._monitor import Monitor
from ...mujoco.state import MjState


class MovieRecorder(Monitor):
    def __init__(
            self,
            frequency,
            width, height,
            path: Path):
        super().__init__(frequency)
        # self.name = name
        self.path = str(path)
        self.width, self.height = width, height
        self.renderer, self.writer = None, None

    def start(self, state: MjState):
        self.renderer = Renderer(state.model, height=self.height, width=self.width)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.path, fourcc, self.frequency, (self.width, self.height))

    def _step(self, state: MjState):
        self.renderer.update_scene(state.data)#, scene_option=visuals, camera=camera)
        self.writer.write(self.renderer.render())

    def stop(self, state: MjState):
        self.writer.release()
