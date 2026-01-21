from pathlib import Path
from typing import Optional

import cv2
from mujoco import Renderer, MjvCamera, MjvOption, mjtVisFlag, mjtRndFlag

from .._monitor import Monitor
from ...mujoco.state import MjState


class MovieRecorder(Monitor):
    def __init__(
            self,
            frequency,
            width, height,
            path: Path,
            speed_up=1,
            camera: int | str | MjvCamera = -1,
            shadows: bool = False,
            visuals: Optional[MjvOption] = None,
    ):
        super().__init__(frequency)
        # self.name = name
        self.path = path
        self.width, self.height = width, height
        self.renderer, self.writer = None, None
        self.shadows = shadows
        self.speed_up = speed_up

        if camera is None:
            camera = -1
        elif isinstance(camera, str):
            try:
                camera = int(camera)
            except ValueError:
                pass

        self.visuals, self.camera = visuals, camera

    def start(self, state: MjState):
        self.renderer = Renderer(state.model, height=self.height, width=self.width)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.path), fourcc, self.frequency * self.speed_up, (self.width, self.height))

        self.renderer.scene.flags[mjtRndFlag.mjRND_SHADOW] = self.shadows

    def _step(self, state: MjState):
        self.renderer.update_scene(state.data, scene_option=self.visuals, camera=self.camera)
        frame = self.renderer.render()
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def stop(self, state: MjState):
        self.writer.release()
