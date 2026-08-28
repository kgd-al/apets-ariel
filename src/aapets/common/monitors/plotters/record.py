from pathlib import Path
from typing import Optional

import cv2

from PIL import Image
from mujoco import Renderer, MjvCamera, MjvOption, mjtRndFlag

from .._monitor import MonitorBase
from ...mujoco.state import MjState


class MovieRecorder(MonitorBase):
    def __init__(
            self,
            frequency,
            width, height,
            path: Path,
            speed_up=1,
            camera: int | str | MjvCamera = -1,
            shadows: bool = False,
            visuals: Optional[MjvOption] = None,
            *args, **kwargs
    ):
        super().__init__(frequency, *args, **kwargs)
        # self.name = name
        self.path = path
        self.width, self.height = width, height
        self.renderer = None
        self.shadows = shadows
        self.speed_up = speed_up
        self.framerate = self.frequency * self.speed_up

        self.gif = (self.path.suffix == ".gif")
        self.images = []

        if camera is None:
            camera = -1
        elif isinstance(camera, str):
            try:
                camera = int(camera)
            except ValueError:
                pass

        self.visuals, self.camera = visuals, camera

    def start(self, state: MjState):
        super().start(state)
        self.renderer = Renderer(state.model, height=self.height, width=self.width)
        self.images = []

        self.renderer.scene.flags[mjtRndFlag.mjRND_SHADOW] = self.shadows

    def _step(self, state: MjState):
        self.renderer.update_scene(state.data, scene_option=self.visuals, camera=self.camera)
        frame = self.renderer.render()
        if self.gif:
            self.images.append(Image.fromarray(frame))

        else:
            self.images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def stop(self, state: MjState):
        self._step(state)

        framerate = self.speed_up * len(self.images) / state.time
        if self.framerate / framerate > 1.1 or framerate / self.framerate > 1.1:
            print(f"Warning: actual framerate of {framerate}Hz differs from expected"
                  f" {self.framerate}Hz by more than 10%")
        
        if self.gif:
            self.images[0].save(
                self.path, append_images=self.images[1:],
                duration=1000 / framerate, loop=0,
                optimize=False
            )
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(self.path), fourcc, framerate, (self.width, self.height))
            for frame in self.images:
                writer.write(frame)
            writer.release()
