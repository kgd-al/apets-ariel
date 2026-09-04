from pathlib import Path
from typing import Callable, Optional

import cv2

from PIL import Image
from mujoco import Renderer, MjvCamera, MjvOption, mjtRndFlag, MjvScene

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
            drawings: Optional[Callable[[MjvScene], None]] = None,
            *args, **kwargs
    ):
        super().__init__(frequency / speed_up, *args, **kwargs)
        # self.name = name
        self.path = path
        self.width, self.height = width, height
        self.renderer = None
        self.shadows = shadows
        self.speed_up = speed_up
        self.framerate = frequency

        self.drawings = drawings or []
        if not isinstance(self.drawings, list):
            self.drawings = [self.drawings]

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

        for drawer in self.drawings:
            drawer(self.renderer.scene)

        frame = self.renderer.render()
        if self.gif:
            self.images.append(Image.fromarray(frame))

        else:
            self.images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def stop(self, state: MjState):
        self._step(state)

        framerate = round(self.speed_up * len(self.images) / state.time, 2)
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

            overlay = None
            if self.speed_up != 1:
                overlay = f"x{self.speed_up}"

            for frame in self.images:
                if overlay is not None:
                    cv2.putText(frame, overlay, (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                writer.write(frame)
            writer.release()
