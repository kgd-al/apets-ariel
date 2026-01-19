from pathlib import Path
from typing import Any, Callable

from PyQt6.QtGui import QPixmap

from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.utils.renderers import single_frame_renderer
from ..common import canonical_bodies
from ..common.world_builder import compile_world, make_world


def get_image(body: CoreModule, name: str, cache_folder: Path, loader: Callable[[str], Any] = QPixmap) -> Any:
    path = cache_folder.joinpath(f"{name}.png")
    if not path.exists():
        state, model, data = compile_world(make_world(body.spec, camera_zoom=.95, camera_centered=True))

        # mujoco.viewer.launch(model, data)

        single_frame_renderer(
            model, data, width=200, height=200,
            camera="apet1_tracking-cam",
            save=True, save_path=path,
            transparent=True
        )

        print("Rendering", name, "to", path)

    return loader(str(path))


def get_all_images(cache_folder: Path, loader=QPixmap):
    return {
        name: get_image(fn(), name, cache_folder, loader) for name, fn in canonical_bodies.get_all().items()
    }
