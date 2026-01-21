import pprint
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Annotated, Type

import numpy as np
from PIL import Image
from PyQt6.QtGui import QPixmap, QMovie
from mujoco import Renderer, mj_forward, MjvOption, mj_step, MjvCamera

from aapets.common.misc.config_base import IntrospectiveAbstractConfig
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer
from aapets.common import canonical_bodies, morphological_measures
from aapets.common.world_builder import compile_world, make_world


_ImageLoader = Callable[[str], Any] | Type


def get_image(
            body: CoreModule, name: str,
            cache_folder: Path,
            loader: _ImageLoader = QPixmap,
            overwrite: bool = False,
        ) -> Any:

    path = cache_folder.joinpath(f"{name}.png")
    if not path.exists() or overwrite:
        state, model, data = compile_world(make_world(body.spec, camera_zoom=.95, camera_centered=True))
        single_frame_renderer(
            model, data, width=200, height=200,
            camera="apet1_tracking-cam",
            save=True, save_path=path,
            transparent=True
        )

        print("Rendering", name, "to", path)

    return loader(str(path))


def get_gif(
            body: CoreModule, name: str,
            cache_folder: Path,
            overwrite: bool = False,
            duration: int = 10,
        ) -> str:

    path = cache_folder.joinpath(f"{name}.gif")

    robot_name = "apet"

    if not path.exists() or overwrite:
        aabb = SimpleFlatWorld.get_aabb(body.spec, "")
        center = .5 * (aabb[1][:2] + aabb[0][:2])
        body.spec.worldbody.pos[:2] -= center

        world_spec = make_world(body.spec, robot_name, camera_centered=True, camera_angle=45)

        state, model, data = compile_world(world_spec)

        framerate = 25
        substeps = int(1 / (model.opt.timestep * framerate))

        visuals = MjvOption()
        camera = MjvCamera()

        camera.distance = 1.5 * max(aabb[1][0] - aabb[0][0], aabb[1][1] - aabb[0][1])

        frames: list[Image.Image] = []

        with Renderer(model, height=200, width=200) as renderer:
            mj_forward(model, data)

            while data.time < duration:
                camera.azimuth = 360 * data.time / duration

                renderer.update_scene(data, scene_option=visuals, camera=camera)
                frames.append(Image.fromarray(renderer.render()))
                mj_step(model, data, nstep=substeps)

        frames[0].save(
            path, append_images=frames[1:],
            duration=1000 / framerate, loop=0,
            optimize=False
        )

        print("Rendering", name, "to", path)

    return str(path)


def get_all_images(cache_folder: Path, loader: _ImageLoader = QPixmap, overwrite=False):
    return {
        name: get_image(fn(), name, cache_folder, loader, overwrite)
        for name, fn in canonical_bodies.get_all().items()
    }


def get_all_gifs(cache_folder: Path, overwrite=False, duration=10):
    return {
        name: get_gif(fn(), name, cache_folder, overwrite)
        for name, fn in canonical_bodies.get_all().items()
    }


def foo():
    for body in canonical_bodies.get_all():
        print(body)

        mm = morphological_measures.measure(canonical_bodies.get(body).spec)
        assert all(0 <= x <= 1 for x in mm.major_metrics.values()), \
            f"Out-of-range error:\n{pprint.pformat(mm.major_metrics)}"
        pprint.pprint(mm.all_metrics)


if __name__ == "__main__":
    @dataclass
    class Arguments(IntrospectiveAbstractConfig):
        cache_folder: Annotated[Path, "Where to store showcase data"] = Path("tmp/cache")
        overwrite: Annotated[bool, "Whether to overwrite cached data"] = False
        duration: Annotated[int, "Duration of the camera loop for gifs"] = 20

    args = Arguments.parse_command_line_arguments(
        description="Generates images (png) and visuals (gif) of all canonical morphologies")
    get_all_images(args.cache_folder, loader=lambda s: None, overwrite=args.overwrite)
    get_all_gifs(args.cache_folder, overwrite=args.overwrite, duration=args.duration)
