import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
from PIL import ImageDraw
from mujoco import MjModel, MjData

from aapets.common import canonical_bodies
from aapets.common.misc.config_base import ConfigBase
from aapets.common.world_builder import make_world, compile_world
from ariel.simulation.environments import BaseWorld
from ariel.utils.renderers import single_frame_renderer


@dataclass
class Config(ConfigBase):
    body: Annotated[str, "Robot body to generate the html for",
                    dict(choices=canonical_bodies.get_all().keys(), required=True)] = None

    image_size: Annotated[int, "How big should the body plan be"] = 200
    camera_zoom: Annotated[float, "What proportion of the image to fill with the robot"] = 0.95

    output: Annotated[Path, "Where to store the resulting data"] = Path("tmp/anthropomorphism")


def annotated_image(world: BaseWorld, model: MjModel, data: MjData, path: Path, config: Config):
    aabb = world.get_aabb(world.spec, "apet")[:, :2] / config.camera_zoom

    img = single_frame_renderer(
        model, data, width=config.image_size, height=config.image_size,
        camera="apet1_tracking-cam",
        transparent=True
    )

    geoms = {}

    for i in range(model.ngeom):
        if (g := data.geom(i)).name[:4] == "apet":
            name = g.name.replace("stator", "").replace("rotor", "")
            pos = g.xpos[:2]

            # Merge stator and rotor positions
            if (other_pos := geoms.get(name, None)) is not None:
                pos = .5 * (other_pos + pos)

            geoms[name] = pos

    textbox_size = (8, 6)
    text_bbox = np.array([-1, -1, 1, 1]) * np.tile(textbox_size, 2)

    drawer = ImageDraw.Draw(img, "RGBA")
    for i, pos in enumerate(geoms.values()):
        pos = ((aabb[1, :] - pos) / (aabb[1, :] - aabb[0, :])) * config.image_size
        pos[0] = config.image_size - pos[0]
        drawer.rectangle(tuple(text_bbox + np.tile(pos, 2)),
                         fill=(255, 255, 255, 255), outline="black")
        drawer.text(pos, str(i+1), anchor="mm", fill="black")

    img.save(path)

    return len(geoms)


def main(config: Config):
    body = canonical_bodies.get(config.body)
    world = make_world(body.spec, camera_zoom=.95, camera_centered=True)
    model, data = compile_world(world)

    filename = Path(f"{config.body}.html")
    html_file = config.output.joinpath(filename)

    n = annotated_image(world, model, data, html_file.with_suffix(".png"), config)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Name-a-robot</title>
    <style>
        h1 {{text-align: center;}}
        div {{text-align: center;}}
    </style>
</head>
<body>

<div style="width: 100%">
    <div style="display: inline-block">
    <div>
        <h1>Locomotion</h1>
        <img src="placeholder.gif" alt="Video of a walking robot"/>
    </div>
    <div>
        <h1>Body plan</h1>
        <img src="{filename.with_suffix('.png')}" alt="Body plan of the robot"/>
    </div>
    </div>
    <div style="display: inline-block">
        <h1>Survey</h1>
        <form>
            <section>
                <p>
                    <label for="name">Robot name:</label>
                    <input type="text" id="name" name="username" required />
                </p>
            </section>
            <section>
                <h2>Body plan details</h2>"""

    for i in range(n):
        html += f"""
                <p>
                    <label for="module{i}">{i+1}</label>
                    <input type="text" id="module{i}-name" name="module{i}-name" required />
                </p>
            """

    html += """
            </section>
            <section>
                <h2>General feelings</h2>
                <p>
                    <label for="name">Friendliness:</label>
                    <input type="range" min=0 max=10 id="name" name="username" required />
                </p>
            </section>
            <section>
                <p>
                    <button type="submit">Validate</button>
                </p>
            </section>
        </form>
    </div>
</div>

</body>
</html>
"""

    with open(html_file, "w") as f:
        f.write(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Main evolution script")
    Config.populate_argparser(parser)
    # parser.print_help()
    parsed_config = parser.parse_args(namespace=Config())

    main(parsed_config)
