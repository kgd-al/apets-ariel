from datetime import timedelta
import os
import shutil
import sys
import time
from typing import Annotated, Optional

import humanize
import numpy as np
import pandas as pd

from aapets.common.monitors._monitor import MonitorBase
os.environ["MUJOCO_GL"] = "egl"

from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
from mujoco import mj_forward, mj_step, mjtGeom, mju_euler2Quat, MjSpec

from concurrent.futures import ProcessPoolExecutor, as_completed
from rich.progress import Progress

from aapets.common import controllers
from aapets.common.metrics_storage import BAD, GOOD, RESET
from aapets.common.monitors.plotters.record import MovieRecorder
from aapets.common.mujoco.callback import MjcbCallbacks
from aapets.common.mujoco.state import MjState
from aapets.common.robot_storage import RerunnableRobot
from aapets.bin.rerun import Arguments as RerunArguments
from aapets.zoo.config import Arguments as ZooArguments
from aapets.common.world_builder import adjust_side_camera


@dataclass
class Arguments(RerunArguments):
    house_size: Annotated[float, "Size of the house"] = 4
    house_tex_dir: Annotated[Optional[Path], "Path the texture folder"] = None


def simulate_one(record: RerunnableRobot, file: Path, args: Arguments):
    state, model, data = MjState.from_spec(record.mj_spec).unpacked
    mj_forward(model, data)

    brain = controllers.get(record.brain[0])(
        weights=record.brain[2], state=state, name=args.robot_name_prefix, **record.brain[1])

    monitors = dict(movie_recorder=MovieRecorder(
        args.movie_framerate, args.movie_width, args.movie_height,
        file,
        camera=args.camera, shadows=True
    ))

    with MjcbCallbacks(state, [brain], monitors, args) as callback:
        mj_step(model, data, nstep=int(args.duration / model.opt.timestep))

    return callback.metrics

## ===================


def build_house(spec: MjSpec, args: Arguments):
    h_prefix = "house_" 
    hs = args.house_size
    hh = 2.5      # Wall height
    ws = 0.025    # Wall depth
    ds = .3 * hs  # Door size
    dh = .8 * hh  # Door height

    rs = .3       # Robot size
    c_inset = 1.0 * rs  # anchor distance from wall / half-depth of mats
    c_foot = 0.9 * rs       

    gs = 10        # Garden size      

    world = spec.worldbody

    tex_dir = args.house_tex_dir or Path(sys.argv[0]).parent.joinpath("textures")

    def box(parent, name, pos, half, mat, collide=True):
        g = parent.add_geom(name=h_prefix + name, type=mjtGeom.mjGEOM_BOX,
                            pos=list(pos), size=list(half),
                            material=h_prefix + mat)
        if not collide:
            g.contype = g.conaffinity = 0       # visual only
        return g
    
    def tex2d(name, rgb, noise, file):
        path = tex_dir / file
        if path.exists():
            spec.add_texture(name=h_prefix + name, type=mj.mjtTexture.mjTEXTURE_2D, file=str(path.resolve()))
        else:                                   # procedural fallback: flat colour + speckle
            spec.add_texture(name=h_prefix + name, type=mj.mjtTexture.mjTEXTURE_2D,
                             builtin=mj.mjtBuiltin.mjBUILTIN_FLAT, rgb1=list(rgb),
                             mark=mj.mjtMark.mjMARK_RANDOM, markrgb=[1, 1, 1], random=noise,
                             width=512, height=512)
 
    def material(name, rgba, tex=None, repeat=None, **kw):
        m = spec.add_material(name=h_prefix + name, rgba=list(rgba), **kw)
        if tex:
            m.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = h_prefix + tex
        if repeat:
            m.texrepeat, m.texuniform = list(repeat), True
        return m

    # --
    
    camera = spec.camera(args.camera)
    camera.pos = [-.75*hs, 0, 1.75]
    mju_euler2Quat(camera.quat, [0, -.4*np.pi, -.5 * np.pi], "xyz")

    spec.delete(spec.geom("floor"))
    for light in spec.lights:
        spec.delete(light)
        
    for texture in spec.textures:
        if texture.type == mj.mjtTexture.mjTEXTURE_SKYBOX:
            spec.delete(texture)

    spec.add_texture(builtin=mj.mjtBuiltin.mjBUILTIN_FLAT,
                     rgb1=[1, 1, 1], rgb2=[.8, .8, 1],
                     width=1024, height=1024,
                     random=.01, mark=mj.mjtMark.mjMARK_RANDOM, markrgb=[1, 1, 1],
                     type=mj.mjtTexture.mjTEXTURE_SKYBOX, name="skybox")

    spec.body("apet1_world").pos[0] -= .25 * hs

    # --

    vis = spec.visual
    vis.quality.shadowsize = 8192
    vis.quality.offsamples = 8
    vis.global_.offwidth, vis.global_.offheight = 1920, 1080
    vis.map.shadowscale = 1.0                   # let spot-light shadows use the full cone
    vis.headlight.ambient = [0.30, 0.29, 0.27]  # dim ambient only; the lamps do the work
    vis.headlight.diffuse = [0.0, 0.0, 0.0]
    vis.headlight.specular = [0.0, 0.0, 0.0]

    # --

    tex2d("t_wood", (0.85, 0.85, 0.85), 0.07, "floor_wood.png")
    tex2d("t_plaster", (0.95, 0.95, 0.95), 0.004, "wall_plaster.png")
    tex2d("t_fabric", (0.92, 0.92, 0.92), 0.25, "fabric.png")
    tex2d("t_grass", (0.72, 0.92, 0.72), 0.25, "grass.png")
    material("floor", [0.58, 0.47, 0.38, 1], "t_wood", (2, 2), reflectance=0.03, specular=0.15, shininess=0.3)
    material("wall", [0.94, 0.91, 0.86, 1], "t_plaster", (6, 6), specular=0.02)
    material("ceiling", [0.97, 0.96, 0.94, 1], "t_plaster", (6, 6), specular=0.0)
    material("grass", [0.97, 0.96, 0.94, 1], "t_grass", (6, 6), specular=0.0)
    material("trim", [0.97, 0.96, 0.93, 1], specular=0.2)
    material("door", [0.45, 0.60, 0.54, 1], specular=0.3, shininess=0.5)       # sage-green accent
    material("brass", [0.86, 0.70, 0.32, 1], specular=0.9, shininess=0.9, reflectance=0.05)
    material("mat", [0.62, 0.58, 0.52, 1], "t_fabric", (4, 4))
    material("rug", [0.55, 0.62, 0.68, 1], "t_fabric", (4, 4))
    material("bed_rim", [0.88, 0.60, 0.48, 1], "t_fabric", (3, 3))
    material("bed_cush", [0.80, 0.45, 0.33, 1], "t_fabric", (3, 3))
    material("charger", [0.90, 0.91, 0.93, 1], specular=0.4, shininess=0.6)
    material("pad", [0.18, 0.19, 0.21, 1], specular=0.3)
    material("led_on", [0.20, 1.00, 0.45, 1], emission=1.0)
    material("lamp", [1.00, 0.95, 0.82, 1], emission=1.0)
    material("hall", [0.95, 0.90, 0.80, 1], emission=0.35)

    # --

    box(world, "floor", [0, 0, -0.02], [hs, hs, ws], "floor")
    box(world, "ceiling", [0, 0, hh + ws], [hs, hs, ws], "ceiling", collide=False)

    box(world, "wall_left", [hs, -.5*hs, 0], [ws, .5*(hs-ds), hh], "wall")
    box(world, "wall_right", [hs, +.5*hs, 0], [ws, .5*(hs-ds), hh], "wall")
    box(world, "wall_up", [hs, 0, hh + .5 * (hh - dh)], [ws, ds, hh-dh], "wall")

    # --

    dock = world.add_body(name=h_prefix + "charger",
                          pos=[hs, -1.1 * ds, 0.01], quat=[0, 0, 0, 1])
    box(dock, "dock_pad", [c_inset, 0, 0.01], [c_foot, 0.5 * rs, 0.01], "pad")
    box(dock, "dock_unit", [0.15 * rs, 0, 0.45 * rs], [0.15 * rs, 0.42 * rs, 0.45 * rs], "charger")
    for s in (+1, -1):                           # contact strips
        box(dock, f"dock_contact_{s}", [0.55 * rs, s * 0.2 * rs, 0.0215],
            [0.2 * rs, 0.035 * rs, 0.0015], "brass", collide=False)
    led = dock.add_geom(name=h_prefix + "dock_led", type=mjtGeom.mjGEOM_SPHERE, size=[0.05 * rs, 0, 0],
                        pos=[0.3 * rs + 0.005, 0, 0.72 * rs], material=h_prefix + "led_on")
    led.contype = led.conaffinity = 0
    # dock.add_light(name=h_prefix + "dock_glow", type=mj.mjtLightType.mjLIGHT_POINT,
    #                 pos=[0.3 * rs + 0.12 * rs, 0, 0.72 * rs], diffuse=[0.05, 0.30, 0.12],
    #                 attenuation=[1, 0, 12], castshadow=False)

    # --

    a_ax, b_ax, rim_r = 0.95 * rs, 0.75 * rs, 0.17 * rs
    bed = world.add_body(name=h_prefix + "bed", pos=[hs-c_inset, 1.1*ds, 0])
    box(bed, "rug", [0, 0, 0.004], [0.9 * rs, 1.1 * rs, 0.004], "rug", collide=False)
    n = 32                                       # bolster = ring of capsules (long axis along the wall)
    ring = [(b_ax * np.sin(2 * np.pi * i / n), a_ax * np.cos(2 * np.pi * i / n)) for i in range(n + 1)]
    for i in range(n):
        bed.add_geom(name=f"{h_prefix}bed_rim_{i}", type=mjtGeom.mjGEOM_CAPSULE, size=[rim_r, 0, 0],
                        fromto=[*ring[i], rim_r, *ring[i + 1], rim_r], material=h_prefix + "bed_rim")
    bed.add_geom(name=h_prefix + "bed_cushion", type=mjtGeom.mjGEOM_ELLIPSOID, size=[b_ax, a_ax, 0.14 * rs],
                    material=h_prefix + "bed_cush")
    
    # --

    box(world, "grass", [hs+.5*gs, 0, -0.025], [.5*gs, .5*gs, 0.025], "grass")

    # --

    zl = hh - 0.03
    world.add_light(name=h_prefix + "pendant", type=mj.mjtLightType.mjLIGHT_SPOT, pos=[0, 0, zl], dir=[0, 0, -1],
                    cutoff=89, exponent=.5, diffuse=[0.80, 0.74, 0.62], specular=[0.10, 0.10, 0.10],
                    attenuation=[1, 0, 0], castshadow=True)
    world.add_light(name=h_prefix + "sun", type=mj.mjtLightType.mjLIGHT_SPOT,
                    pos=[hs+gs, 0, .5*hs], dir=[-1, 0, -1],
                    diffuse=[0.8, 0.8, 0.8], specular=[0.2, 0.2, 0.2],
                    range=100, exponent=.01, castshadow=True)


## ===================
    


def work(archive: Path, args: Arguments):
    record = RerunnableRobot.load(archive)
    output_prefix = archive.with_suffix("")

    if args.is_default("duration"):
        args.duration = record.config.duration

    #### Default simulation, just with bigger ground

    args.camera = f"{args.robot_name_prefix}1_tracking-cam"
    args.camera_angle = 45
    args.camera_distance = 2
    adjust_side_camera(record.mj_spec, args, args.robot_name_prefix)

    spec = record.mj_spec
    floor = spec.geom("floor")  # or spec.geom("floor") depending on your mujoco version
    floor.size = [10, 10, 0.1]  # 20x20 units, grid spacing 0.1

    movie_file = output_prefix.with_suffix(".mp4")
    simulate_one(record, movie_file, args)

    print(f"Generated {movie_file}")

    #### Tweaked simulation, with positive and negative outcomes

    args.camera = "pretty-cam"
    build_house(spec, args)

    movie_file = output_prefix.with_suffix(".house.mp4")
    simulate_one(record, movie_file, args)
    print(f"Generated {movie_file}")

    return archive
    
def main(args: Arguments):
    start = time.perf_counter()
    
    archives = sorted(list(Path("remote/zoo/__champions/").glob("*/champion.zip")))
    n = len(archives)

    args.duration = 10

    with Progress() as progress, ProcessPoolExecutor(max_workers=os.cpu_count()-1) as executor:
        task = progress.add_task("Processing...", total=n)
        futures = [executor.submit(work, archive, args) for archive in archives]
        failures = 0
        for future in as_completed(futures):
            archive = future.result()
            progress.update(task, advance=1, description=archive)

    output_folder = Path("remote/zoo/__hs_videos")
    output_folder.mkdir(exist_ok=True, parents=True)
    for a in archives:
        for v in ["", ".house"]:
            src = a.with_suffix(f"{v}.mp4")
            dst = output_folder.joinpath(f"{a.parent.name}{v}.mp4")
            print(src, "->", dst)
            shutil.copyfile(src, dst)

    if failures == 0:
        msg = f"{GOOD}Successfully processed {n} items{RESET}"
    else:
        msg = f"{BAD}Failed to process {failures} archives (out of {n}){RESET}"

    duration = humanize.precisedelta(timedelta(seconds=time.perf_counter() - start))
    print(msg, f"in {duration}s")


if __name__ == "__main__":
    main(Arguments.parse_command_line_arguments(
        description="Simple utility to generate all videos for the human study"))
