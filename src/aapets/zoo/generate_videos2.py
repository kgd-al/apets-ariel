from datetime import timedelta
import os
import shutil
import sys
import time
from typing import Annotated, Optional

import humanize
import numpy as np

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


NO_COND_PREFIX = "empty"
COND_PREFIX = "house"


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
    
    spec.delete(spec.geom("floor"))
    for light in spec.lights:
        spec.delete(light)
        
    for texture in spec.textures:
        if texture.type == mj.mjtTexture.mjTEXTURE_SKYBOX:
            spec.delete(texture)

    tex = spec.add_texture(name="skybox", type=mj.mjtTexture.mjTEXTURE_SKYBOX,
                            width=1024, height=6144)
    tex.builtin = mj.mjtBuiltin.mjBUILTIN_GRADIENT
    tex.rgb1 = [0.22, 0.48, 0.85]                 # zenith: deeper blue
    tex.rgb2 = [1.0, 1.0, 1.0]                    # nadir: white, so the horizon lands at pale blue
    tex.mark = mj.mjtMark.mjMARK_NONE             # drop the random "stars"
    tex.random = 0

    # optional: match the haze to the horizon colour (currently a dark blue)
    spec.visual.rgba.haze = [0.62, 0.75, 0.92, 1]

    # --

    vis = spec.visual
    vis.quality.shadowsize = 8192
    vis.quality.offsamples = 8
    vis.map.shadowscale = 1.0                   # let spot-light shadows use the full cone
    vis.headlight.ambient = [0.30, 0.29, 0.27]  # dim ambient only; the lamps do the work
    vis.headlight.diffuse = [0.0, 0.0, 0.0]
    vis.headlight.specular = [0.0, 0.0, 0.0]

    # --

    tex2d("t_wood", (0.85, 0.85, 0.85), 0.07, "floor_wood.png")
    tex2d("t_plaster", (0.95, 0.95, 0.95), 0.004, "wall_plaster.png")
    tex2d("t_fabric", (0.92, 0.92, 0.92), 0.25, "fabric.png")
    tex2d("t_grass", (0.72, 0.92, 0.72), 0.25, "grass.png")
    material("floor", [0.88, 0.77, 0.68, 1], "t_wood", (2, 2), reflectance=0.03, specular=0.15, shininess=0.3)
    material("wall", [0.94, 0.91, 0.86, 1], "t_plaster", (6, 6), specular=0.02)
    material("ceiling", [0.97, 0.96, 0.94, 1], "t_plaster", (6, 6), specular=0.0)
    material("grass", [0.97, 0.96, 0.94, 1], "t_grass", (6, 6), specular=0.0)
    material("trim", [0.97, 0.96, 0.93, 1], specular=0.2)
    material("door", [0.45, 0.60, 0.54, 1], specular=0.3, shininess=0.5)       # sage-green accent
    material("brass", [0.86, 0.70, 0.32, 1], specular=0.9, shininess=0.9, reflectance=0.05)
    material("mat", [0.62, 0.58, 0.52, 1], "t_fabric", (4, 4))
    material("rug", [0.55, 0.62, 0.68, 1], "t_fabric", (4, 4))
    material("bed_rim", [0.88, 0.80, 0.80, 1], "t_fabric", (3, 3))
    material("bed_cush", [0.80, 0.45, 0.33, 1], "t_fabric", (3, 3))
    material("charger", [0.90, 0.91, 0.93, 1], specular=0.4, shininess=0.6)
    material("pad", [0.18, 0.19, 0.21, 1], specular=0.3)
    material("led_on", [0.20, 1.00, 0.45, 1], emission=1.0)
    material("batt_dark", (0.15, 0.16, 0.18, 1), specular=0.1)
    material("batt_fill", (0.2, 1.0, 0.45, 1), emission=1)
    material("lamp", [1.00, 0.95, 0.82, 1], emission=1.0)
    material("alu",   (0.16, 0.17, 0.19, 1), specular=0.5, shininess=0.5)
    material("sill",  (0.80, 0.80, 0.78, 1), specular=0.2)
    material("glass", (0.62, 0.80, 0.86, 0.14), specular=0.9, shininess=0.9)
    material("glint", (1.0, 1.0, 1.0, 0.10))
    material("curtain_a", (0.80, 0.64, 0.40, 1), tex="t_fabric", repeat=(3, 3))
    material("curtain_b", (0.68, 0.53, 0.32, 1), tex="t_fabric", repeat=(3, 3))   # shaded side of each fold
    material("curtain_tie", (0.50, 0.34, 0.20, 1))

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
    bat_x = 0.091          # just in front of the face at x = 0.09
    bat_z = 0.12           # centred between the pad and the LED
    bat_h = 0.001          # half-thickness of each layer

    box(dock, "batt_shell", (bat_x,          0.0,    bat_z),
        (bat_h, 0.080, 0.044), "batt_dark", collide=False)
    box(dock, "batt_tip",   (bat_x,          0.088,  bat_z),
        (bat_h, 0.008, 0.017), "batt_dark", collide=False)
    box(dock, "batt_inner", (bat_x + 0.0015, 0.0,    bat_z),
        (bat_h, 0.073, 0.037), "charger",   collide=False)

    for i, y in enumerate((-0.046, 0.0, 0.046), start=1):
        box(dock, f"batt_bar_{i}", (bat_x + 0.003, y, bat_z),
            (bat_h, 0.019, 0.029), "batt_fill", collide=False)

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

    def baie_vitree(world, park_m=1.2, side=-1):
        """Single wide sliding pane mounted outside the wall.
        park_m: 0 = closed across the opening, 1.2 = fully parked beside it (opening fully clear).
        side:   -1 parks it towards -y, +1 towards +y."""
        X0, Z0, Z1 = 4.0, 0.005, 2.25
        zc, hz = (Z0 + Z1) / 2, (Z1 - Z0) / 2
        PW, SW = 1.2, 0.03                       # pane width, stile width
        XP = 4.06                                # pane plane, in front of the wall (outer face x = 4.025)
        ZB, ZT = 0.013, 2.2
        park_m = max(0.0, min(park_m, PW))
        yc = side * park_m                       # current pane centre

        # ---- trim frame in the opening (leaves 1.12 m clear) ----
        for s in (-1, 1):
            box(world, f"door_stile_{s}", (X0, s * 0.58, zc), (0.03, 0.02, hz), "alu")
        box(world, "door_head", (X0, 0.0, Z1 - 0.025), (0.03, 0.56, 0.025), "alu")
        box(world, "door_sill", (X0, 0.0, 0.007), (0.04, 0.6, 0.002), "sill", collide=False)

        # ---- outside: threshold strip, floor guide rail, top track (cover closed + parked span) ----
        yt, ht = side * PW / 2, 0.6 + PW / 2     # track centre / half-length
        box(world, "door_ext_sill",  (4.08, yt, 0.007),  (0.04, ht, 0.002), "sill", collide=False)
        box(world, "door_ext_rail",  (XP,   yt, 0.011),  (0.004, ht, 0.002), "alu", collide=False)
        box(world, "door_top_track", (XP,   yt, 2.265),  (0.03, ht, 0.015), "alu", collide=False)
        for k, dy in enumerate((-0.4, 0.4)):     # hangers joining pane and track
            box(world, f"door_hanger_{k}", (XP, yc + dy, 2.225), (0.008, 0.025, 0.025), "alu", collide=False)

        # ---- the pane ----
        hz_p = (ZT - ZB) / 2
        for s in (-1, 1):                         # stiles
            box(world, f"door_slide_stile_{s}", (XP, yc + s * (PW / 2 - SW / 2), (ZB + ZT) / 2),
                (0.010, SW / 2, hz_p), "alu")
        yr = PW / 2 - SW
        box(world, "door_slide_rail_top", (XP, yc, ZT - 0.02), (0.010, yr, 0.02), "alu")
        box(world, "door_slide_rail_bot", (XP, yc, ZB + 0.03), (0.010, yr, 0.03), "alu")
        z_lo, z_hi = ZB + 0.06, ZT - 0.04
        box(world, "door_slide_glass", (XP, yc, (z_lo + z_hi) / 2),
            (0.003, yr + 0.002, (z_hi - z_lo) / 2 + 0.002), "glass")

        # diagonal glints and handles on both faces (garden side +x, room side -x)
        for f, sx in (("g", 1), ("r", -1)):
            for k, dy in enumerate((-0.35, 0.0, 0.3)):
                gl = box(world, f"door_glint_{f}{k}", (XP + sx * 0.0034, yc + dy, 1.3),
                        (0.0004, 0.012, 0.4), "glint", collide=False)
                a = 0.35
                gl.quat = [np.cos(a / 2), np.sin(a / 2), 0.0, 0.0]
            # handle on the trailing stile, the one left beside the opening when parked
            box(world, f"door_handle_{f}", (XP + sx * 0.016, yc - side * (PW / 2 - SW / 2), 1.0),
                (0.006, 0.008, 0.12), "brass", collide=False)

    baie_vitree(spec.worldbody, park_m=1.0)

    # --

    def prim(parent, name, gtype, pos, size, mat, quat=None):
        g = parent.add_geom(name=h_prefix + name, type=gtype, pos=list(pos), size=list(size),
                            material=h_prefix + mat)
        if quat is not None:
            g.quat = list(quat)
        g.contype = g.conaffinity = 0                 # visual only
        return g

    def tube(parent, name, p0, p1, r, mat):
        g = parent.add_geom(name=h_prefix + name, type=mjtGeom.mjGEOM_CAPSULE, size=[r, 0, 0],
                            fromto=[*p0, *p1], material=h_prefix + mat)
        g.contype = g.conaffinity = 0                 # visual only
        return g

    def curtains(world, n_col=8, r=0.032):
        X_C, Z_ROD, Z_TOP, Z_TIE, Z_BOT = 3.93, 2.38, 2.35, 1.0, 0.04
        Y_ROD, W_ROD = 0.78, 0.32            # panel centre / width where it hangs on the rod
        Y_TIE, W_TIE = 0.88, 0.16            # bundle centre / width at the tieback (pulled outward)
        W_HEM = 0.40                         # width of the fabric fanned out at the hem
        HOOK_Y = 1.12                        # brass hook on the wall that holds the tie
        BULGE = 0.035                        # slack above the tie sags toward the room (-x)
        N_UP, N_LO = 8, 8

        # rod, brackets, finials
        prim(world, "curtain_rod", mjtGeom.mjGEOM_CYLINDER, (X_C, 0, Z_ROD), (0.012, 0.95, 0), "brass",
            quat=(0.7071068, 0.7071068, 0, 0))
        for s in (-1, 1):
            prim(world, f"curtain_finial_{s}", mjtGeom.mjGEOM_SPHERE, (X_C, s * 0.97, Z_ROD), (0.02, 0, 0), "brass")
            prim(world, f"curtain_bracket_{s}", mjtGeom.mjGEOM_BOX, (3.955, s * 0.9, Z_ROD), (0.02, 0.012, 0.012), "brass")

            for i in range(n_col):
                u = i / (n_col - 1) - 0.5
                pleat = 0.012 * (-1) ** i
                mat = "curtain_a" if i % 2 == 0 else "curtain_b"
                pts = []
                # upper part: straight line rod -> bundle, sagging slightly into the room
                for k in range(N_UP + 1):
                    t = k / N_UP
                    y = (Y_ROD + u * W_ROD) * (1 - t) + (Y_TIE + u * W_TIE) * t
                    pts.append((X_C + pleat - BULGE * np.sin(np.pi * t), s * y,
                                Z_TOP + (Z_TIE - Z_TOP) * t))
                # lower part: hangs straight down from the bundle, fanning out (fast at first, then settling)
                for k in range(1, N_LO + 1):
                    t = k / N_LO
                    w = W_TIE + (W_HEM - W_TIE) * (1 - (1 - t) ** 2)
                    pts.append((X_C + pleat * (1 + t), s * (Y_TIE + u * w), Z_TIE + (Z_BOT - Z_TIE) * t))
                for k in range(len(pts) - 1):
                    tube(world, f"curtain_{s}_{i}_{k}", pts[k], pts[k + 1], r, mat)

            # tieback: band around the bundle, strap to a hook on the wall
            prim(world, f"curtain_band_{s}", mjtGeom.mjGEOM_ELLIPSOID, (X_C, s * Y_TIE, Z_TIE),
                (0.05, W_TIE / 2 + 0.04, 0.016), "curtain_tie")
            tube(world, f"curtain_strap_{s}", (X_C, s * (Y_TIE + 0.12), Z_TIE), (3.968, s * HOOK_Y, Z_TIE),
                0.008, "curtain_tie")
            prim(world, f"curtain_hook_{s}", mjtGeom.mjGEOM_SPHERE, (3.972, s * HOOK_Y, Z_TIE), (0.014, 0, 0), "brass")

    curtains(spec.worldbody)

    # --

    # Stone texture: uses stone.png if present, else the procedural speckle fallback of tex2d
    tex2d("t_stone", (0.60, 0.58, 0.54), 0.25, "stone.png")
    for n, c in (("stone_a", (1.00, 1.00, 1.00)), ("stone_b", (0.88, 0.90, 0.92)), ("stone_c", (1.00, 0.95, 0.88))):
        material(n, (*c, 1), tex="t_stone", repeat=(3, 3), specular=0.05)
    material("grout", (0.33, 0.31, 0.28, 1))

    def terrace(world, depth=1.6, half_w=1.8, tile_d=0.40, tile_w=0.60, gap=0.012):
        X_WALL = 4.025
        pitch = tile_d + gap
        n_rows = int(depth // pitch)
        x_end = X_WALL + n_rows * pitch
        # One collidable grout slab (top 2 mm below the tiles)
        box(world, "terrace_grout", ((X_WALL + x_end) / 2, 0.0, -0.011),
            ((x_end - X_WALL) / 2, half_w, 0.014), "grout")
        for r in range(n_rows):
            x = X_WALL + gap + tile_d / 2 + r * pitch - gap * 0.5
            y0 = -half_w - (tile_w / 2 if r % 2 else 0.0)        # offset every other row
            c = 0
            while y0 < half_w:
                a, b = max(y0, -half_w), min(y0 + tile_w - gap, half_w)
                if b - a > 0.05:                                  # skip slivers at the ends
                    box(world, f"terrace_tile_{r}_{c}", (x, (a + b) / 2, -0.001),
                        (tile_d / 2, (b - a) / 2, 0.006), f"stone_{'abc'[(2 * r + c) % 3]}", collide=False)
                y0 += tile_w + gap
                c += 1

    terrace(spec.worldbody)

    # --

    for n, c in (("bush_a", (0.16, 0.38, 0.14)), ("bush_b", (0.21, 0.46, 0.17)), ("bush_c", (0.12, 0.31, 0.12))):
        material(n, (*c, 1), specular=0.0)

    def bushes(world, seed=7, spacing=0.55, x0=4.6, x1=13.4, y_edge=4.8):
        rng = np.random.default_rng(seed)
        ns, nb = int((x1 - x0) / spacing), int(2 * y_edge / spacing)

        # path: left side -> back -> right side
        xs = np.linspace(x0, x1, ns + 1)
        ys = np.linspace(-y_edge, y_edge, nb + 1)[1:-1]
        pts = np.vstack([
            np.column_stack([xs, np.full_like(xs, -y_edge)]),
            np.column_stack([np.full_like(ys, x1), ys]),
            np.column_stack([xs[::-1], np.full_like(xs, y_edge)]),
        ])

        # nudge each point towards the garden centre so the hedge isn't a straight line
        to_c = np.array([9.0, 0.0]) - pts
        pts = pts + to_c / np.linalg.norm(to_c, axis=1, keepdims=True) * rng.uniform(0.0, 0.4, (len(pts), 1))

        mats = ("bush_a", "bush_b", "bush_c")
        def blob(name, x, y, rx, ry, rz):
            a = rng.uniform(0, np.pi)
            prim(world, name, mjtGeom.mjGEOM_ELLIPSOID, (x, y, rz * 0.8), (rx, ry, rz),
                mats[rng.integers(3)], quat=(np.cos(a / 2), 0, 0, np.sin(a / 2)))

        for k, (x, y) in enumerate(pts):
            h = rng.uniform(1.2, 1.9)
            if x > 11.5 and abs(y) < 2.2:            # keep the sun's line to the house clear
                h = min(h, 1.25)
            r = rng.uniform(0.45, 0.70)
            blob(f"bush_{k}_0", x, y, r, r * rng.uniform(0.85, 1.1), h / 2)
            for m in range(1, 4):                      # satellites
                blob(f"bush_{k}_{m}", x + rng.uniform(-r, r), y + rng.uniform(-r, r),
                    r * 0.7, r * 0.7, h / 2 * rng.uniform(0.5, 0.8))

    bushes(spec.worldbody)
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


    # print(spec.to_xml())

## ===================
    


def work(archive: Path, args: Arguments):
    record = RerunnableRobot.load(archive)
    output_prefix = archive.with_suffix("")

    if args.is_default("duration"):
        args.duration = record.config.duration

    #### Default simulation, just with bigger ground

    spec = record.mj_spec

    args.camera = "pretty-cam"
    camera = spec.camera(args.camera)
    camera.pos = [-.75*args.house_size, 0, 1.75]
    mju_euler2Quat(camera.quat, [0, -.4*np.pi, -.5 * np.pi], "xyz")

    spec.body("apet1_world").pos[0] -= .25 * args.house_size

    floor = spec.geom("floor")  # or spec.geom("floor") depending on your mujoco version
    floor.size = [10, 10, 0.1]  # 20x20 units, grid spacing 0.1

    spec.visual.global_.offwidth = args.movie_width
    spec.visual.global_.offheight = args.movie_height

    movie_file = output_prefix.with_suffix(f".{NO_COND_PREFIX}.mp4")
    simulate_one(record, movie_file, args)

    print(f"Generated {movie_file}")

    #### Tweaked simulation, from a household perspective

    build_house(spec, args)

    movie_file = output_prefix.with_suffix(f".{COND_PREFIX}.mp4")
    simulate_one(record, movie_file, args)
    print(f"Generated {movie_file}")

    return archive
    
def main(args: Arguments):
    start = time.perf_counter()
    
    archives = sorted(list(Path("remote/zoo/__champions/").glob("*/champion.zip")))
    n = len(archives)

    args.duration = 10
    args.movie_width = args.movie_height = 1080

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
        for v in [NO_COND_PREFIX, COND_PREFIX]:
            src = a.with_suffix(f".{v}.mp4")
            dst = output_folder.joinpath(f"{a.parent.name}.{v}.mp4")
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
