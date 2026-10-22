"""Print-in-place keychain / desk toy of any ARIEL robogen_lite robot.

Reads the MuJoCo specification of a robot (the XML written by
``spec.to_xml()``, or a compiled ``.mjb``) and builds a 3D-printable model of
the core / hinge / brick tree it describes. The geometry generalises the
original "spider" keychain:

    * Every hinge module becomes a captive print-in-place joint. The stator
      is fused to whatever the hinge hangs off (core, brick or the rotor of
      another hinge), the rotor to whatever hangs off the hinge.
    * A hinge whose joint axis is vertical in the spec (``rotate(0)``) becomes
      a *yaw* joint: a stepped post with a flange under a 45-degree cone cap.
      A hinge whose axis is horizontal (``rotate(+-90)``) becomes a *pitch*
      joint: a three-knuckle hinge with cone-shaped pins in cone sockets.
      Both stop at roughly +-90 degrees, like the SER0019 servos.
    * Cores and bricks are filleted boxes. The core carries the engraved TEXT
      on its bottom face, a camera module (only when the spec has a camera on
      the core) and, for the desk toy, a raised logo. By default the camera
      sits on a flat 45-degree face cut into a vertical corner of the core,
      like a cyclopean eye, and the logo is rotated to look the same way
      (``--camera-corner front`` restores the camera on the flat front face).
    * Everything prints flat on the bed in one piece, without supports. The
      number of separate solids is 1 + (number of hinges).

Poses, sizes and joint axes are read from the *compiled* model at its
reference configuration, so ``<frame>`` placements, quaternions, prefixes and
the core's attachment sites are all handled by MuJoCo itself. Everything is
expressed relative to the core, so the robot may be placed anywhere in the
world of the file.

Scale: the printed hinge cross-section is HINGE_W mm (the joint details below
are tuned for it); every other length follows from the ratios in the spec.
With the original ARIEL dimensions (52 mm hinge) this is exactly 1/7.5.

Limits (a flat, supportless print):
    * Modules must lie in the horizontal plane at the reference pose: nothing
      attached to the top/bottom face of a brick, no brick rotated by 45
      degrees. Offending subtrees raise an error, or are dropped with
      ``--drop-unprintable``.
    * Hinges tilted by 45 degrees (e.g. the gecko's back legs) have no
      printable joint; they are snapped to yaw or pitch (``--snap45``).
    * Like the original, the model ignores the vertical placement of the
      modules in the simulator: all of them sit flush with the bed.

Requires ``pip install build123d trimesh mujoco``.

    python robot_keychain.py spider.xml
    python robot_keychain.py gecko.xml --text Gecko --variant desk
"""

import argparse
import warnings
from dataclasses import dataclass, field
from math import atan2, degrees, sqrt
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from build123d import (
    Align,
    Axis,
    Box,
    Circle,
    Compound,
    Cone,
    Cylinder,
    Face,
    FontStyle,
    Plane,
    Polygon,
    Pos,
    Rectangle,
    Rot,
    Text,
    export_step,
    export_stl,
    extrude,
    fillet,
    import_svg,
    revolve,
)

from aapets.common.robot_storage import RerunnableRobot
from aapets.g_cpg.config import Config  # noqa: F401

# --- Keychain parameters (mm) --- #
HINGE_W = 52 / 7.5  # printed hinge cross-section (width == height); sets the scale
FALLBACK_SCALE = 1 / 7.5  # printed mm per real mm for specs without any hinge
CLR = 0.30  # radial / normal clearance between moving parts
AXIAL_GAP = 0.35  # gap between pitch-hinge knuckles
EDGE_FILLET = 1.0  # rounding of core / brick edges
FUSE_OVERLAP = 0.5  # how far a hinge half is sunk into the part it is fused to

# Yaw joint (vertical axis)
YAW_ROOF_Z = 2.4  # height of the rotor ring rim before the 45 deg roof
POST_R = 1.2  # post shaft radius
FLANGE_R = 2.0  # retaining flange radius at the post foot
FLANGE_Z = 0.6  # flange straight height before its 45 deg taper

# Pitch joint (horizontal axis)
FORK_W = 1.7  # width of each outer knuckle
PIN_R = 1.5  # cone pin base radius
PIN_L = 1.0  # cone pin length (45 deg, truncated)

# Camera module on the front face
CAM_W = 8.0  # housing width
CAM_H = 4.5  # housing height
CAM_D = 2.2  # housing protrusion (underside is a 45 deg chamfer of this depth)
CAM_CORNER = "back-right"  # front, or front/back-left/right: where the camera looks
CAM_FACE_MARGIN = 0.5  # flat face left around the housing on each side (corner mount)
LENS_R = 1.5
LENS_L = 0.8
LENS_HOLE_R = 0.8
SCREW_R = 0.45
SCREW_X = 3.0

# Engraving in the core's bottom face (reads correctly when flipped over)
TEXT = "Gram"
TEXT_SIZE = 6.5
TEXT_DEPTH = 0.6  # 3 layers at 0.2 mm
TEXT_FONT = "Arial"
TEXT_MARGIN = 1.5  # free border around the engraving; longer text is shrunk to fit

# Logo on the core's top face (desk toy only: the key ring loop sits there).
# An SVG with filled shapes, scaled to fit LOGO_SIZE (shrunk to fit small cores).
LOGO_SVG = "vu_logo.svg"
LOGO_SIZE = 17.5  # longest side, mm
LOGO_HEIGHT = 0.6  # raised by 3 layers; set a colour change at this height to make it pop

# Key ring loop on top of the core
LOOP_W = 10.0
LOOP_T = 3.0
LOOP_HOLE_R = 2.2

SCRIPT_DIR = Path(__file__).resolve().parent

# --- Derived joint geometry (depends on HINGE_W only) --- #
W = H = HINGE_W
R = W / 2  # yaw ring radius == pitch knuckle radius

GAP_45 = CLR * sqrt(2)  # vertical gap that gives CLR normal to a 45 deg face
YAW_RC = R * sqrt(2) + CLR  # radius of the lowered zone, sets the +-90 deg stop
ROOF_TOP = YAW_ROOF_Z + R
CENTER_W = W - 2 * FORK_W - 2 * AXIAL_GAP  # pitch rotor knuckle width

assert ROOF_TOP + GAP_45 < H, "yaw cap would be thinner than zero at the axis"
assert CENTER_W > 2 * (PIN_L + CLR - AXIAL_GAP), "pin sockets would meet"


def box(x0, x1, y0, y1, z0, z1):
    """Axis-aligned box from min/max corners."""
    return Pos(x0, y0, z0) * Box(
        x1 - x0, y1 - y0, z1 - z0, align=(Align.MIN, Align.MIN, Align.MIN)
    )


def rounded_box(x0, x1, y0, y1, height):
    """Box on the bed with its top and vertical edges filleted."""
    if min(x1 - x0, y1 - y0, height) <= 2 * EDGE_FILLET:
        raise ValueError(
            f"a core/brick of {x1 - x0:.1f} x {y1 - y0:.1f} x {height:.1f} mm is too small "
            f"for EDGE_FILLET = {EDGE_FILLET} mm"
        )
    b = box(x0, x1, y0, y1, 0, height)
    edges = b.edges().filter_by(Axis.Z) + b.edges().group_by(Axis.Z)[-1]
    return fillet(edges, EDGE_FILLET)


def revolved(pts):
    """Revolve an (r, z) polygon about the local Z axis."""
    return revolve(Plane.XZ * Polygon(*pts, align=None), Axis.Z)


def roof(r):
    """Top surface height of the yaw rotor ring at radius r (45 deg cone)."""
    return ROOF_TOP - r


# --- Yaw joint tools, local frame: axis = Z through origin, rotor side is +y --- #
yaw_ring = revolved([(0, 0), (R, 0), (R, YAW_ROOF_Z), (0, ROOF_TOP)])
yaw_rotor_clear = revolved(
    [(0, ROOF_TOP), (YAW_RC, roof(YAW_RC)), (YAW_RC, H + 1), (0, H + 1)]
)
yaw_stator_clear = revolved(
    [(0, -1), (YAW_RC, -1), (YAW_RC, roof(YAW_RC) + GAP_45), (0, ROOF_TOP + GAP_45)]
)
yaw_cap = revolved([(0, ROOF_TOP + GAP_45), (R, roof(R) + GAP_45), (R, H), (0, H)])
taper = FLANGE_R - POST_R
yaw_post = revolved(
    [(0, 0), (FLANGE_R, 0), (FLANGE_R, FLANGE_Z),
     (POST_R, FLANGE_Z + taper), (POST_R, H), (0, H)]
)
yaw_bore = revolved(
    [(0, -1), (FLANGE_R + CLR, -1), (FLANGE_R + CLR, FLANGE_Z + CLR),
     (POST_R + CLR, FLANGE_Z + CLR + taper), (POST_R + CLR, H + 1), (0, H + 1)]
)

# --- Pitch joint tools, local frame: axis = X at height R, rotor side is +y --- #
fork_x = CENTER_W / 2 + AXIAL_GAP  # inner face of each outer knuckle


def x_cylinder(x0, x1, radius):
    return Pos((x0 + x1) / 2, 0, R) * Rot(0, 90, 0) * Cylinder(radius, x1 - x0)


def x_cone(x_base, direction, r0, length):
    """Cone along X with its base at x_base, pointing towards `direction` (+-1)."""
    cone = Cone(r0, r0 - length, length, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return Pos(x_base, 0, R) * Rot(0, 90 * direction, 0) * cone


pitch_forks = x_cylinder(fork_x, W / 2, R) + x_cylinder(-W / 2, -fork_x, R)
pitch_pins = x_cone(fork_x, -1, PIN_R, PIN_L) + x_cone(-fork_x, 1, PIN_R, PIN_L)
sock_r = PIN_R + GAP_45
pitch_sockets = x_cone(fork_x, -1, sock_r, PIN_L + CLR) + x_cone(
    -fork_x, 1, sock_r, PIN_L + CLR
)
pitch_center = x_cylinder(-CENTER_W / 2, CENTER_W / 2, R)
pitch_stator_clear = box(-fork_x, fork_x, -(R + CLR), 1, -1, H + 1)
pitch_rotor_clear = box(CENTER_W / 2, W, -1, R + CLR, -1, H + 1) + box(
    -W, -CENTER_W / 2, -1, R + CLR, -1, H + 1
)


# ----------------------------------------------------------------------------
# Reading the MuJoCo specification
# ----------------------------------------------------------------------------
class SpecError(ValueError):
    """The specification cannot be turned into a printable model."""


@dataclass
class Part:
    """One module of the robot, in print coordinates (mm, core centre at 0, 0)."""

    kind: str  # "core" | "brick" | "hinge"
    name: str
    parent: int | None  # index of the module this one hangs off
    # core / brick: footprint centre and full extents
    centre: tuple[float, float] = (0.0, 0.0)
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # hinge: back-end origin, heading, joint type and half lengths
    origin: tuple[float, float] = (0.0, 0.0)
    heading: float = 0.0  # degrees; 0 = pointing along +y, 90 = along -x
    joint: str = ""  # "yaw" | "pitch"
    ls: float = 0.0  # stator length (back end to joint axis)
    lr: float = 0.0  # rotor length (joint axis to front end)
    has_child: bool = False


@dataclass
class Robot:
    parts: list[Part]
    scale: float  # printed mm per real mm
    camera_z: float | None  # camera centre height above the bed, mm (None: no camera)
    skipped: list[str] = field(default_factory=list)


def load_robot(path, core_name=None, snap45="pitch", drop_unprintable=False):
    """Read the core/hinge/brick tree from a MuJoCo XML (or .mjb) file."""
    path = Path(path)
    if path.suffix == ".zip":
        model = RerunnableRobot.load(path).mj_spec.compile()
    elif path.suffix == ".mjb":
        model = mujoco.MjModel.from_binary_path(str(path))
    else:
        model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)
    ]
    parent_of = model.body_parentid

    def children(b):
        return [c for c in range(1, model.nbody) if parent_of[c] == b]

    def ancestors(b):
        while b > 0:
            b = parent_of[b]
            yield b

    # --- find the core --- #
    cores = [b for b in range(1, model.nbody) if body_names[b].endswith("core")]
    if core_name is not None:
        cores = [b for b in cores if body_names[b] == core_name]
    if not cores:
        raise SpecError(
            f"no body whose name ends in 'core' in {path.name}"
            + (f" matches --core {core_name!r}" if core_name else "")
            + f"; bodies are: {', '.join(n for n in body_names if n)}"
        )
    if len(cores) > 1:
        raise SpecError(
            "several cores found, pick one with --core: "
            + ", ".join(body_names[b] for b in cores)
        )
    core = cores[0]

    # --- classify the modules below the core --- #
    kinds = {core: "core"}
    for b in range(1, model.nbody):
        if b == core or core not in ancestors(b):
            continue
        name = body_names[b]
        sub = [body_names[c] for c in children(b)]
        if name.endswith("hinge") and any(s.endswith("stator") for s in sub) and any(
            s.endswith("rotor") for s in sub
        ):
            kinds[b] = "hinge"
        elif name.endswith("brick"):
            kinds[b] = "brick"
    order = sorted(kinds)  # MuJoCo numbers parents before children

    # --- express everything in the core's frame --- #
    Rc = data.xmat[core].reshape(3, 3)
    pc = data.xpos[core]

    def pt(p):
        return Rc.T @ (np.asarray(p) - pc)

    def vec(v):
        return Rc.T @ np.asarray(v)

    def box_geom(b):
        for g in np.flatnonzero(model.geom_bodyid == b):
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                return g
        return None

    def aabb(g):
        """Centre and half extents (m, core frame), or None if not axis-aligned."""
        rel = Rc.T @ data.geom_xmat[g].reshape(3, 3)
        if not np.allclose(np.abs(rel).max(axis=1), 1.0, atol=1e-3):
            return None
        return pt(data.geom_xpos[g]), np.abs(rel) @ model.geom_size[g]

    def parent_module(b):
        p = parent_of[b]
        while p > 0 and p not in kinds:
            p = parent_of[p]  # skips stator / rotor / frame bodies
        return p if p > 0 else None

    # --- hinge geometry, to find the scale --- #
    hinge_data = {}
    for b in order:
        if kinds[b] != "hinge":
            continue
        sub = children(b)
        rotor = next(c for c in sub if body_names[c].endswith("rotor"))
        g = box_geom(rotor)
        jnts = [j for j in range(model.njnt) if model.jnt_bodyid[j] == rotor
                and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        if g is None or not jnts:
            raise SpecError(f"hinge {body_names[b]!r} has no box rotor geom or hinge joint")
        sx, sy, sz = model.geom_size[g]
        hinge_data[b] = (g, jnts[0], 2 * sx, 2 * sz, 2 * sy)
    widths = [v[2] for v in hinge_data.values()]
    if widths and max(widths) - min(widths) > 0.01 * max(widths):
        raise SpecError(
            "hinges of different widths are not supported: "
            + ", ".join(f"{body_names[b]}={v[2] * 1000:.1f} mm" for b, v in hinge_data.items())
        )
    if widths:
        w_real, h_real = hinge_data[next(iter(hinge_data))][2:4]
        if abs(w_real - h_real) > 0.02 * w_real:
            warnings.warn(
                f"hinge cross-section is {w_real * 1000:.1f} x {h_real * 1000:.1f} mm; "
                "the joints assume a square one and use the width"
            )
        scale = HINGE_W / (w_real * 1000)
    else:
        scale = FALLBACK_SCALE

    def mm(metres):
        return metres * 1000 * scale

    # --- core box: sets the print origin --- #
    g_core = box_geom(core)
    if g_core is None or aabb(g_core) is None:
        raise SpecError("the core needs an axis-aligned box geom")
    core_centre, core_half = aabb(g_core)
    origin_xy = core_centre[:2]
    core_bottom = core_centre[2] - core_half[2]

    def xy(p):
        return tuple(float(mm(v)) for v in (p[:2] - origin_xy))

    camera_z = None
    cams = np.flatnonzero(model.cam_bodyid == core)
    if len(cams):
        h_cam = mm(pt(data.cam_xpos[cams[0]])[2] - core_bottom)
        h_core = mm(2 * core_half[2])
        camera_z = float(min(max(h_cam, EDGE_FILLET + CAM_H / 2), h_core - EDGE_FILLET - CAM_H / 2))

    # --- build the parts --- #
    parts: list[Part] = []
    index: dict[int, int] = {}
    skipped: list[str] = []
    skipped_ids: set[int] = set()

    def reject(b, why):
        msg = f"{body_names[b]}: {why}"
        if not drop_unprintable:
            raise SpecError(msg + " (use --drop-unprintable to leave it out)")
        warnings.warn(f"dropping {msg}")
        skipped.append(msg)
        skipped_ids.add(b)

    for b in order:
        kind = kinds[b]
        par = parent_module(b)
        if par in skipped_ids:
            skipped.append(f"{body_names[b]}: hangs off a dropped module")
            skipped_ids.add(b)
            continue
        if kind == "core":
            parts.append(Part("core", body_names[b], None, (0.0, 0.0),
                              tuple(float(mm(2 * h)) for h in core_half)))
            index[b] = 0
            continue
        forward = vec(data.xmat[b].reshape(3, 3)[:, 1])
        if abs(forward[2]) > 0.05:
            reject(b, "points up or down (attached to a top/bottom face), which cannot print flat")
            continue
        if kind == "brick":
            g = box_geom(b)
            if g is None:
                reject(b, "has no box geom")
                continue
            box_ = aabb(g)
            if box_ is None:
                # Tilted about its own forward axis (it hangs off a 45-degree
                # hinge): print it upright, like the hinge it hangs off.
                sx, sy, sz = model.geom_size[g]
                quarter = degrees(atan2(-forward[0], forward[1])) / 90
                if abs(quarter - round(quarter)) * 90 > 1:
                    reject(b, "is not aligned with the core's axes")
                    continue
                half = np.array([sy, sx, sz] if round(quarter) % 2 else [sx, sy, sz])
                box_ = pt(data.geom_xpos[g]), half
                warnings.warn(f"{body_names[b]}: tilted about its axis, printed upright")
            centre, half = box_
            parts.append(Part("brick", body_names[b], index[par], xy(centre),
                              tuple(float(mm(2 * h)) for h in half)))
        else:
            g, jnt, _, _, _ = hinge_data[b]
            origin = pt(data.xpos[b])
            axis = vec(data.xaxis[jnt])
            anchor = pt(data.xanchor[jnt])
            ls = float(forward @ (anchor - origin))
            site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, body_names[b] + "-front")
            lr = (float(forward @ (pt(data.site_xpos[site]) - anchor)) if site >= 0
                  else 2 * model.geom_size[g][1])
            tilt = degrees(np.arccos(min(1.0, abs(axis[2]))))  # 0 = vertical axis
            if tilt < 22.5:
                joint = "yaw"
            elif tilt > 67.5:
                joint = "pitch"
            else:
                joint = snap45
                warnings.warn(
                    f"{body_names[b]}: joint axis is tilted {tilt:.0f} degrees from vertical, "
                    f"which has no printable joint; printed as {snap45}"
                )
            part = Part("hinge", body_names[b], index[par], origin=xy(origin),
                        heading=degrees(atan2(-forward[0], forward[1])), joint=joint,
                        ls=float(mm(ls)), lr=float(mm(lr)))
            need = R + CLR if joint == "pitch" else R
            if part.ls < need or part.lr < need:
                raise SpecError(
                    f"{body_names[b]}: stator/rotor of {part.ls:.1f}/{part.lr:.1f} mm are too "
                    f"short for a {joint} joint (need at least {need:.1f} mm each)"
                )
            parts.append(part)
        index[b] = len(parts) - 1
        parts[index[par]].has_child = True
    return Robot(parts, scale, camera_z, skipped)


# ----------------------------------------------------------------------------
# Building the model
# ----------------------------------------------------------------------------
def key_loop(core_h):
    """Upright tab on top of the core with a teardrop hole (prints unsupported)."""
    z0, zc = core_h - 1, core_h + LOOP_W / 2 + 1
    outline = Pos(0, (z0 + zc) / 2) * Rectangle(LOOP_W, zc - z0) + Pos(0, zc) * Circle(
        LOOP_W / 2
    )
    k = LOOP_HOLE_R / sqrt(2)
    hole = Pos(0, zc) * (
        Circle(LOOP_HOLE_R)
        + Polygon((-k, k), (0, LOOP_HOLE_R * sqrt(2)), (k, k), (0, 0), align=None)
    )
    return extrude(Plane.XZ * (outline - hole), LOOP_T / 2, both=True)


def camera(face_y, cam_z):
    """Camera module on the front (+y) face: housing, lens barrel, two screws."""
    zb, zt = cam_z - CAM_H / 2, cam_z + CAM_H / 2
    side = Polygon((-0.2, zb - CAM_D), (CAM_D, zb), (CAM_D, zt), (-0.2, zt), align=None)
    housing = Pos(0, face_y, 0) * extrude(Plane.YZ * side, CAM_W / 2, both=True)

    front = Pos(0, face_y + CAM_D, cam_z)
    to_y = Rot(-90, 0, 0)  # local +Z -> global +y
    lens = Cone(LENS_R, LENS_R - 0.3, LENS_L, align=(Align.CENTER, Align.CENTER, Align.MIN))
    lens_hole = Pos(0, 0, LENS_L) * Cylinder(LENS_HOLE_R, 1.2)
    screws = [Pos(x, 0, 0) * Cylinder(SCREW_R, 0.6) for x in (-SCREW_X, SCREW_X)]
    return housing + front * to_y * lens - front * to_y * (lens_hole + screws[0] + screws[1])


@dataclass
class Eye:
    """Where the camera looks: a flat 45-degree face cut into a vertical core corner."""

    corner: str  # "front" or "{front,back}-{left,right}"
    kx: int = 0  # corner signs (0 for the flat front face)
    ky: int = 1
    heading: float = 0.0  # degrees, Rot(0, 0, heading) turns +y towards the camera

    @classmethod
    def parse(cls, corner):
        if corner == "front":
            return cls("front")
        try:
            fb, lr = corner.split("-")
            ky, kx = {"front": 1, "back": -1}[fb], {"right": 1, "left": -1}[lr]
        except (ValueError, KeyError):
            raise SpecError(f"unknown camera corner {corner!r}") from None
        heading = degrees(atan2(-kx, ky))  # +y -> (kx, ky) / sqrt(2)
        return cls(corner, kx, ky, heading)

    @property
    def diagonal(self):
        return self.corner != "front"

    def face_cut(self, sx, sy, face_w):
        """Leg length of the corner cut that leaves a flat face `face_w` wide."""
        c = face_w / sqrt(2)
        for half in (sx / 2, sy / 2):
            if c > half - R:
                raise SpecError(
                    f"core of {sx:.1f} x {sy:.1f} mm is too small for a diagonal camera "
                    f"without reaching the legs (use --camera-corner front)"
                )
        return c

    def plane(self, sx, sy, c):
        """Chamfer plane n . p = d, n pointing towards the camera."""
        n = (self.kx / sqrt(2), self.ky / sqrt(2))
        return n, (sx / 2 + sy / 2 - c) / sqrt(2)


def cut_corner(core, eye, sx, sy, sz, c):
    """Remove the vertical corner of the core, leaving a flat 45-degree face."""
    a, b, kx, ky = sx / 2, sy / 2, eye.kx, eye.ky
    pts = [(kx * (a - c), ky * b), (kx * (a - c), ky * (b + 1)),
           (kx * (a + 1), ky * (b + 1)), (kx * (a + 1), ky * (b - c)), (kx * a, ky * (b - c))]
    return core - extrude(Pos(0, 0, -1) * Polygon(*pts, align=None), 0.1)#sz + 2)


def mount_camera(core, eye, sx, sy, sz, cam_z):
    """Add the camera module on the front face or on a cut corner of the core."""
    if not eye.diagonal:
        if CAM_W > sx - 2 * EDGE_FILLET:
            raise SpecError(f"core of {sx:.1f} mm is too narrow for the camera module")
        return core + camera(sy / 2, cam_z)
    c = eye.face_cut(sx, sy, CAM_W + 2 * CAM_FACE_MARGIN)
    core = cut_corner(core, eye, sx, sy, sz, c)
    centre = (eye.kx * (sx - c) / 2, eye.ky * (sy - c) / 2)  # middle of the flat face
    return core + Pos(centre[0], centre[1], 0) * Rot(0, 0, eye.heading) * camera(0, cam_z)


def engraving(text, sx, sy):
    """Text on the bottom face, laid out as seen from below with its top towards +y.

    Text too big for the core is shrunk to fit inside TEXT_MARGIN.
    """
    from_below = Plane((0, 0, 0), x_dir=(+1, +1, 0), z_dir=(0, 0, -1))
    size = TEXT_SIZE
    for _ in range(3):
        shape = Text(text, size, font=TEXT_FONT, font_style=FontStyle.BOLD)
        bb = shape.bounding_box()
        k = min((sx - 2 * TEXT_MARGIN) / bb.size.X, (sy - 2 * TEXT_MARGIN) / bb.size.Y)
        if k >= 0.999:
            break
        size *= k * 0.99
    if size < TEXT_SIZE:
        print(f"text {text!r} shrunk to size {size:.1f} mm to fit under the core")
    return extrude(from_below * shape, TEXT_DEPTH, dir=(0, 0, 1))


def find_logo(name, extra_dirs=()):
    """Locate the logo SVG: as given, then next to the script, the cwd, the spec."""
    if name is None:
        return None
    for d in (Path(), SCRIPT_DIR, Path.cwd(), *extra_dirs):
        if (d / name).exists():
            return d / name
    print(f"no {name} found, skipping the logo")
    return None


def logo(path, core_h, sx, sy, eye):
    """Raised logo from an SVG on the top face, with its top towards the camera."""
    faces = [s for s in import_svg(str(path)) if isinstance(s, Face)]
    if not faces:
        raise ValueError(f"{path.name} has no filled shapes (outline-only SVG?)")
    # Illustrator exports can leave holes touching their outline, which gives
    # an invalid face; rebuilding it as outline minus holes repairs that.
    faces = [
        f if f.is_valid else Face(f.outer_wire()) - [Face(w) for w in f.inner_wires()]
        for f in faces
    ]
    shape = Compound(children=faces)
    bb = shape.bounding_box()

    shape = Compound(children=faces)
    bb = shape.bounding_box()
    shape = shape.scale(LOGO_SIZE / max(bb.size.X, bb.size.Y))
    centre = shape.bounding_box().center()
    shape = shape.translate((-centre.X, -centre.Y, core_h - centre.Z))
    return extrude(shape, LOGO_HEIGHT)


def fuse(shapes):
    out = shapes[0]
    for s in shapes[1:]:
        out = out + s
    return out


@dataclass
class Group:
    """Everything that ends up in one rigid printed solid."""

    boxes: list = field(default_factory=list)  # hinge halves, cut by `clears`
    clears: list = field(default_factory=list)
    adds: list = field(default_factory=list)  # knuckles, pins, posts, caps, rings
    bores: list = field(default_factory=list)  # sockets and bores, cut after `adds`
    fixed: list = field(default_factory=list)  # cores and bricks, never cut

    def solid(self):
        parts = []
        if self.boxes:
            s = fuse(self.boxes)
            for tool in self.clears:
                s = s - tool
            for tool in self.adds:
                s = s + tool
            for tool in self.bores:
                s = s - tool
            parts.append(s)
        parts += self.fixed
        return fuse(parts)


def build(robot, text=TEXT, with_loop=True, logo_path=None, camera_corner=CAM_CORNER):
    """Return the list of rigid solids (core group first)."""
    eye = Eye.parse(camera_corner)
    groups: list[Group] = []
    attach: list[int] = []  # group that children of part i are fused to

    def new_group():
        groups.append(Group())
        return len(groups) - 1

    for i, p in enumerate(robot.parts):
        if p.kind in ("core", "brick"):
            gid = new_group() if p.kind == "core" else attach[p.parent]
            attach.append(gid)
            (cx, cy), (sx, sy, sz) = p.centre, p.size
            body = rounded_box(cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, sz)
            if p.kind == "core":
                if robot.camera_z is not None:
                    body = mount_camera(body, eye, sx, sy, sz, robot.camera_z)
                if text:
                    body = body - engraving(text, sx, sy)
                if with_loop:
                    body = body + key_loop(sz)
                elif logo_path is not None:
                    body = body + logo(logo_path, sz, sx, sy, eye)
            groups[gid].fixed.append(body)
            continue

        # hinge: stator joins the parent's group, rotor starts a new one
        stator_g, rotor_g = groups[attach[p.parent]], None
        rotor_id = new_group()
        rotor_g = groups[rotor_id]
        attach.append(rotor_id)
        frame = Pos(p.origin[0], p.origin[1], 0) * Rot(0, 0, p.heading)
        joint = frame * Pos(0, p.ls, 0)
        stator_g.boxes.append(frame * box(-R, R, -FUSE_OVERLAP, p.ls, 0, H))
        rotor_end = p.ls + p.lr + (FUSE_OVERLAP if p.has_child else 0)
        rotor_g.boxes.append(frame * box(-R, R, p.ls, rotor_end, 0, H))
        if p.joint == "yaw":
            stator_g.clears.append(joint * yaw_stator_clear)
            stator_g.adds.append(joint * (yaw_cap + yaw_post))
            rotor_g.clears.append(joint * yaw_rotor_clear)
            rotor_g.adds.append(joint * yaw_ring)
            rotor_g.bores.append(joint * yaw_bore)
        else:
            stator_g.clears.append(joint * pitch_stator_clear)
            stator_g.adds.append(joint * (pitch_forks + pitch_pins))
            rotor_g.clears.append(joint * pitch_rotor_clear)
            rotor_g.adds.append(joint * pitch_center)
            rotor_g.bores.append(joint * pitch_sockets)
    return [g.solid() for g in groups]


def check_interference(solids, tolerance=1e-3):
    """Warn about solids that overlap (they would print fused together)."""
    for i, a in enumerate(solids):
        for j in range(i + 1, len(solids)):
            b = solids[j]
            ba, bb = a.bounding_box(), b.bounding_box()
            if (ba.max.X < bb.min.X or bb.max.X < ba.min.X or ba.max.Y < bb.min.Y
                    or bb.max.Y < ba.min.Y or ba.max.Z < bb.min.Z or bb.max.Z < ba.min.Z):
                continue
            inter = a.intersect(b)
            volume = inter.volume if inter else 0.0
            if volume > tolerance:
                warnings.warn(
                    f"solids {i} and {j} overlap by {volume:.2f} mm^3 and will print "
                    "fused together (limbs too close at the reference pose?)"
                )


def export(name, out_dir, solids):
    model = Compound(children=solids)
    export_step(model, str(out_dir / f"{name}.step"))
    stl_path = out_dir / f"{name}.stl"
    export_stl(model, str(stl_path), tolerance=0.005, angular_tolerance=0.1)

    # OCCT leaves zero-area slivers at the poles of the filleted cube corners;
    # weld them shut so the STL is made of watertight bodies.
    mesh = trimesh.load(stl_path)
    mesh.merge_vertices(digits_vertex=3)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight:
        warnings.warn(f"{name}.stl is not watertight; check it in the slicer")
    if mesh.body_count != len(solids):
        warnings.warn(
            f"{name}.stl has {mesh.body_count} separate bodies, expected {len(solids)}: "
            "some solids touch each other"
        )
    mesh.export(stl_path)

    bb = model.bounding_box()
    print(f"{name}: {bb.size.X:.1f} x {bb.size.Y:.1f} x {bb.size.Z:.1f} mm, "
          f"{len(solids)} solids")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("spec", type=Path, help="MuJoCo XML (or .mjb) of the robot")
    ap.add_argument("--core", help="name of the core body, if the file holds several robots")
    ap.add_argument("--name", help="output base name (default: the spec's file name)")
    ap.add_argument("--out-dir", type=Path, help="output folder (default: next to the spec)")
    ap.add_argument("--text", default=TEXT, help="text engraved in the bottom of the core")
    ap.add_argument("--logo", default=LOGO_SVG, help="SVG for the desk toy's top face")
    ap.add_argument("--variant", choices=("both", "keychain", "desk"), default="both")
    ap.add_argument("--snap45", choices=("yaw", "pitch"), default="pitch",
                    help="how to print hinges tilted by 45 degrees")
    ap.add_argument("--drop-unprintable", action="store_true",
                    help="leave out modules that cannot print flat instead of failing")
    ap.add_argument("--no-check", action="store_true", help="skip the overlap check")
    args = ap.parse_args()

    try:
        robot = load_robot(args.spec, args.core, args.snap45, args.drop_unprintable)
    except SpecError as e:
        ap.error(str(e))
    count = lambda kind: sum(p.kind == kind for p in robot.parts)
    hinges = [p for p in robot.parts if p.kind == "hinge"]
    print(f"{args.spec.name}: 1 core, {count('brick')} bricks, {len(hinges)} hinges "
          f"({sum(p.joint == 'yaw' for p in hinges)} yaw, "
          f"{sum(p.joint == 'pitch' for p in hinges)} pitch), "
          f"scale 1:{1 / robot.scale:.2f}")

    out_dir = args.out_dir or args.spec.resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.name or args.spec.stem
    if args.variant in ("both", "keychain"):
        solids = build(robot, args.text, with_loop=True)
        if not args.no_check:
            check_interference(solids)
        export(f"{base}_keychain", out_dir, solids)
    if args.variant in ("both", "desk"):
        logo_path = find_logo(args.logo, (args.spec.resolve().parent,))
        solids = build(robot, args.text, with_loop=False, logo_path=logo_path)
        export(f"{base}_desk_toy", out_dir, solids)


if __name__ == "__main__":
    main()
