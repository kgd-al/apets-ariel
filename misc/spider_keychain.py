"""Print-in-place keychain / desk toy of the ARIEL "spider" robot.

Mirrors ``ariel/body_phenotypes/robogen_lite/prebuilt_robots/spider.py``:
a core with four legs, each leg = yaw hinge -> pitch hinge -> brick.
Real module sizes come from ``ArielModulesConfig`` and are scaled by SCALE.

The model prints flat on the bed in one piece, without supports. It consists of
nine separate solids (core + 4 middle links + 4 feet) that are captive in
each other, so the eight joints move straight off the printer.

    * Yaw joints (vertical axis): a stepped post with a flange at its base,
      under a 45-degree cone-shaped cap. Every downward-facing surface is at
      most 45 degrees.
    * Pitch joints (horizontal axis): a three-knuckle hinge with cone-shaped
      pins in cone-shaped sockets.
    * Both joints stop at roughly +-90 degrees, like the SER0019 servos.
    * The bottom face carries an engraved TEXT; the desk toy can carry a raised
      logo (LOGO_SVG) on its top face.
    * The front face carries a camera module at the height of the core's
      MuJoCo camera (which has no geometry of its own).

Deviation from the simulator: the legs sit flush with the bed instead of at
the core's -0.25 side-site height. Otherwise the hinges would float 1.5 mm
above the bed and need supports.

Requires ``pip install build123d trimesh``. Run ``python spider_keychain.py`` to
regenerate STEP + STL for both variants (with and without the key ring loop)
next to this file.
"""

from math import sqrt
from pathlib import Path

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

# --- Real robot dimensions (mm), from ArielModulesConfig --- #
REAL_CORE = 150.0
REAL_BRICK = 75.0
REAL_HINGE_W = 52.0  # hinge cross-section
REAL_HINGE_HALF_L = 37.5  # stator length == rotor length

# --- Keychain parameters (mm) --- #
SCALE = 1 / 7.5  # ~80 mm tip to tip
CLR = 0.30  # radial / normal clearance between moving parts
AXIAL_GAP = 0.35  # gap between pitch-hinge knuckles
EDGE_FILLET = 1.0  # rounding of core / brick edges

# Yaw joint (vertical axis)
YAW_ROOF_Z = 2.4  # height of the rotor ring rim before the 45 deg roof
POST_R = 1.2  # post shaft radius
FLANGE_R = 2.0  # retaining flange radius at the post foot
FLANGE_Z = 0.6  # flange straight height before its 45 deg taper

# Pitch joint (horizontal axis)
FORK_W = 1.7  # width of each outer knuckle
PIN_R = 1.5  # cone pin base radius
PIN_L = 1.0  # cone pin length (45 deg, truncated)

# Camera module on the front face (MuJoCo: 20 mm below the core top, real scale)
CAM_W = 8.0  # housing width
CAM_H = 4.5  # housing height
CAM_D = 1.2  # housing protrusion (underside is a 45 deg chamfer of this depth)
CAM_Z = 16.8  # housing centre height; just below the core's top fillet
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

# Logo on the core's top face (desk toy only: the key ring loop sits there).
# Drop an SVG with filled shapes next to this file; it is scaled to fit LOGO_SIZE.
LOGO_SVG = "vu_logo.svg"
LOGO_SIZE = 17.5  # longest side, mm; the flat top is 18 mm (20 mm minus 1 mm fillets)
LOGO_HEIGHT = 0.6  # raised by 3 layers; set a colour change at this height to make it pop

# Key ring loop on top of the core
LOOP_W = 10.0
LOOP_T = 3.0
LOOP_HOLE_R = 2.2

OUT_DIR = Path(__file__).parent

# --- Derived geometry --- #
CORE = REAL_CORE * SCALE
BRICK = REAL_BRICK * SCALE
W = H = REAL_HINGE_W * SCALE  # hinge cross-section (width == height)
R = W / 2  # yaw ring radius == pitch knuckle radius
HALF_L = REAL_HINGE_HALF_L * SCALE

# Leg stations along the leg's own +y axis (core centre at origin)
CORE_FACE = CORE / 2
YAW_Y = CORE_FACE + HALF_L  # 1st hinge: stator | rotor split
PITCH_Y = CORE_FACE + 3 * HALF_L  # 2nd hinge: stator | rotor split
BRICK_Y = CORE_FACE + 4 * HALF_L

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


def rounded_box(size, y0):
    """Cube on the bed, centred in x, starting at y0, top/vertical edges filleted."""
    b = box(-size / 2, size / 2, y0, y0 + size, 0, size)
    edges = b.edges().filter_by(Axis.Z) + b.edges().group_by(Axis.Z)[-1]
    return fillet(edges, EDGE_FILLET)


def revolved(pts):
    """Revolve an (r, z) polygon about the local Z axis."""
    return revolve(Plane.XZ * Polygon(*pts, align=None), Axis.Z)


def roof(r):
    """Top surface height of the yaw rotor ring at radius r (45 deg cone)."""
    return ROOF_TOP - r


# --- Yaw joint, local frame: axis = Z through origin, rotor side is +y --- #
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

# --- Pitch joint, local frame: axis = X at height R, rotor side is +y --- #
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

AT_YAW = Pos(0, YAW_Y, 0)
AT_PITCH = Pos(0, PITCH_Y, 0)


def leg_root():
    """Yaw stator: fused to the core."""
    stator = box(-R, R, CORE_FACE - 0.5, YAW_Y, 0, H) - AT_YAW * yaw_stator_clear
    return stator + AT_YAW * (yaw_cap + yaw_post)


def leg_middle():
    """Yaw rotor + pitch stator (hinge 1 rotor fused to hinge 2 stator)."""
    link = box(-R, R, YAW_Y, PITCH_Y, 0, H) - AT_YAW * yaw_rotor_clear
    link = (link + AT_YAW * yaw_ring) - AT_YAW * yaw_bore
    link -= AT_PITCH * pitch_stator_clear
    return link + AT_PITCH * (pitch_forks + pitch_pins)


def leg_foot():
    """Pitch rotor + brick."""
    link = box(-R, R, PITCH_Y, BRICK_Y + 0.5, 0, H) - AT_PITCH * pitch_rotor_clear
    link = (link + AT_PITCH * pitch_center) - AT_PITCH * pitch_sockets
    return link + rounded_box(BRICK, BRICK_Y)


def key_loop():
    """Upright tab on top of the core with a teardrop hole (prints unsupported)."""
    z0, zc = CORE - 1, CORE + LOOP_W / 2 + 1
    outline = Pos(0, (z0 + zc) / 2) * Rectangle(LOOP_W, zc - z0) + Pos(0, zc) * Circle(
        LOOP_W / 2
    )
    k = LOOP_HOLE_R / sqrt(2)
    hole = Pos(0, zc) * (
        Circle(LOOP_HOLE_R)
        + Polygon((-k, k), (0, LOOP_HOLE_R * sqrt(2)), (k, k), (0, 0), align=None)
    )
    return extrude(Plane.XZ * (outline - hole), LOOP_T / 2, both=True)


def camera():
    """Camera module on the front (+y) face: housing, lens barrel, two screws."""
    zb, zt = CAM_Z - CAM_H / 2, CAM_Z + CAM_H / 2
    side = Polygon((-0.2, zb - CAM_D), (CAM_D, zb), (CAM_D, zt), (-0.2, zt), align=None)
    housing = Pos(0, CORE_FACE, 0) * extrude(Plane.YZ * side, CAM_W / 2, both=True)

    front = Pos(0, CORE_FACE + CAM_D, CAM_Z)
    to_y = Rot(-90, 0, 0)  # local +Z -> global +y
    lens = Cone(LENS_R, LENS_R - 0.3, LENS_L, align=(Align.CENTER, Align.CENTER, Align.MIN))
    lens_hole = Pos(0, 0, LENS_L) * Cylinder(LENS_HOLE_R, 1.2)
    screws = [Pos(x, 0, 0) * Cylinder(SCREW_R, 0.6) for x in (-SCREW_X, SCREW_X)]
    return housing + front * to_y * lens - front * to_y * (lens_hole + screws[0] + screws[1])


def engraving():
    """TEXT on the bottom face, laid out as seen from below with its top towards +y."""
    from_below = Plane((0, 0, 0), x_dir=(-1, 0, 0), z_dir=(0, 0, -1))
    text = Text(TEXT, TEXT_SIZE, font=TEXT_FONT, font_style=FontStyle.BOLD)
    return extrude(from_below * text, TEXT_DEPTH, dir=(0, 0, 1))


def logo():
    """Raised logo from LOGO_SVG on the top face, or None if the file is missing."""
    path = OUT_DIR / LOGO_SVG
    if not path.exists():
        print(f"no {LOGO_SVG} found, skipping the logo")
        return None
    faces = [s for s in import_svg(str(path)) if isinstance(s, Face)]
    if not faces:
        raise ValueError(f"{LOGO_SVG} has no filled shapes (outline-only SVG?)")
    # Illustrator exports can leave holes touching their outline, which gives
    # an invalid face; rebuilding it as outline minus holes repairs that.
    faces = [
        f if f.is_valid else Face(f.outer_wire()) - [Face(w) for w in f.inner_wires()]
        for f in faces
    ]
    shape = Compound(children=faces)
    bb = shape.bounding_box()
    shape = shape.scale(LOGO_SIZE / max(bb.size.X, bb.size.Y))
    centre = shape.bounding_box().center()
    shape = shape.translate((-centre.X, -centre.Y, CORE - centre.Z))
    return extrude(shape, LOGO_HEIGHT)


def build(with_loop=True):
    """Return (core, [middle links], [feet]) for all four legs."""
    rotations = [Rot(0, 0, 90 * i) for i in range(4)]  # front, left, back, right
    core = rounded_box(CORE, -CORE / 2) + camera() - engraving()
    if with_loop:
        core += key_loop()
    elif (raised := logo()) is not None:
        core += raised
    root, middle, foot = leg_root(), leg_middle(), leg_foot()
    for rot in rotations:
        core += rot * root
    return core, [rot * middle for rot in rotations], [rot * foot for rot in rotations]


def export(name, with_loop):
    core, middles, feet = build(with_loop)
    spider = Compound(children=[core, *middles, *feet])
    export_step(spider, str(OUT_DIR / f"{name}.step"))
    stl_path = OUT_DIR / f"{name}.stl"
    export_stl(spider, str(stl_path), tolerance=0.005, angular_tolerance=0.1)

    # OCCT leaves zero-area slivers at the poles of the filleted cube corners;
    # weld them shut so the STL is 9 watertight bodies.
    mesh = trimesh.load(stl_path)
    mesh.merge_vertices(digits_vertex=3)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    assert mesh.is_watertight, "STL is not watertight"
    mesh.export(stl_path)

    bb = spider.bounding_box()
    print(f"{name}: {bb.size.X:.1f} x {bb.size.Y:.1f} x {bb.size.Z:.1f} mm")


if __name__ == "__main__":
    export("spider_keychain", with_loop=True)
    export("spider_desk_toy", with_loop=False)
