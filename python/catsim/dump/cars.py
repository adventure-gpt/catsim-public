"""Cars: handling presets (hitbox, wheels) and car bodies, read from the game's packages.

Where the numbers live (our own description):
- A handling preset (`HandlingPreset_TA`, in Startup.upk and in some bodies' own packages) lists its axles'
  WheelRadius and LocalRestPosition where they differ from the class defaults (TAGame.upk), and names a
  physics asset whose one body setup has one box: its size X, Y, Z and its matrix's translation t.
- A car body (`ProductAsset_Body_TA`, in its Body_*_SF.upk) names its preset, its axles' wheel radii and
  bone names, and its skeletal mesh (bounds, origin, reference skeleton).

How the physics gets its numbers from those (float32, each step rounded, in this order):
    s     = 1.0 * 0.02
    half  = (((s * X) * 0.5 + 0.025) * 50.0) * 0.02        (same for Y, Z)
    off   = ((s * t) * 50.0) * 0.02                        (x and z; y is 0)
    min hit distance squared = ((Z * 0.5 - 1.0) * 0.02)^2
    wheel (per axle, right wheel; the left one mirrors y):
      hx = X * 0.5 - 2.0, hy = Y * 0.5 - 2.0
      cx = clamp(rest.x, t.x - hx, hx + t.x), cy = clamp(rest.y, t.y - hy, hy + t.y), cz = t.z
      connection = (cx, cy, cz) * 0.02, radius = WheelRadius * 0.02, rest = (cz - rest.z) * 0.02
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from .f32 import C002, C0025, F, add, bits, frombits, mul, sub
from .props import export_by_path, obj_props, struct_array, struct_props
from .upk import Package, PackageError


@dataclass
class Axle:
    radius: float
    rest: tuple[float, float, float]


@dataclass
class Preset:
    name: str
    box: tuple[float, float, float]        # X, Y, Z as stored
    t: tuple[float, float, float]          # the box matrix's translation
    front: Axle
    back: Axle

    def row(self) -> list[int]:
        X, Y, Z = self.box
        s = mul(F(1.0), C002)
        half = [mul(mul(add(mul(mul(s, v), 0.5), C0025), 50.0), C002) for v in (X, Y, Z)]
        off = [mul(mul(mul(s, v), 50.0), C002) for v in (self.t[0], self.t[2])]
        m = mul(sub(mul(Z, 0.5), 1.0), C002)
        words = [bits(v) for v in half] + [bits(v) for v in off] + [bits(mul(m, m))]
        hx, hy = sub(mul(X, 0.5), 2.0), sub(mul(Y, 0.5), 2.0)
        for ax in (self.front, self.back):
            rx, ry, rz = ax.rest
            cx = min(max(rx, sub(self.t[0], hx)), add(hx, self.t[0]))
            cy = min(max(ry, sub(self.t[1], hy)), add(hy, self.t[1]))
            cz = self.t[2]
            words += [bits(mul(cx, C002)), bits(mul(cy, C002)), bits(mul(cz, C002)),
                      bits(mul(sub(cz, rz), C002)), bits(mul(ax.radius, C002)), bits(rx), bits(ry)]
        return words

    def phys_box_row(self) -> list[int]:
        return [bits(self.box[0]), bits(self.box[1]), bits(self.box[2]), bits(self.t[0]), bits(self.t[2])]


def _axle(pkg: Package, props: dict, name: str, default: Axle) -> Axle:
    q = props.get(name)
    if q is None:
        return default
    f = struct_props(pkg, q.raw)
    r = f["WheelRadius"].value if "WheelRadius" in f else default.radius
    rest = struct.unpack("<3f", f["LocalRestPosition"].raw) if "LocalRestPosition" in f else default.rest
    return Axle(r, rest)


def preset_defaults(tagame: Package) -> tuple[Axle, Axle]:
    e = next((e for e in tagame.exports if e.name == "Default__HandlingPreset_TA"), None)
    if e is None:
        raise PackageError("TAGame.upk has no HandlingPreset_TA defaults")
    p = obj_props(tagame, e)
    zero = Axle(0.0, (0.0, 0.0, 0.0))
    return _axle(tagame, p, "FrontAxle", zero), _axle(tagame, p, "BackAxle", zero)


def _box(pkg: Package, phys_path: str) -> tuple[tuple, tuple]:
    pa = export_by_path(pkg, phys_path)
    if pa is None:
        raise PackageError(f"physics asset {phys_path} not in {pkg.path.name}")
    raw = obj_props(pkg, pa)["BodySetup"].raw
    n, ref = struct.unpack_from("<ii", raw, 0)
    if n != 1:
        raise PackageError(f"{phys_path}: {n} body setups, expected 1")
    setup = obj_props(pkg, pkg.exports[ref - 1])
    boxes = struct_array(pkg, struct_props(pkg, setup["AggGeom"].raw)["BoxElems"].raw)
    if len(boxes) != 1:
        raise PackageError(f"{phys_path}: {len(boxes)} boxes, expected 1")
    b = boxes[0]
    m = struct.unpack("<16f", b["TM"].raw)
    # The matrix as stored reads (0,1,0,0)(0,0,1,0)(0,0,0,1)(1,tx,ty,tz): identity rows, translation last.
    if m[:13] != (0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1):
        raise PackageError(f"{phys_path}: the box is rotated or scaled: unsupported")
    X, Y, Z = (struct.unpack("<f", b[k].raw)[0] for k in ("X", "Y", "Z"))
    return (X, Y, Z), (m[13], m[14], m[15])


def presets_in(pkg: Package, defaults: tuple[Axle, Axle]) -> list[Preset]:
    out = []
    for e in pkg.exports:
        if pkg.class_of(e) != "HandlingPreset_TA" or e.name.startswith("Default__"):
            continue
        p = obj_props(pkg, e)
        if "PhysAsset" not in p:
            continue
        box, t = _box(pkg, pkg.obj_path(p["PhysAsset"].value))
        out.append(Preset(e.name, box, t, _axle(pkg, p, "FrontAxle", defaults[0]), _axle(pkg, p, "BackAxle", defaults[1])))
    return out


# Car bodies catSim knows: product id (the number RLBot loadouts use), name, package, product asset.
# A public list, not game data: the numbers each body brings are read from its package.
BODY_LIST = [
    (23, "Octane", "Body_Octane_SF.upk", "Body_Octane"),
    (31, "Venom", "Body_Venom_SF.upk", "Body_Venom"),
    (22, "Force", "Body_Force_SF.upk", "Body_Force"),
    (30, "Vanquish", "Body_Vanquish_SF.upk", "Body_Vanquish"),
    (403, "MuscleCar", "Body_MuscleCar_SF.upk", "Body_MuscleCar"),
    (803, "Darkcar", "Body_Darkcar_SF.upk", "Body_Darkcar"),
    (24, "Orion", "Body_Orion_SF.upk", "Body_Orion"),
    (7052, "Cured", "body_cured_SF.upk", "body_cured"),
    (9427, "Skunkz", "body_skunkz_SF.upk", "body_skunkz"),
    (7012, "Future", "body_future_SF.upk", "body_future"),
    (8669, "AA", "body_aa_SF.upk", "body_aa"),
    (10697, "Hero", "body_hero_SF.upk", "body_hero"),
    (7901, "Pete", "body_pete_SF.upk", "body_pete"),
    (10698, "Smoke", "body_smoke_SF.upk", "body_smoke"),
    (9388, "Pixie", "body_pixie_SF.upk", "body_pixie"),
    (4780, "BB", "body_bb_SF.upk", "body_bb"),
    (4906, "Lemonade", "body_lemonade_SF.upk", "body_lemonade"),
    (5039, "Lemonade_Tier3", "body_lemonade_SF.upk", "Body_Lemonade_Tier3"),
    (4782, "Psyclops", "body_tritip_SF.upk", "body_tritip"),
]


@dataclass
class Skeleton:
    bounds: tuple            # origin xyz, extent xyz, radius: float32 bits
    origin: tuple[float, float, float]
    bones: dict              # name -> (parent index, local position (floats), index)
    order: list


def skeleton(pkg: Package, mesh_path: str) -> Skeleton:
    """A skeletal mesh's bounds, origin and reference skeleton, from its native data after the tagged
    properties: bounds (7 floats), materials (count + refs), origin, rotation origin, then the bones
    (name, flags, orientation quaternion, position, child count, parent index, colour: 52 bytes each)."""
    from .upk import export_props
    e = export_by_path(pkg, mesh_path)
    if e is None or pkg.class_of(e) != "SkeletalMesh":
        raise PackageError(f"skeletal mesh {mesh_path} not found")
    b = pkg.export_bytes(e)
    _, at = export_props(pkg, e)
    bounds = struct.unpack_from("<7I", b, at)
    at += 28
    at += 4 + 4 * struct.unpack_from("<i", b, at)[0]
    origin = struct.unpack_from("<3f", b, at)
    at += 24
    n = struct.unpack_from("<i", b, at)[0]
    at += 4
    if not 0 < n < 512:
        raise PackageError(f"{mesh_path}: implausible bone count {n}")
    bones, order = {}, []
    for i in range(n):
        ni, nn = struct.unpack_from("<ii", b, at)
        q = struct.unpack_from("<4f", b, at + 12)
        if any(abs(v) > 1e-6 for v in q[:3]) or abs(abs(q[3]) - 1.0) > 1e-6:
            pass  # rotated bones are fine for names we don't use; checked where used
        pos = struct.unpack_from("<3f", b, at + 28)
        _nc, parent = struct.unpack_from("<ii", b, at + 40)
        name = pkg.fname(ni, nn)
        bones[name] = (parent, pos, i, q)
        order.append(name)
        at += 52
    return Skeleton(bounds, origin, bones, order)


def bone_z(sk: Skeleton, name: str) -> float:
    """A bone's reference-pose height in the mesh: its local z plus its parents', root first, in float32.
    Only unrotated chains are supported (every car body's wheel bones are)."""
    chain = []
    i = sk.bones[name][2]
    while True:
        nm = sk.order[i]
        parent, pos, idx, q = sk.bones[nm]
        if any(abs(v) > 1e-6 for v in q[:3]):
            raise PackageError(f"bone {nm} is rotated: unsupported")
        chain.append(pos[2])
        if idx == 0 or parent == idx:
            break
        i = parent
    z = 0.0
    for v in reversed(chain):
        z = add(z, v)
    return z


def compose_z(sk: Skeleton, name: str) -> float:
    """A bone's height in the mesh's reference pose: each bone's local transform (its quaternion and
    position as a matrix) multiplied in float32 from the bone itself up to the root, leaf first."""
    from .rot import quat_to_basis  # noqa: F401  (same float rules)

    def local(nm):
        _parent, pos, _idx, q = sk.bones[nm]
        X, Y, Z, W = q
        x2, y2, z2 = add(X, X), add(Y, Y), add(Z, Z)
        xx, xy, xz = mul(X, x2), mul(X, y2), mul(X, z2)
        yy, yz, zz = mul(Y, y2), mul(Y, z2), mul(Z, z2)
        wx, wy, wz = mul(W, x2), mul(W, y2), mul(W, z2)
        return [[sub(1.0, add(yy, zz)), add(xy, wz), sub(xz, wy), 0.0],
                [sub(xy, wz), sub(1.0, add(xx, zz)), add(yz, wx), 0.0],
                [add(xz, wy), sub(yz, wx), sub(1.0, add(xx, yy)), 0.0],
                [pos[0], pos[1], pos[2], 1.0]]

    def mm(A, B):
        return [[add(add(add(mul(A[i][0], B[0][j]), mul(A[i][1], B[1][j])), mul(A[i][2], B[2][j])), mul(A[i][3], B[3][j]))
                 for j in range(4)] for i in range(4)]

    i = sk.bones[name][2]
    m = None
    while True:
        nm = sk.order[i]
        m = local(nm) if m is None else mm(m, local(nm))
        parent, _pos, idx, _q = sk.bones[nm]
        if idx == 0:
            break
        i = parent
    return m[3][2]


def curve_points(pkg: Package, raw: bytes) -> list[tuple[int, int]]:
    """An InterpCurveFloat's points as (InVal, OutVal) float bits."""
    pts = struct_array(pkg, struct_props(pkg, raw)["Points"].raw)
    return [(struct.unpack("<I", p["InVal"].raw)[0] if "InVal" in p else 0,
             struct.unpack("<I", p["OutVal"].raw)[0] if "OutVal" in p else 0) for p in pts]


def _cr(fn: str, x: float) -> float:
    """tanf / atanf, correctly rounded (what reproduces the game's folded keys; checked on every preset)."""
    try:
        import mpmath as mp
        mp.mp.prec = 120
        return F(float(getattr(mp, fn)(mp.mpf(x))))
    except ImportError:
        import math
        return F(getattr(math, fn)(x))


def fold_key(value_bits: int, front_rest: tuple[float, float], back_rest_x: float) -> int:
    """One steer key, folded once at car setup with the wheels' layout (the mean of the inner and the outer
    wheel's angle), in float32, in this order:
        r = |value * 0.017453292|;  L = |front.x - back.x|;  W = |(-front.y) - front.y|
        a = atanf(L / (W + L / tanf(r)));  a = (a - r) + r
        key = ((a - r) * 0.5 + r) * 57.295776"""
    v = frombits(value_bits)
    r = abs(mul(v, F(0.017453292)))
    L = abs(sub(front_rest[0], back_rest_x))
    W = abs(sub(-front_rest[1], front_rest[1]))
    a = _cr("atan", F(L / add(W, F(L / _cr("tan", r)))))
    a = add(sub(a, r), r)
    return bits(mul(add(mul(sub(a, r), 0.5), r), F(57.295776)))


def fold_curve(points, preset: Preset) -> list[int]:
    return [fold_key(o, (preset.front.rest[0], preset.front.rest[1]), preset.back.rest[0]) for _i, o in points]
