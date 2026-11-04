"""Everything in tables.txt, read from the game's packages (plus a few documented rules where the packages
don't say). See docs/DUMPER.md for what comes from where."""
from __future__ import annotations

import struct
from pathlib import Path

from . import cars
from .f32 import C002, F, bits, frombits, mul
from .props import export_by_path, obj_props, struct_props
from .rot import basis_bits, scaled_basis
from .upk import Package, PackageError, export_props

NUM_PADS = 34

# Rules where the packages don't decide (each is explained in docs/PROVENANCE.md):
# - The bodies whose pickup box is their own physics box (the ones checked against the running game);
#   every other preset uses the Octane's box.
PICKUP_BOX_PRESETS = ["Octane", "Darkcar", "Plank", "Merc", "Hybrid", "MuscleCar", "LongCar", "Psyclops"]
# - Bodies with no handling preset of their own get the one whose physics box theirs equals, with that preset's
#   wheel rest positions rounded to 0.01 uu (measured in the running game on the Darkcar: its wheels sit at y 27.8
#   and 20.28 where the Plank preset says 27.799904 and 20.280005; x and z are already two-decimal numbers).
PRESET_FOR_PRESETLESS = {"Darkcar": "Plank"}


def _hundredths(v: float) -> float:
    """The float32 nearest to v rounded to two decimals."""
    from decimal import ROUND_HALF_EVEN, Decimal
    return F(float(Decimal(v).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)))


def _rounded_axle(a: "cars.Axle") -> "cars.Axle":
    return cars.Axle(a.radius, tuple(_hundredths(v) for v in a.rest))


def _h(w: int) -> str:
    return f"{w & 0xFFFFFFFF:08x}"


def _vec(q) -> tuple[float, float, float]:
    return struct.unpack("<3f", q.raw)


def _inherited(pkg: Package, e, name: str):
    """A property of an export or, failing that, of its archetype chain (placed actors inherit from prefabs)."""
    seen = set()
    while e is not None and e.index not in seen:
        seen.add(e.index)
        try:
            p = obj_props(pkg, e)
        except PackageError:
            p = {}
        if name in p:
            return p[name]
        e = pkg.exports[e.archetype - 1] if e.archetype > 0 else None
    return None


# ---------------------------------------------------------------- the map

def pads(stadium: Package) -> list[list[int]]:
    rows = []
    for e in stadium.exports:
        if stadium.class_of(e) != "VehiclePickup_Boost_TA" or not stadium.obj_path(e.index).startswith("TheWorld."):
            continue
        loc = _inherited(stadium, e, "Location")
        if loc is None:
            raise PackageError(f"pickup {e.name} has no location")
        arche = [stadium.obj_path(a) for a in _arch_chain(stadium, e)]
        big = any(a.endswith("VehiclePickup_BoostPill") for a in arche)
        small = any(a.endswith("VehiclePickup_BoostPad") for a in arche)
        if big == small:
            raise PackageError(f"pickup {e.name}: neither or both of pad and pill")
        x, y, z = struct.unpack("<3I", loc.raw)
        rows.append([x, y, z, 1 if big else 0])
    if len(rows) != NUM_PADS:
        raise PackageError(f"expected {NUM_PADS} boost pickups, found {len(rows)}")
    # The order catSim keeps pads in: by y, then by x.
    rows.sort(key=lambda r: (frombits(r[1]), frombits(r[0])))
    return rows


def _arch_chain(pkg: Package, e) -> list[int]:
    out, seen = [], set()
    a = e.archetype
    while a and a not in seen:
        seen.add(a)
        out.append(a)
        a = pkg.exports[a - 1].archetype if a > 0 else 0
    return out


def pad_heights(level: Package) -> tuple[int, int]:
    """(small, big) pad heights placed in another map (its level instances override the prefab's)."""
    zs = {False: set(), True: set()}
    for r in pads(level):
        zs[bool(r[3])].add(r[2])
    if len(zs[False]) != 1 or len(zs[True]) != 1:
        raise PackageError(f"{level.path.name}: pads at more than one height per size")
    return zs[False].pop(), zs[True].pop()


def goals(stadium: Package) -> list[list[int]]:
    rows = []
    for e in stadium.exports:
        if stadium.class_of(e) != "GoalVolume_TA" or not stadium.obj_path(e.index).startswith("TheWorld."):
            continue
        loc = _inherited(stadium, e, "Location")
        brush = obj_props(stadium, e)["Brush"].value
        model = stadium.exports[brush - 1]
        polys = next(x for x in stadium.exports if stadium.class_of(x) == "Polys" and x.outer == model.index)
        faces = _polys(stadium, polys)
        if len(faces) != 6:
            raise PackageError(f"goal volume {e.name}: {len(faces)} faces, expected 6")
        lx, ly, lz = struct.unpack("<3I", loc.raw)
        # The goal at negative y is the blue team's: orange (1) scores in it.
        team_scoring = 1 if frombits(ly) < 0 else 0
        words = [lx, ly, lz, team_scoring]
        for normal, vertex in faces:
            words += [*normal, *vertex]
        rows.append(words)
    if len(rows) != 2:
        raise PackageError(f"expected 2 goal volumes, found {len(rows)}")
    rows.sort(key=lambda r: frombits(r[1]))
    return rows


def _polys(pkg: Package, e) -> list[tuple]:
    """A brush's polygons: each one's normal and first vertex, as float bits. Layout after the (empty)
    property list: count, max, the owner, then per polygon base, normal, texture U and V (12 bytes each),
    the vertices (count + 12 each) and a fixed-size tail."""
    b = pkg.export_bytes(e)
    _, at = export_props(pkg, e)
    n, mx = struct.unpack_from("<ii", b, at)
    if not 0 < n <= mx < 256:
        raise PackageError(f"{e.name}: bad polygon count")
    o = at + 12
    sizes = []
    out = []
    starts = []
    for _ in range(n):
        starts.append(o)
        normal = struct.unpack_from("<3I", b, o + 12)
        nv = struct.unpack_from("<i", b, o + 48)[0]
        first = struct.unpack_from("<3I", b, o + 52)
        out.append((normal, first))
        body = 52 + 12 * nv
        sizes.append(body)
        o += body
        if len(out) == 1:
            tail = (len(b) - at - 12 - sum(52 + 12 * nv for _ in range(n))) // n
            if tail < 0:
                raise PackageError(f"{e.name}: polygon tail size")
        o += tail
    if o != len(b):
        raise PackageError(f"{e.name}: polygons don't fill the record (variable vertex counts?)")
    return out


def planes(stadium: Package, count: int = 6) -> list[list[int]]:
    """The arena planes: the placed components of the plane mesh, in the level's order (six in Stadium_P)."""
    plane_mesh = next(x for x in stadium.exports if stadium.class_of(x) == "StaticMesh" and x.name == "Plane32768")
    b = stadium.export_bytes(plane_mesh)
    _, at = export_props(stadium, plane_mesh)
    ext_x, ext_y = struct.unpack_from("<2f", b, at + 12)
    rows = []
    for e in stadium.exports:
        if stadium.class_of(e) != "StaticMeshComponent" or not stadium.obj_path(e.index).startswith("TheWorld."):
            continue
        m = _inherited(stadium, e, "StaticMesh")
        if m is None or m.value != plane_mesh.index:
            continue
        p = obj_props(stadium, e)
        t = _vec(p["Translation"]) if "Translation" in p else (0.0, 0.0, 0.0)
        r = struct.unpack("<3i", p["Rotation"].raw) if "Rotation" in p else (0, 0, 0)
        s3 = p.get("Scale3D") or _inherited(stadium, e, "Scale3D")
        s = _vec(s3) if s3 is not None else (1.0, 1.0, 1.0)
        half = [bits(mul(mul(ext_x, s[0]), C002)), bits(mul(mul(ext_y, s[1]), C002))]
        basis = [0 if w == 0x80000000 else w for w in basis_bits(scaled_basis(r[0], r[1], r[2], s))]
        origin = [bits(mul(v, C002)) for v in t]
        rows.append(half + basis + origin)
    if len(rows) != count:
        raise PackageError(f"expected {count} arena planes, found {len(rows)}")
    return rows


# ---------------------------------------------------------------- the cars

def all_tables(cooked: Path, keys: list[bytes], log=print) -> dict[str, list]:
    from .upk import open_package

    def op(name):
        return open_package(cooked / name, keys)

    stadium, tagame, startup, gameinfo = op("Stadium_P.upk"), op("TAGame.upk"), op("Startup.upk"), op("GameInfo_Soccar_SF.upk")
    defaults = cars.preset_defaults(tagame)
    presets = cars.presets_in(startup, defaults)
    by_name = {p.name: p for p in presets}
    body_pkgs: dict[str, Package] = {}
    body_rows, variants = [], []
    phys_boxes_by_body = {}
    bdef = obj_props(tagame, next(e for e in tagame.exports if e.name == "Default__ProductAsset_Body_TA"))
    for pid, name, pkgname, asset in cars.BODY_LIST:
        pkg = body_pkgs.get(pkgname) or op(pkgname)
        body_pkgs[pkgname] = pkg
        e = next((x for x in pkg.exports if x.name == asset and pkg.class_of(x) == "ProductAsset_Body_TA"), None)
        if e is None:
            raise PackageError(f"{pkgname}: no body {asset}")
        d = obj_props(pkg, e)
        preset_name = None
        if "HandlingPreset" in d:
            path = pkg.obj_path(d["HandlingPreset"].value)
            short = path.split(".")[-1]
            if short in by_name:
                preset_name = short
            else:
                own = [p for p in cars.presets_in(pkg, defaults) if pkg.obj_path(export_by_path(pkg, path).index) == path and p.name == short]
                if not own:
                    raise PackageError(f"{name}: preset {path} not found")
                own[0].name = name          # a preset shipped inside a body's package takes the body's name
                presets.append(own[0])
                by_name[name] = own[0]
                preset_name = name
        elif name in PRESET_FOR_PRESETLESS:
            base = by_name[PRESET_FOR_PRESETLESS[name]]
            if "PhysicsAsset" in d:
                box, t = cars._box(pkg, pkg.obj_path(d["PhysicsAsset"].value))
                if (box, t) != (base.box, base.t):
                    raise PackageError(f"{name}'s physics box is not {base.name}'s: the rule no longer holds")
                phys_boxes_by_body[name] = cars.Preset(name, box, t, base.front, base.back)
            p = cars.Preset(name, base.box, base.t, _rounded_axle(base.front), _rounded_axle(base.back))
            presets.append(p)
            by_name[name] = p
            preset_name = name
        else:
            raise PackageError(f"{name}: no handling preset and no rule for it")
        body_rows.append(_body_row(pkg, tagame, bdef, d, pid, name, by_name[preset_name], preset_name))
        mut = next((x for x in pkg.exports if pkg.class_of(x) == "Mutator_Robin_TA"), None)
        if mut is not None:
            variants.append(_variant_row(pkg, mut, by_name[preset_name]))
    curves = obj_props(gameinfo, export_by_path(gameinfo, "Archetypes.Car.Car_Default.DefaultVehicleSim"))
    steer = cars.curve_points(gameinfo, curves["SteerAngleCurve"].raw)
    handbrake = cars.curve_points(gameinfo, curves["HandbrakeSteerAngleCurve"].raw)
    t = {
        "presets": [[f"s:{p.name}"] + [_h(w) for w in p.row()] for p in presets],
        "steer_folded": [[_h(w) for w in cars.fold_curve(steer, p)] for p in presets],
        "handbrake_folded": [[_h(w) for w in cars.fold_curve(handbrake, p)] for p in presets],
        "bodies": body_rows,
        "vehicle_variants": variants,
        "pads": [[_h(w) for w in r] for r in pads(stadium)],
        "goals": [[_h(w) for w in r] for r in goals(stadium)],
        "planes": [[_h(w) for w in r] for r in planes(stadium)],
    }
    uf = op("UF_Night_P.upk")
    small, big = pad_heights(uf)
    t["pad_z_UF_Night_P"] = [[_h(small), _h(big)]]
    check_second_map(stadium, uf)
    boxes = []
    for nm in PICKUP_BOX_PRESETS:
        p = phys_boxes_by_body.get(nm) or by_name[nm]
        boxes.append([f"s:{nm}"] + [_h(w) for w in p.phys_box_row()])
    t["phys_boxes"] = boxes
    o = by_name["Octane"]
    t["default_phys_box"] = [[_h(bits(mul(o.box[0], 0.5))), _h(bits(mul(o.box[1], 0.5))), _h(bits(mul(o.box[2], 0.5))),
                              _h(bits(o.t[0])), _h(0), _h(bits(o.t[2]))]]
    octane_body = body_rows[0]
    t["default_mesh_bounds"] = [octane_body[6:13]]
    log(f"tables: {len(presets)} presets, {len(body_rows)} bodies, {len(variants)} vehicle variants, 34 pads, 2 goals, 6 planes")
    return t


def _body_row(pkg, tagame, bdef, d, pid, name, preset: cars.Preset, preset_name: str) -> list[str]:
    legacy = bool(d["bUseLegacySuspensionOffsets"].value) if "bUseLegacySuspensionOffsets" in d else False
    xo = d["HandlingPresetXOffset"].value if "HandlingPresetXOffset" in d else 0.0
    zo = d["HandlingPresetZOffset"].value if "HandlingPresetZOffset" in d else 0.0
    tmin = d["SuspensionTravelMin"].value if "SuspensionTravelMin" in d else 0.0
    sk = cars.skeleton(pkg, pkg.obj_path(d["Mesh"].value))
    has_preset = "HandlingPreset" in d
    zs = []
    for ax, rp in (("FrontAxle", preset.front), ("BackAxle", preset.back)):
        f = dict(struct_props(tagame, bdef[ax].raw))
        own = struct_props(pkg, d[ax].raw) if ax in d else {}
        f.update(own)
        rb = f["WheelRadius"].value if legacy else f["WheelMeshRadius"].value
        bone = struct_props(pkg if "RightWheel" in own else tagame, f["RightWheel"].raw)["BoneName"].value
        if bone not in sk.bones:
            zs = None
            break
        from .f32 import add, sub
        zs.append(add(mul(2.0, sub(rb, rp.radius)), sub(rp.rest[2], cars.compose_z(sk, bone))))
    from .f32 import add
    if not has_preset or zs is None:
        mz = 0.0                 # no preset, or wheel bones other than the usual four: the mesh isn't moved
    else:
        mz = max(zs)
        if not legacy:
            mz = add(mz, tmin)
        mz = add(mz, zo)
    fwd = add(xo, sk.origin[0])
    return [f"s:{name}", f"s:{preset_name}"] + [_h(w) for w in [pid, bits(mz), bits(fwd), bits(xo), *sk.bounds]]


def _variant_row(pkg: Package, mut, preset: cars.Preset) -> list[str]:
    """A body that brings its own car rules (a mutator in its package): its own steer curves, wheel friction and
    contact distance, random torques and flip time, and its sticky and auto-flip settings. Settings the
    mutator doesn't give are zero, as its own struct defaults."""
    m = obj_props(pkg, mut)
    sim = pkg.exports[m["VehicleSim"].value - 1]
    s = obj_props(pkg, sim)
    steer = cars.curve_points(pkg, s["SteerAngleCurve"].raw)
    hb = cars.curve_points(pkg, s["HandbrakeSteerAngleCurve"].raw)
    fs = cars.fold_curve(steer, preset)
    fh = cars.fold_curve(hb, preset)
    wheels = struct.unpack_from("<5i", s["Wheels"].raw)
    w0 = obj_props(pkg, pkg.exports[wheels[1] - 1])
    lat = cars.curve_points(pkg, w0["LatFrictionCurve"].raw)
    sticky = struct_props(pkg, m["StickyForce"].raw) if "StickyForce" in m else {}
    autoflip = struct_props(pkg, m["AutoFlip"].raw) if "AutoFlip" in m else {}

    def fval(dct, k):
        return struct.unpack("<f", dct[k].raw)[0] if k in dct else 0.0

    comps = {pkg.class_of(x): obj_props(pkg, x) for x in pkg.exports if x.outer == mut.index}

    def rand(cls):
        c = comps[cls]
        out = []
        for k in ("RandomPitch", "RandomYaw", "RandomRoll"):
            r = struct_props(pkg, c[k].raw)
            out += [bits(fval(r, "Frequency")), bits(fval(r, "Amplitude"))]
        return out

    contact = w0["ContactForceDistance"].value if "ContactForceDistance" in w0 else 0.0
    flip = comps["CarComponent_FlipCar_TA"]["FlipCarTime"].value
    words = [len(steer)] + [w for (i, _o), f in zip(steer, fs) for w in (i, f)]
    words += [len(hb)] + [w for (i, _o), f in zip(hb, fh) for w in (i, f)]
    if len(lat) != 2:
        raise PackageError("variant lateral friction: expected 2 points")
    words += [lat[0][0], lat[0][1], lat[1][0], lat[1][1]]
    words += [bits(fval(sticky, "Ground")), bits(fval(autoflip, "Torque")), bits(fval(autoflip, "Force")),
              bits(mul(F(contact), C002))]
    words += rand("CarComponent_AirControl_Robin_TA") + rand("CarComponent_Jump_Robin_TA") + rand("CarComponent_DoubleJump_Robin_TA")
    words += [bits(F(flip))]
    return [f"s:{preset.name}"] + [_h(w) for w in words]


def check_second_map(stadium: Package, level: Package) -> None:
    """The engine runs UF_Night_P as Stadium_P without the first two planes, with its own pad heights. Check that
    against the level: the same 16 collision meshes in the same places, and its planes Stadium_P's last four."""
    from .arena import soccar_pieces
    a, b = soccar_pieces(stadium), soccar_pieces(level)
    if len(a) != len(b) or any((x.verts, x.tris, x.scale, x.translation) != (y.verts, y.tris, y.scale, y.translation)
                               for x, y in zip(a, b)):
        raise PackageError(f"{level.path.name}: its collision meshes are not Stadium_P's")
    if planes(level, count=4) != planes(stadium)[2:]:
        raise PackageError(f"{level.path.name}: its planes are not Stadium_P's last four")


def write_tables(out: Path, t: dict) -> Path:
    lines = ["catsim-tables 1", "# made by catsim.dump from your own copy of the game; do not share"]
    for name, rows in t.items():
        lines.append(f"[{name}]")
        lines += [" ".join(r) for r in rows]
    p = out / "tables.txt"
    p.write_text("\n".join(lines) + "\n", encoding="ascii")
    return p
