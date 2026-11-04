"""What each car item and each map adds to the random-draw model, read from the game's packages.

The game draws random numbers outside physics too. Every skeletal (animated) mesh component in the world runs a
level-of-detail timer that draws one number from the game's random stream every 0.75 to 1.25 s, so the cars'
items and the map decide how many draws a frame makes. catSim models those draws (catsim.draw) so a replay of a
real game keeps the game's random stream, which decides kickoff spawn choices.

This module writes `items.txt` into the data folder:

    [items]   product id, slot, package, skeletal components, status
    [maps]    map package, skeletal components placed in the level (and its always-loaded sublevels)

`status` is `ok` when the item's parts are all ones the model knows (static meshes, skeletal meshes, particle
effects that don't draw), or `unmodeled:<why>` when the item carries something the model has no rule for; a match
with such an item isn't covered and the model stays off for it. How each slot is read (our own description):

- Body: the body asset's mesh is one skeletal component (the car).
- Wheels: the wheel asset's generator lists the four wheels' settings; each wheel whose setting names a skeletal
  mesh is one skeletal component. Wheels carrying attachments of their own are unmodeled.
- Antenna and topper: the attachment asset's list; each entry with a skeletal mesh is one skeletal component, and so
  is an antenna's stick (its antenna component's own skeletal mesh). Entries with a particle system, an effects
  actor or a component template are unmodeled.
- Goal explosion: the skeletal components of its effects actor (0 or 1 are modeled).
- Decal, boost, trail, paint finish, engine sound: no skeletal mesh anywhere in the package (else unmodeled).
"""
from __future__ import annotations

import struct
from pathlib import Path

from .props import export_by_path, obj_props, struct_array, struct_props
from .upk import Package, PackageError, export_props, open_package

# The product slots the model reads, by the slot object's name in the product database.
SLOTS = {
    "Body": "body", "Wheels": "wheels", "Antenna": "antenna", "Hat": "topper", "GoalExplosion": "goal_explosion",
    "Skin": "decal", "Boost": "boost", "SupersonicTrail": "trail", "PaintFinish": "paint_finish",
    "EngineAudio": "engine_audio",
}


def products(tagame: Package) -> list[tuple[int, str, str, str]]:
    """(product id, slot, package name, asset path) for every product in a slot the model reads. The id is the
    index in the game's product database (the number RLBot loadouts use)."""
    db = export_by_path(tagame, "ProductsDB.Database.ProductsDB")
    if db is None:
        raise PackageError("TAGame.upk has no product database")
    q = next((q for q in export_props(tagame, db)[0] if q.name == "Products_New"), None)
    if q is None:
        raise PackageError("the product database has no product list")
    n = struct.unpack_from("<i", q.raw, 0)[0]
    if len(q.raw) != 4 + 4 * n:
        raise PackageError("the product list is not a list of references")
    out = []
    for pid, ref in enumerate(struct.unpack_from(f"<{n}i", q.raw, 4)):
        if ref <= 0:
            continue
        p = obj_props(tagame, tagame.exports[ref - 1])
        s = p.get("Slot")
        slot = tagame.obj_path(s.value).split(".")[-1] if s is not None and s.value else None
        if slot not in SLOTS or "AssetPackageName" not in p:
            continue
        path = p["AssetPath"].value if "AssetPath" in p else ""
        out.append((pid, SLOTS[slot], p["AssetPackageName"].value, path))
    return out


class _Ctx:
    """The item's package, plus the always-loaded package (Startup.upk) where references out of it can lead:
    the game's default items keep their parts there."""

    def __init__(self, pkg: Package, startup: Package | None):
        self.pkg, self.startup = pkg, startup

    def deref(self, pkg: Package, ref: int):
        """(package, export) for a reference, following an import into Startup.upk by its path."""
        if ref > 0:
            return pkg, pkg.exports[ref - 1]
        if ref < 0 and self.startup is not None:
            e = export_by_path(self.startup, pkg.obj_path(ref))
            if e is not None:
                return self.startup, e
        raise PackageError(f"reference {pkg.obj_path(ref)} not found")

    def cls(self, pkg: Package, ref: int) -> str:
        if ref > 0:
            return pkg.class_of(pkg.exports[ref - 1])
        if ref < 0:
            return pkg.imports[-ref - 1].class_name
        return ""


def _in_thumbnail(pkg: Package, e) -> bool:
    return pkg.obj_path(e.index).startswith("ThumbnailRenderers.")


def _asset(pkg: Package, path: str, cls_prefix: str):
    e = export_by_path(pkg, path) if path else None
    if e is None:
        e = next((x for x in pkg.exports if pkg.class_of(x).startswith(cls_prefix) and not x.name.startswith("Default__")
                  and not _in_thumbnail(pkg, x)), None)
    if e is None or pkg.class_of(e) != cls_prefix:
        raise PackageError(f"no {cls_prefix} asset {path!r}")
    return e


def _count(raw: bytes) -> int:
    return struct.unpack_from("<i", raw, 0)[0]


def _no_skeletal(c: _Ctx, path: str) -> tuple[int, str]:
    pkg = c.pkg
    for e in pkg.exports:
        if _in_thumbnail(pkg, e):
            continue
        k = pkg.class_of(e)
        if k == "SkeletalMesh" or "SkeletalMeshComponent" in k:
            return 0, "unmodeled:carries a skeletal mesh"
    return 0, "ok"


def _attachments(c: _Ctx, pkg: Package, raw: bytes) -> tuple[int, str]:
    """An attachment list: skeletal entries and antenna sticks counted; unknown parts reported."""
    count = 0
    for it in struct_array(pkg, raw):
        for k in ("ParticleSystem", "FXActor", "Component"):
            if k in it and it[k].value:
                return 0, f"unmodeled:attachment {k}"
        if "SkeletalMesh" in it and it["SkeletalMesh"].value:
            count += 1
        a = it.get("Antenna")
        if a is not None and a.value:
            apkg, ae = c.deref(pkg, a.value)
            ap = obj_props(apkg, ae)
            if "AntennaMesh" in ap and ap["AntennaMesh"].value:      # the antenna's stick: a skeletal mesh of its own
                if c.cls(apkg, ap["AntennaMesh"].value) != "SkeletalMesh":
                    return 0, "unmodeled:antenna stick not skeletal"
                count += 1
    return count, "ok"


def _body(c: _Ctx, path: str) -> tuple[int, str]:
    pkg = c.pkg
    d = obj_props(pkg, _asset(pkg, path, "ProductAsset_Body_TA"))
    if "Mesh" not in d or c.cls(pkg, d["Mesh"].value) != "SkeletalMesh":
        return 0, "unmodeled:no skeletal car mesh"
    if "Attachments" in d and _count(d["Attachments"].raw) > 0:
        n, status = _attachments(c, pkg, d["Attachments"].raw)
        if status != "ok":
            return 1, status
        if n:
            return 1, "unmodeled:skeletal body attachments"
    return 1, "ok"


def _wheels(c: _Ctx, path: str) -> tuple[int, str]:
    pkg = c.pkg
    d = obj_props(pkg, _asset(pkg, path, "ProductAsset_Wheel_TA"))
    g = d.get("Generator")
    if g is None or g.value == 0:
        return 0, "unmodeled:no wheel generator"
    gpkg, ge = c.deref(pkg, g.value)
    gp = obj_props(gpkg, ge)
    if "Wheels" not in gp:
        return 0, "unmodeled:generator lists no wheels"
    raw = gp["Wheels"].raw
    n = _count(raw)
    if n != 4:
        return 0, f"unmodeled:{n} wheels"
    count = 0
    for r in struct.unpack_from(f"<{n}i", raw, 4):
        wpkg, we = c.deref(gpkg, r)
        w = obj_props(wpkg, we)
        if "Attachments" in w and _count(w["Attachments"].raw) > 0:
            return 0, "unmodeled:wheel attachments"
        if "SkeletalMesh" in w and w["SkeletalMesh"].value:
            count += 1
        elif not ("StaticMesh" in w and w["StaticMesh"].value):
            return 0, "unmodeled:wheel without a mesh"
    return count, "ok"


def _attachment(c: _Ctx, path: str) -> tuple[int, str]:
    pkg = c.pkg
    d = obj_props(pkg, _asset(pkg, path, "ProductAsset_Attachment_TA"))
    if "Attachments" not in d:
        return 0, "ok"
    return _attachments(c, pkg, d["Attachments"].raw)


def _goal_explosion(c: _Ctx, path: str) -> tuple[int, str]:
    pkg = c.pkg
    d = obj_props(pkg, _asset(pkg, path, "ProductAsset_GoalExplosion_TA"))
    fx = d.get("GoalExplosionFX")
    if fx is None or fx.value == 0:
        return 0, "ok"
    fpkg, fe = c.deref(pkg, fx.value)
    root = fe.index
    n = 0
    for e in fpkg.exports:
        if "SkeletalMeshComponent" not in fpkg.class_of(e):
            continue
        o = e.outer
        while o > 0 and o != root:
            o = fpkg.exports[o - 1].outer
        if o == root:
            n += 1
    if n > 1:
        return n, "unmodeled:several skeletal components"
    return n, "ok"


READERS = {"body": _body, "wheels": _wheels, "antenna": _attachment, "topper": _attachment,
           "goal_explosion": _goal_explosion}


def classify(pkg: Package, slot: str, path: str, startup: Package | None = None) -> tuple[int, str]:
    """(skeletal components, status) of one product. `startup`: Startup.upk, for parts the item takes from it."""
    try:
        return READERS.get(slot, _no_skeletal)(_Ctx(pkg, startup), path)
    except (PackageError, struct.error, IndexError, KeyError) as e:
        return 0, f"unmodeled:unreadable ({type(e).__name__})"


def map_components(cooked: Path, keys: list[bytes], level: str) -> int:
    """Skeletal mesh components placed in a map's level and in the sublevels it always loads."""
    seen, todo, n = set(), [level], 0
    while todo:
        name = todo.pop()
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        f = cooked / (name if name.lower().endswith(".upk") else f"{name}.upk")
        if not f.is_file():
            raise PackageError(f"{level}: sublevel {name} not found")
        pkg = open_package(f, keys)
        for e in pkg.exports:
            c = pkg.class_of(e)
            if not pkg.obj_path(e.index).startswith("TheWorld."):
                continue
            if "SkeletalMeshComponent" in c:
                n += 1
            elif c == "LevelStreamingAlwaysLoaded":
                p = obj_props(pkg, e)
                if "PackageName" in p:
                    todo.append(str(p["PackageName"].value))
    return n


MAPS = ("Stadium_P", "UF_Night_P")


def dump_items(cooked: Path, keys: list[bytes], out: Path, log=print) -> Path:
    tagame = open_package(cooked / "TAGame.upk", keys)
    startup = open_package(cooked / "Startup.upk", keys)
    rows, missing, opened = [], 0, {}
    prods = products(tagame)
    for k, (pid, slot, pkgname, path) in enumerate(prods):
        f = cooked / f"{pkgname}_SF.upk"
        if not f.is_file():
            missing += 1
            rows.append((pid, slot, pkgname, 0, "missing"))
            continue
        try:
            pkg = opened.get(pkgname) or open_package(f, keys)
        except (PackageError, OSError) as e:
            why = "no key in your key file opens it" if "decrypts" in str(e) else f"package unreadable ({type(e).__name__})"
            rows.append((pid, slot, pkgname, 0, f"unmodeled:{why}"))
            continue
        opened[pkgname] = pkg
        if len(opened) > 8:
            opened.pop(next(iter(opened)))
        n, status = classify(pkg, slot, path, startup)
        rows.append((pid, slot, pkgname, n, status))
        if log and (k + 1) % 1000 == 0:
            log(f"items: {k + 1} of {len(prods)}")
    lines = ["catsim-items 1", "# made by catsim.dump from your own copy of the game; do not share",
             "# id slot package skeletal_components status", "[items]"]
    lines += [f"{pid} {slot} {pkg} {n} {status.replace(' ', '_')}" for pid, slot, pkg, n, status in rows]
    lines.append("[maps]")
    for m in MAPS:
        lines.append(f"{m} {map_components(cooked, keys, m)}")
    p = out / "items.txt"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok = sum(1 for r in rows if r[4] == "ok")
    if log:
        log(f"items: {len(rows)} products, {ok} modeled, {len(rows) - ok - missing} unmodeled, {missing} not installed")
    return p
