"""The soccar arena's collision geometry, read from the map package (Stadium_P.upk).

What the game collides against (our own description, checked bit for bit, see docs/PROVENANCE.md):

- The map's persistent level carries cooked per-triangle collision data for its static meshes:
  a list of (static mesh, Scale3D) -> index, then one cooked triangle mesh per index, in index
  order. Each cooked mesh is a block starting with the bytes "NXS\\x01MESH": version, flags,
  three more header words, the vertex count and the triangle count, then the vertices (three
  float32 each, already in physics units with the mirror applied) and the triangles (three
  indices each; 8-bit if flags has 0x8, 16-bit if it has 0x10, else 32-bit).
- The arena's triangle meshes are the entries whose mesh is one of the arena collision meshes
  (the `*_Collision*` meshes), in index order. That order is the order the physics world holds
  them in, and it matters: it decides contact order.
- Where each piece sits comes from its component in the level (Translation, uu), and the sign
  of its mirror from its Scale3D. Both go in arena.txt; the engine turns them into transforms.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from .upk import Package, PackageError, export_props

MAGIC = b"NXS\x01MESH"


@dataclass
class Piece:
    mesh_name: str
    scale: tuple[float, float, float]
    verts: list[tuple[int, int, int]]      # float32 bits
    tris: list[tuple[int, int, int]]
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _cooked(buf: bytes, at: int) -> tuple[list, list, int]:
    if buf[at:at + 8] != MAGIC:
        raise PackageError("cooked mesh block expected")
    _ver, flags, _thr, _a, _b, nv, nt = struct.unpack_from("<iIIiiii", buf, at + 8)
    if not (0 < nv < 1_000_000 and 0 < nt < 2_000_000):
        raise PackageError(f"implausible cooked mesh: {nv} vertices, {nt} triangles")
    o = at + 36
    verts = [struct.unpack_from("<III", buf, o + 12 * i) for i in range(nv)]
    o += 12 * nv
    w, fmt = (1, "<BBB") if flags & 0x8 else ((2, "<HHH") if flags & 0x10 else (4, "<III"))
    tris = [struct.unpack_from(fmt, buf, o + 3 * w * i) for i in range(nt)]
    if max(max(t) for t in tris) >= nv:
        raise PackageError("cooked mesh triangle index out of range")
    return verts, tris, o + 3 * w * nt


def _cache_map(pkg: Package, level: bytes) -> tuple[int, list[tuple[str, tuple]]]:
    """Find the (mesh, Scale3D, index) list for the per-triangle cache: the run of 20-byte rows
    whose indices count 0, 1, 2, ... and whose meshes are StaticMesh exports."""
    meshes = {e.index: e.name for e in pkg.exports if pkg.class_of(e) == "StaticMesh"}
    best = (0, -1, [])
    starts = set()
    for ref in meshes:                      # a list starts with a count, then a row whose index is 0
        pat = struct.pack("<i", ref)
        o = level.find(pat, 4)
        while o >= 0:
            if o + 20 <= len(level) and struct.unpack_from("<i", level, o + 16)[0] == 0:
                starts.add(o - 4)
            o = level.find(pat, o + 1)
    for o in sorted(starts):
        n = struct.unpack_from("<i", level, o)[0]
        if not (1 <= n <= 4096 and o + 4 + 20 * n <= len(level)):
            continue
        rows = []
        for k in range(n):
            ref, sx, sy, sz, idx = struct.unpack_from("<i3fi", level, o + 4 + 20 * k)
            if ref not in meshes or idx != k:
                break
            rows.append((meshes[ref], (sx, sy, sz)))
        if len(rows) == n and n > best[0] and any("_Collision" in r[0] for r in rows):
            best = (n, o, rows)
    if best[1] < 0:
        raise PackageError("the level's collision cache list was not found")
    return best[1] + 4 + 20 * best[0], best[2]


def _translations(pkg: Package) -> dict[tuple[str, tuple], list[tuple[float, float, float]]]:
    """Each placed collision component's Translation (uu), keyed by (mesh, Scale3D)."""
    meshes = {e.index: e.name for e in pkg.exports if pkg.class_of(e) == "StaticMesh"}
    comp_mesh: dict[int, str] = {}
    comp_props: dict[int, dict] = {}
    for e in pkg.exports:
        if pkg.class_of(e) != "StaticMeshComponent":
            continue
        try:
            props, _ = export_props(pkg, e)
        except PackageError:
            continue
        d = {q.name: q for q in props}
        comp_props[e.index] = d
        m = d.get("StaticMesh")
        if m is not None and m.value in meshes:
            comp_mesh[e.index] = meshes[m.value]

    def mesh_of(i: int) -> str | None:
        seen = set()
        while i and i not in seen:
            seen.add(i)
            if i in comp_mesh:
                return comp_mesh[i]
            i = pkg.exports[i - 1].archetype if i > 0 else 0
        return None

    def vec(d: dict, name: str, default):
        q = d.get(name)
        return struct.unpack("<3f", q.raw) if q is not None and q.struct == "Vector" else default

    out: dict = {}
    for e in pkg.exports:
        if pkg.class_of(e) != "StaticMeshComponent" or not pkg.obj_path(e.index).startswith("TheWorld."):
            continue
        name = mesh_of(e.index)
        if not name or "_Collision" not in name:
            continue
        d = comp_props.get(e.index, {})
        # A collection actor's component carries its whole transform; anything else would need its actor's.
        if pkg.class_of(pkg.exports[e.outer - 1]) != "StaticMeshCollectionActor":
            raise PackageError(f"collision component {pkg.obj_path(e.index)} is not in a mesh collection: unsupported layout")
        if d.get("Rotation") is not None or d.get("Scale") is not None:
            raise PackageError(f"collision component {pkg.obj_path(e.index)} is rotated or scaled: unsupported layout")
        key = (name, vec(d, "Scale3D", (1.0, 1.0, 1.0)))
        out.setdefault(key, []).append(vec(d, "Translation", (0.0, 0.0, 0.0)))
    return out


def soccar_pieces(pkg: Package) -> list[Piece]:
    levels = [e for e in pkg.exports if pkg.class_of(e) == "Level"]
    if len(levels) != 1:
        raise PackageError(f"expected one level in {pkg.path.name}, found {len(levels)}")
    level = pkg.export_bytes(levels[0])
    at, rows = _cache_map(pkg, level)
    places = _translations(pkg)
    pieces = []
    for k, (name, scale) in enumerate(rows):
        at = level.find(MAGIC, at)
        if at < 0:
            raise PackageError(f"cooked mesh {k} of {len(rows)} is missing")
        verts, tris, at = _cooked(level, at)
        if "_Collision" not in name:
            continue
        spots = places.get((name, scale), [])
        if len(spots) != 1:
            raise PackageError(f"{name} {scale}: expected one placed component, found {len(spots)}")
        pieces.append(Piece(name, scale, verts, tris, spots[0]))
    return pieces


def write_cmf(path: Path, piece: Piece) -> None:
    """The mesh file the engine reads: i32 triangle count, i32 vertex count, the triangles as i32
    triples, the vertices as float32 triples (physics units, local to the piece)."""
    out = bytearray(struct.pack("<ii", len(piece.tris), len(piece.verts)))
    for t in piece.tris:
        out += struct.pack("<iii", *t)
    for v in piece.verts:
        out += struct.pack("<III", *v)
    path.write_bytes(bytes(out))


def write_arena(out_dir: Path, pieces: list[Piece]) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    lines = ["# catsim arena: one line per mesh file, in the physics world's order",
             "# index mesh_name scale_x scale_y scale_z translation_x translation_y translation_z (floats as bits)"]
    for i, pc in enumerate(pieces):
        f = out_dir / f"mesh_{i}.cmf"
        write_cmf(f, pc)
        files.append(f)
        bits = [struct.unpack("<I", struct.pack("<f", x))[0] for x in (*pc.scale, *pc.translation)]
        lines.append(f"{i} {pc.mesh_name} " + " ".join(f"{b:08x}" for b in bits))
    t = out_dir / "arena.txt"
    t.write_text("\n".join(lines) + "\n", encoding="ascii")
    files.append(t)
    return files
