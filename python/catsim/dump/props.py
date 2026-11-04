"""Reading UE3 tagged properties into plain Python values: nested structs, arrays of structs, vectors."""
from __future__ import annotations

import struct

from .upk import Package, PackageError, Prop, export_props, read_props


def f32bits(raw: bytes, at: int = 0) -> int:
    return struct.unpack_from("<I", raw, at)[0]


def struct_props(pkg: Package, raw: bytes) -> dict[str, Prop]:
    """A tagged struct's fields by name (array elements as name[i])."""
    out = {}
    for q in read_props(pkg, raw, 0)[0]:
        out[q.name if q.index == 0 else f"{q.name}[{q.index}]"] = q
    return out


def struct_array(pkg: Package, raw: bytes) -> list[dict[str, Prop]]:
    """An ArrayProperty of tagged structs: count, then each element's fields up to None."""
    n = struct.unpack_from("<i", raw, 0)[0]
    at, items = 4, []
    for _ in range(n):
        ps, at = read_props(pkg, raw, at)
        items.append({q.name: q for q in ps})
    return items


def vector_bits(q: Prop | None) -> tuple[int, int, int] | None:
    if q is None:
        return None
    if q.struct != "Vector" or len(q.raw) != 12:
        raise PackageError(f"{q.name} is not a vector")
    return struct.unpack("<3I", q.raw)


def obj_props(pkg: Package, e) -> dict[str, Prop]:
    return {q.name if q.index == 0 else f"{q.name}[{q.index}]": q for q in export_props(pkg, e)[0]}


def export_by_path(pkg: Package, path: str):
    for e in pkg.exports:
        if pkg.obj_path(e.index) == path:
            return e
    return None


def ref_path(pkg: Package, ref: int) -> str | None:
    return pkg.obj_path(ref) if ref else None
