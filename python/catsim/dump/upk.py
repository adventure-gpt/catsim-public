"""A reader for the game's cooked Unreal Engine 3 packages (.upk), read-only and in memory.

The game's packages keep their name, import and export tables AES-encrypted and their
object data zlib-compressed in chunks. This module decrypts the tables with a key you
supply (see docs/DUMPER.md: no key ships with catSim), inflates the chunks, and gives
the tables, each export's bytes and UE3 tagged properties. It never writes the game's
files.

Format notes (public UE3 package layout plus the game's own variations, our own words):
- Summary: tag, file/licensee version, header size, folder, flags, then the counts and
  offsets of the name, export and import tables, depends, GUIDs, generations, engine and
  cooker versions, compression flags and an (empty) chunk list, then three small extra
  fields and the game's compression metadata: garbage size, chunk table offset (relative
  to the encrypted area) and last block size.
- The encrypted area starts at the name table and runs to the end of the header, less
  the garbage, rounded up to 16 bytes; AES-256 in ECB mode.
- The chunk table inside it has 24- or 36-byte rows starting with an int64 uncompressed
  offset, int32 size, int64 compressed offset, int32 size. The first row starts at the
  depends offset, which is how a key (and a row size) is checked.
- Each chunk: a 16-byte header (tag, block size, compressed and uncompressed totals),
  the block sizes, then zlib blocks.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from .aes import decrypt_ecb

TAG = 0x9E2A83C1


class PackageError(Exception):
    pass


class _R:
    def __init__(self, buf: bytes, at: int = 0):
        self.b, self.o = buf, at

    def i32(self) -> int:
        v = struct.unpack_from("<i", self.b, self.o)[0]
        self.o += 4
        return v

    def u32(self) -> int:
        v = struct.unpack_from("<I", self.b, self.o)[0]
        self.o += 4
        return v

    def u16(self) -> int:
        v = struct.unpack_from("<H", self.b, self.o)[0]
        self.o += 2
        return v

    def i64(self) -> int:
        v = struct.unpack_from("<q", self.b, self.o)[0]
        self.o += 8
        return v

    def u64(self) -> int:
        v = struct.unpack_from("<Q", self.b, self.o)[0]
        self.o += 8
        return v

    def fstring(self) -> str:
        n = self.i32()
        if n == 0:
            return ""
        if n > 0:
            s = self.b[self.o:self.o + n]
            self.o += n
            return s.rstrip(b"\0").decode("latin-1")
        s = self.b[self.o:self.o - 2 * n]
        self.o += -2 * n
        return s.decode("utf-16-le").rstrip("\0")


@dataclass
class Summary:
    file_version: int
    licensee_version: int
    header_size: int
    folder: str
    flags: int
    name_count: int
    name_offset: int
    export_count: int
    export_offset: int
    import_count: int
    import_offset: int
    depends_offset: int
    engine_version: int
    cooker_version: int
    compression_flags: int
    garbage_size: int = 0
    chunks_offset: int = 0
    last_block_size: int = 0


@dataclass
class Import:
    class_package: str
    class_name: str
    outer: int
    name: str


@dataclass
class Export:
    index: int            # 1-based object reference
    class_ref: int
    super_ref: int
    outer: int
    name: str
    archetype: int
    flags: int
    size: int
    offset: int


@dataclass
class Package:
    path: Path
    summary: Summary
    data: bytes
    names: list[str] = field(default_factory=list)
    imports: list[Import] = field(default_factory=list)
    exports: list[Export] = field(default_factory=list)

    # ---- object references: >0 export (1-based), <0 import (-1-based), 0 none
    def obj_name(self, ref: int) -> str:
        if ref > 0:
            return self.exports[ref - 1].name
        if ref < 0:
            return self.imports[-ref - 1].name
        return "None"

    def obj_path(self, ref: int) -> str:
        parts = []
        while ref:
            parts.append(self.obj_name(ref))
            ref = self.exports[ref - 1].outer if ref > 0 else self.imports[-ref - 1].outer
        return ".".join(reversed(parts))

    def class_of(self, e: Export) -> str:
        return self.obj_name(e.class_ref) if e.class_ref else "Class"

    def export_bytes(self, e: Export) -> bytes:
        return self.data[e.offset:e.offset + e.size]

    def find(self, cls: str | None = None, name: str | None = None) -> list[Export]:
        return [e for e in self.exports if (cls is None or self.class_of(e) == cls) and (name is None or e.name == name)]

    def fname(self, idx: int, num: int) -> str:
        s = self.names[idx]
        return s if num == 0 else f"{s}_{num - 1}"


def _summary(b: bytes) -> tuple[Summary, int]:
    r = _R(b)
    if r.u32() != TAG:
        raise PackageError("not an Unreal package (bad tag)")
    fv, lv = r.u16(), r.u16()
    hs = r.i32()
    folder = r.fstring()
    flags = r.u32()
    nc, no, ec, eo, ic, io, do = (r.i32() for _ in range(7))
    r.i32(); r.i32(); r.i32(); r.i32()          # import/export guids offset and counts, thumbnail table
    r.o += 16                                   # package guid
    gens = r.i32()
    r.o += 12 * gens                            # generations: export, name, net object counts (gens read first)
    ev, cv = r.u32(), r.u32()
    cf = r.u32()
    n = r.i32()
    r.o += 16 * n                               # the summary's own chunk list (empty in cooked packages)
    r.i32()                                     # package source
    for _ in range(r.i32()):                    # additional packages to cook
        r.fstring()
    for _ in range(r.i32()):                    # texture allocations
        r.o += 20
        k = r.i32()
        r.o += 4 * k
    s = Summary(fv, lv, hs, folder, flags, nc, no, ec, eo, ic, io, do, ev, cv, cf)
    s.garbage_size, s.chunks_offset, s.last_block_size = r.i32(), r.i32(), r.i32()
    return s, r.o


def _chunk_rows(dec: bytes, at: int, depends: int) -> list[tuple[int, int, int, int]]:
    n = struct.unpack_from("<i", dec, at)[0]
    for stride in (36, 24):
        if n < 1 or at + 4 + n * stride > len(dec):
            continue
        rows = [struct.unpack_from("<qiqi", dec, at + 4 + i * stride) for i in range(n)]
        if rows[0][0] == depends:
            return rows
    raise PackageError("chunk table not found where the key check expects it")


def key_fits(s: Summary, enc: bytes, key: bytes) -> bool:
    lo = s.chunks_offset - s.chunks_offset % 16
    probe = enc[lo:lo + 32]
    if len(probe) < 32:
        return False
    v = decrypt_ecb(key, probe)[s.chunks_offset % 16:]
    n, first = struct.unpack_from("<iq", v, 0) if len(v) >= 12 else (0, 0)
    return n >= 1 and first == s.depends_offset


def open_package(path: str | Path, keys: list[bytes]) -> Package:
    """Decrypt and inflate a package in memory. `keys`: candidate AES-256 keys; the one whose
    decryption puts the chunk table where the summary says is used."""
    path = Path(path)
    raw = path.read_bytes()
    s, _ = _summary(raw)
    if s.compression_flags & 1 == 0:
        raise PackageError(f"{path.name}: not zlib-compressed (flags {s.compression_flags:#x}); unsupported layout")
    size = (s.header_size - s.garbage_size - s.name_offset + 15) & ~15
    enc = raw[s.name_offset:s.name_offset + size]
    key = next((k for k in keys if key_fits(s, enc, k)), None)
    if key is None:
        raise PackageError(f"{path.name}: none of the {len(keys)} supplied keys decrypts this package")
    dec = decrypt_ecb(key, enc)
    rows = _chunk_rows(dec, s.chunks_offset, s.depends_offset)
    end = rows[-1][0] + rows[-1][1]
    buf = bytearray(end)
    buf[:s.name_offset] = raw[:s.name_offset]
    buf[s.name_offset:s.name_offset + len(dec)] = dec[:max(0, min(len(dec), end - s.name_offset))]
    for uoff, usize, coff, _csize in rows:
        _, _block, _ctot, utot = struct.unpack_from("<iiii", raw, coff)
        at, got, sizes = coff + 16, 0, []
        while got < utot:
            c, u = struct.unpack_from("<ii", raw, at)
            sizes.append((c, u))
            got += u
            at += 8
        out = uoff
        for c, u in sizes:
            block = zlib.decompress(raw[at:at + c])
            if len(block) != u:
                raise PackageError(f"{path.name}: a block inflated to {len(block)} bytes, not {u}")
            buf[out:out + u] = block
            out += u
            at += c
        if out - uoff != usize:
            raise PackageError(f"{path.name}: a chunk inflated to {out - uoff} bytes, not {usize}")
    pkg = Package(path, s, bytes(buf))
    _tables(pkg)
    return pkg


def _tables(p: Package) -> None:
    s, b = p.summary, p.data
    r = _R(b, s.name_offset)
    for _ in range(s.name_count):
        p.names.append(r.fstring())
        r.u64()
    r.o = s.import_offset
    for _ in range(s.import_count):
        cp, _n1 = r.i32(), r.i32()
        cn, _n2 = r.i32(), r.i32()
        outer = r.i32()
        nm, num = r.i32(), r.i32()
        p.imports.append(Import(p.names[cp], p.names[cn], outer, p.fname(nm, num)))
    r.o = s.export_offset
    for i in range(s.export_count):
        cls, sup, outer = r.i32(), r.i32(), r.i32()
        nm, num = r.i32(), r.i32()
        arch = r.i32()
        flags = r.u64()
        size = r.i32()
        off = r.i64()
        r.i32()                                  # export flags
        k = r.i32()                              # net objects
        r.o += 4 * k
        r.o += 16 + 4                            # package guid, package flags
        p.exports.append(Export(i + 1, cls, sup, outer, p.fname(nm, num), arch, flags, size, off))


# ---- UE3 tagged properties

@dataclass
class Prop:
    name: str
    type: str
    index: int
    size: int
    value: object
    raw: bytes
    struct: str | None = None


def read_props(p: Package, buf: bytes, at: int) -> tuple[list[Prop], int]:
    """Tagged properties from `at` to the `None` terminator -> (props, offset after it).
    Values: ints, floats, bools, names, object refs (int), struct raw bytes."""
    out = []
    while at + 8 <= len(buf):
        ni, nn = struct.unpack_from("<ii", buf, at)
        if not 0 <= ni < len(p.names):
            raise PackageError(f"bad property name index {ni} at {at}")
        name = p.fname(ni, nn)
        at += 8
        if name == "None":
            return out, at
        ti = struct.unpack_from("<i", buf, at)[0]
        ty = p.names[ti]
        at += 8
        size, idx = struct.unpack_from("<ii", buf, at)
        at += 8
        sname = None
        if ty == "StructProperty":
            si, sn = struct.unpack_from("<ii", buf, at)
            sname = p.fname(si, sn)
            at += 8
        elif ty == "ByteProperty":
            at += 8                               # the enum's name
        if ty == "BoolProperty":
            out.append(Prop(name, ty, idx, size, bool(buf[at]), buf[at:at + 1]))
            at += 1
            continue
        raw = buf[at:at + size]
        if ty == "IntProperty":
            val: object = struct.unpack_from("<i", raw)[0]
        elif ty == "FloatProperty":
            val = struct.unpack_from("<f", raw)[0]
        elif ty in ("ObjectProperty", "ClassProperty", "ComponentProperty"):
            val = struct.unpack_from("<i", raw)[0]
        elif ty == "NameProperty":
            val = p.fname(*struct.unpack_from("<ii", raw))
        elif ty == "ByteProperty":
            val = raw[0] if size == 1 else p.fname(*struct.unpack_from("<ii", raw))
        elif ty == "StrProperty":
            val = _R(raw).fstring()
        else:
            val = raw
        out.append(Prop(name, ty, idx, size, val, raw, sname))
        at += size
    raise PackageError("property list ran off the end of its buffer")


def export_props(p: Package, e: Export) -> tuple[list[Prop], int]:
    """An export's tagged properties. Objects of most classes start with a 4-byte net index; actors
    placed in a level carry a script state frame before it (variable size). The first offset that parses
    cleanly to a `None` terminator wins."""
    buf = p.export_bytes(e)
    for start in (4, 0, 8, 12, *range(13, 96)):
        try:
            props, end = read_props(p, buf, start)
            return props, end
        except (PackageError, struct.error, IndexError, UnicodeDecodeError):
            continue
    raise PackageError(f"no property list found in {e.name}")


def load_keys(path: str | Path) -> list[bytes]:
    """A key file: one key per line, base64 (44 chars) or hex (64 chars); # comments allowed."""
    import base64

    keys = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        t = line.split("#", 1)[0].strip()
        if not t:
            continue
        k = bytes.fromhex(t) if len(t) == 64 and all(c in "0123456789abcdefABCDEF" for c in t) else base64.b64decode(t)
        if len(k) != 32:
            raise ValueError(f"{path}: a key is {len(k)} bytes, not 32")
        keys.append(k)
    return keys
