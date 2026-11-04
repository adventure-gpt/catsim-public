"""The dumper's own building blocks, on synthetic input: AES, the cooked mesh reader, the mesh file writer."""
import struct

import pytest

from catsim.dump import aes
from catsim.dump.arena import MAGIC, Piece, _cooked, write_arena
from catsim.dump.upk import PackageError, load_keys


def test_aes256_known_answer():
    # FIPS-197 appendix C.3
    key = bytes(range(32))
    ct = bytes.fromhex("8ea2b7ca516745bfeafc49904b496089")
    assert aes.decrypt_ecb(key, ct) == bytes.fromhex("00112233445566778899aabbccddeeff")
    assert aes._PyAes(key).decrypt_ecb(ct * 2) == bytes.fromhex("00112233445566778899aabbccddeeff") * 2


def cooked(verts, tris, width):
    flags = {1: 0x8, 2: 0x10, 4: 0}[width]
    fmt = {1: "B", 2: "H", 4: "I"}[width]
    out = MAGIC + struct.pack("<iIIiiii", 1, flags, 0x3A83126F, 255, 0, len(verts), len(tris))
    for v in verts:
        out += struct.pack("<3f", *v)
    for t in tris:
        out += struct.pack("<3" + fmt, *t)
    return out


@pytest.mark.parametrize("width", [1, 2, 4])
def test_cooked_mesh_reader(width):
    verts = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, -0.0, 1.0)]
    buf = b"junk" + cooked(verts, [(0, 1, 2), (2, 1, 0)], width) + b"tail"
    v, t, end = _cooked(buf, 4)
    assert [struct.unpack("<3f", struct.pack("<3I", *x)) for x in v] == verts
    assert struct.pack("<I", v[2][1]) == struct.pack("<f", -0.0)      # the sign of zero survives
    assert t == [(0, 1, 2), (2, 1, 0)]
    assert buf[end:] == b"tail"


def test_cooked_mesh_rejects_bad_index():
    with pytest.raises(PackageError):
        _cooked(cooked([(0, 0, 0)], [(0, 0, 1)], 1), 0)


def test_write_arena(tmp_path):
    one = struct.unpack("<I", struct.pack("<f", 1.0))[0]
    pc = Piece("Thing_Collision", (1.0, -1.0, 1.0), [(0, 0, 0), (one, 0, 0), (0, one, 0)], [(0, 1, 2)], (0.0, 5120.0, 0.0))
    files = write_arena(tmp_path, [pc])
    data = (tmp_path / "mesh_0.cmf").read_bytes()
    assert struct.unpack_from("<ii", data) == (1, 3)
    assert struct.unpack_from("<3i", data, 8) == (0, 1, 2)
    line = (tmp_path / "arena.txt").read_text().splitlines()[-1].split()
    assert line[:2] == ["0", "Thing_Collision"] and line[3] == "bf800000" and line[6] == "45a00000"
    assert len(files) == 2


def test_key_file_formats(tmp_path):
    import base64
    k = bytes(range(32))
    f = tmp_path / "keys.txt"
    f.write_text(f"# a comment\n{base64.b64encode(k).decode()}\n{k.hex()}  # hex too\n\n")
    assert load_keys(f) == [k, k]
    f.write_text(base64.b64encode(b"short").decode())
    with pytest.raises(ValueError):
        load_keys(f)


def test_presetless_body_rule_rounds_rests_to_hundredths():
    from catsim.dump.f32 import bits, frombits
    from catsim.dump.tables import _hundredths
    # Plank's wheel rests as stored, and the Darkcar's as measured in the running game
    assert bits(_hundredths(frombits(0x41de6634))) == 0x41de6666     # 27.799904 -> 27.8
    assert bits(_hundredths(frombits(0x41a23d73))) == 0x41a23d71     # 20.280005 -> 20.28
    assert bits(_hundredths(frombits(0x4247e148))) == 0x4247e148     # 49.97 stays
    assert bits(_hundredths(frombits(0xc0fa8f5c))) == 0xc0fa8f5c     # -7.83 stays
