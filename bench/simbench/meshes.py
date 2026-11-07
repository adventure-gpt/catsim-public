"""RocketSim's collision meshes from catSim's dumper output, so one dump from your copy of the game serves every
simulator here.

    python -m simbench.meshes CATSIM_DATA OUT_DIR        # writes OUT_DIR/soccar/mesh_0.cmf ... mesh_15.cmf

The dumper writes each arena piece in its own frame and its placement in `soccar/arena.txt`; RocketSim's mesh files
(from RLArenaCollisionDumper) hold the goal pieces already moved to their goal. This moves them the same way and checks
every file against the SHA-256 that RocketSim's dumper gives on the same game build (listed below): a file that doesn't
match is reported, never used silently. Beside each moved goal piece it also writes `mesh_N.local`, the piece's own
vertices as the game holds them (little-endian f32 x, y, z per vertex), which RLMAX's loader reads; the others ignore it.
"""
from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path

import numpy as np

GOAL_BT = np.frombuffer(bytes.fromhex("cbcccc42"), "<f4")[0]     # 102.39998626708984

# The RocketSim soccar mesh files for the game builds catSim's dumper knows (RLArenaCollisionDumper output).
KNOWN = {
    "mesh_0.cmf": "8764d43b87ba134cacf1be454d59d493630188e79c0d88639258b359bb511c4e",
    "mesh_1.cmf": "9d42e5db7cde5c7be0783c256e15e819bd09351fcb2debd3205e3d055073c993",
    "mesh_2.cmf": "762ec145db3c317d82314ad94af6f8b591e822c26860ed3c0ee106aba5591eee",
    "mesh_3.cmf": "0db5c559287bcd255b99e6f5db341b874ec9a5fcf6137247cf16d913e178db5c",
    "mesh_4.cmf": "92c3224da44f1a359140647845a44457df35541458b92485e7a3232bb596566a",
    "mesh_5.cmf": "776c24b1231b4f7ab5437af68e4cc1cc4d5cd7875b8cb386a67487162c3941a3",
    "mesh_6.cmf": "a2262bfa2bfacca3caf3a8334f02b9b52624056853cc33a1340a23c3a33804e6",
    "mesh_7.cmf": "c887df022d6a2b33e78a3cf8f5dd65d9cd61f4d72009781d40c32091119a06f3",
    "mesh_8.cmf": "3350567082d9a133f59f8471d941fde6cce20eb2b2bc26c2ab0566b7fb7a21e9",
    "mesh_9.cmf": "4864911a7f0a3add929f41cc8f676cde416dcc3c07ab08a30742c9893e8a6ce8",
    "mesh_10.cmf": "36cc44e14a0e500aeb6167a4d5fce449b218b3d941df3ca46f398d269b01af3d",
    "mesh_11.cmf": "7f4468be0b0835a48996f92d1a2398f1290acbf1152d39e1c53c2701d1fb6ce3",
    "mesh_12.cmf": "952a90a0ff27c2733d507e59bdca8600968276eddad9a41b47a689010e805118",
    "mesh_13.cmf": "b441f782f07e81c2444ed7a5c4ed618c26ef3e1940ac2c63c362154620631750",
    "mesh_14.cmf": "085f7533c53a73da664c129c3b1c93f5ce08ceb302aa5b993ff5fe95af5d6ae8",
    "mesh_15.cmf": "21c1d1eb450f9e09ee8d7d55e1d41bf563518f9b2deb2ab5369c47139faa2e4a",
}


def convert(data: Path, out: Path) -> list[str]:
    src = data / "soccar" if (data / "soccar").is_dir() else data
    dst = out / "soccar"
    dst.mkdir(parents=True, exist_ok=True)
    problems = []
    for line in (src / "arena.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        i = int(parts[0])
        t = [struct.unpack(">f", bytes.fromhex(h))[0] for h in parts[5:8]]
        raw = (src / f"mesh_{i}.cmf").read_bytes()
        nt, nv = struct.unpack_from("<ii", raw, 0)
        head = raw[:8 + nt * 12]
        v = np.frombuffer(raw, np.float32, nv * 3, 8 + nt * 12).reshape(nv, 3).copy()
        if any(x != 0.0 for x in t):
            # the piece in its own frame, as the package holds it, for simulators that read it (RLMAX's mesh_N.local):
            # the world floats above have lost its low bits
            (dst / f"mesh_{i}.local").write_bytes(v.astype("<f4").tobytes())
        for k in range(3):
            if t[k] != 0.0:
                # RocketSim's files hold the goals 2 float steps nearer the middle than the package's 5120 uu placement
                # (102.39998627 Bullet units); reproduce them exactly
                v[:, k] = v[:, k] + np.float32(GOAL_BT if t[k] > 0 else -GOAL_BT)
            v[v == 0.0] = 0.0          # and their zeros positive (the package's mirrored goal halves hold -0.0)
        data_out = head + v.astype("<f4").tobytes() + raw[8 + nt * 12 + nv * 12:]
        name = f"mesh_{i}.cmf"
        (dst / name).write_bytes(data_out)
        if hashlib.sha256(data_out).hexdigest() != KNOWN.get(name):
            problems.append(name)
    return problems


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    bad = convert(Path(sys.argv[1]), Path(sys.argv[2]))
    if bad:
        raise SystemExit(f"these files differ from RocketSim's dumper's on the known game builds: {', '.join(bad)}")
    print(f"wrote {sys.argv[2]}/soccar: 16 mesh files, each equal to RocketSim's dumper's")


if __name__ == "__main__":
    main()
