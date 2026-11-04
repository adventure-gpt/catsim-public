"""The C runtime's math tables that catSim's float routines use, computed from their definitions.

    python tools/gen_crt_tables.py           print each table as Rust
    python tools/gen_crt_tables.py --check   check every table in engine/*/src/math against the computation

The tables are plain math, so they are computed here rather than copied from anywhere:
- POW_INV[i] = 1 / (1/2 + i/512), i = 0..256, rounded to nearest double
- POW_LOG[i] = ln(1 + i/256), i = 0..256, rounded to nearest double
- POW_EXP[i] = 2^(i/64), i = 0..63, rounded to nearest double
- ATAN_TAB[i] = atan((i + 16) / 256), i = 0..240, rounded toward zero
Needs mpmath (pip install mpmath).
"""
import re
import struct
import sys
from pathlib import Path

import mpmath as mp

mp.mp.prec = 300
ROOT = Path(__file__).resolve().parents[1]


def bits(x) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def nearest(x) -> int:
    return bits(mp.mpf(x))


def toward_zero(x) -> int:
    b = bits(x)
    if abs(mp.mpf(struct.unpack("<d", struct.pack("<Q", b))[0])) > abs(x):
        b -= 1
    return b


def tables() -> dict:
    half = mp.mpf(1) / 2
    return {
        "POW_INV": [nearest(1 / (half + mp.mpf(i) / 512)) for i in range(257)],
        "POW_LOG": [nearest(mp.log(1 + mp.mpf(i) / 256)) for i in range(257)],
        "POW_EXP": [nearest(mp.mpf(2) ** (mp.mpf(i) / 64)) for i in range(64)],
        "ATAN_TAB": [toward_zero(mp.atan(mp.mpf(i + 16) / 256)) for i in range(241)],
    }


def check() -> int:
    want = tables()
    seen, bad = 0, 0
    for f in sorted((ROOT / "engine").glob("*/src/math/*.rs")):
        text = f.read_text(encoding="utf-8")
        for name, vals in want.items():
            m = re.search(r"const %s: \[u64; (\d+)\] = \[(.*?)\];" % name, text, re.S)
            if not m:
                continue
            got = [int(x, 16) for x in re.findall(r"0x[0-9a-fA-F]+", m.group(2))]
            seen += 1
            if got != vals:
                bad += 1
                print(f"MISMATCH {f.relative_to(ROOT)} {name}")
            else:
                print(f"ok {f.relative_to(ROOT)} {name} ({len(vals)} values)")
    if seen == 0:
        print("no tables found")
        return 1
    return 1 if bad else 0


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(check())
    for name, vals in tables().items():
        print(f"const {name}: [u64; {len(vals)}] = [")
        for i in range(0, len(vals), 6):
            print("    " + ", ".join(f"{v:#018x}" for v in vals[i:i + 6]) + ",")
        print("];")
