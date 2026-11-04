"""Rotations in float32, written to give the same bits as catSim's engine (engine/core/src/math/rot.rs,
game/arena.rs fmatrix_to_quat, game/rb_state.rs quat_to_btmatrix): a rotator's matrix through the game's
14-bit sine table, a matrix to a quaternion, a quaternion to a physics basis.
"""
from __future__ import annotations

import math

from .f32 import F, add, bits, div, frombits, mul, sub

_PI = frombits(0x40490FDB)
_INV16384 = frombits(0x38800000)


def _sinf(x: float) -> float:
    """sinf, correctly rounded. Used only for the sine table's entries at multiples of a quarter turn,
    where every C runtime agrees (checked against the engine's own table in tests)."""
    try:
        import mpmath as mp
        mp.mp.prec = 120
        return F(float(mp.sin(mp.mpf(x))))
    except ImportError:
        return F(math.sin(x))


def table(i: int) -> float:
    f = F(float(i))
    return _sinf(mul(mul(add(f, f), _PI), _INV16384))


def tsin(units: int) -> float:
    return table((units >> 2) & 0x3FFF)


def tcos(units: int) -> float:
    return table(((units + 0x4000) >> 2) & 0x3FFF)


def rotator_to_matrix(pitch: int, yaw: int, roll: int) -> list[list[float]]:
    sr, sp, sy = tsin(roll), tsin(pitch), tsin(yaw)
    cr, cp, cy = tcos(roll), tcos(pitch), tcos(yaw)
    return [
        [mul(cy, cp), mul(cp, sy), sp],
        [sub(mul(mul(sp, sr), cy), mul(cr, sy)), add(mul(cy, cr), mul(mul(sp, sr), sy)), mul(-sr, cp)],
        [-add(mul(mul(cr, sp), cy), mul(sy, sr)), sub(mul(cy, sr), mul(mul(cr, sp), sy)), mul(cp, cr)],
    ]


def _sqrt(x: float) -> float:
    return F(math.sqrt(x))


def matrix_to_quat(m) -> tuple[float, float, float, float]:
    m0, m1, m2 = m

    def small(r):
        return abs(r[0]) < 0.0001 and abs(r[1]) < 0.0001 and abs(r[2]) < 0.0001

    if small(m0) and small(m1) and small(m2):
        return (0.0, 0.0, 0.0, 1.0)
    t = add(add(m1[1], m0[0]), m2[2])
    if t <= 0.0:
        d = [m0[0], m1[1], m2[2]]
        i = 1 if m0[0] < m1[1] else 0
        if d[i] <= m2[2] and m2[2] != d[i]:
            i = 2
        nxt = [1, 2, 0]
        j, k = nxt[i], nxt[nxt[i]]
        f = div(1.0, _sqrt(add(sub(sub(d[i], d[j]), d[k]), 1.0)))
        h = mul(f, 0.5)
        q = [0.0, 0.0, 0.0, 0.0]
        q[i] = div(0.5, f)
        q[3] = mul(sub(m[j][k], m[k][j]), h)
        q[j] = mul(add(m[i][j], m[j][i]), h)
        q[k] = mul(add(m[i][k], m[k][i]), h)
        return tuple(q)
    f = div(1.0, _sqrt(add(t, 1.0)))
    h = mul(f, 0.5)
    return (mul(sub(m1[2], m2[1]), h), mul(sub(m2[0], m0[2]), h), mul(sub(m0[1], m1[0]), h), div(0.5, f))


def quat_to_basis(q) -> list[list[float]]:
    x, y, z, w = q
    s = div(2.0, add(add(add(mul(y, y), mul(x, x)), mul(z, z)), mul(w, w)))
    zs, ys = mul(s, z), mul(s, y)
    xxs, xws = mul(mul(s, x), x), mul(mul(s, x), w)
    return [
        [sub(1.0, add(mul(zs, z), mul(ys, y))), sub(mul(ys, x), mul(zs, w)), add(mul(zs, x), mul(ys, w))],
        [add(mul(ys, x), mul(zs, w)), sub(1.0, add(mul(zs, z), xxs)), sub(mul(zs, y), xws)],
        [sub(mul(zs, x), mul(ys, w)), add(mul(zs, y), xws), sub(1.0, add(mul(ys, y), xxs))],
    ]


def scaled_basis(pitch: int, yaw: int, roll: int, scale) -> list[list[float]]:
    """A placed component's physics basis: its rotation matrix with each row scaled by Scale3D, the scale
    taken out again (each row times 1/sqrt of its squared length, in float32), through a quaternion."""
    m = rotator_to_matrix(pitch, yaw, roll)
    rows = []
    for r, s in zip(m, scale):
        v = [mul(c, s) for c in r]
        ss = add(add(mul(v[0], v[0]), mul(v[1], v[1])), mul(v[2], v[2]))
        k = div(1.0, _sqrt(ss))
        rows.append([mul(c, k) for c in v])
    return quat_to_basis(matrix_to_quat(rows))


def basis_bits(b) -> list[int]:
    return [bits(v) for row in b for v in row]
