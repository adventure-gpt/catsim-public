"""Single-precision arithmetic in plain Python.

One +, -, * or / on two float32 values, done in double and rounded once to float32, gives exactly the
float32 result (a double holds more than twice float32's precision), so every helper here rounds after
each operation, in the order written.
"""
from __future__ import annotations

import struct


def F(x: float) -> float:
    """Round to float32."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def frombits(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def add(a, b):
    return F(a + b)


def sub(a, b):
    return F(a - b)


def mul(a, b):
    return F(a * b)


def div(a, b):
    return F(a / b)


C002 = F(0.02)      # Unreal units to physics units
C0025 = F(0.025)
