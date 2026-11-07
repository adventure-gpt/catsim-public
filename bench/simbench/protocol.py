"""The wire between the harness and a simulator's worker process.

Every adapter runs in its own process and talks to the harness over its standard input and output, so a simulator of
any language and any licence plugs in without being linked into, or copied into, this repository.

A message is a 4-byte little-endian length, that many bytes of UTF-8 JSON (the header), then `header["bin"]` bytes of
binary payload (little-endian float32 arrays; 0 when absent). Replies have the same shape. A reply header with
`"error"` means the request failed; the worker keeps running. docs/BENCHMARKS.md lists every request.
"""
from __future__ import annotations

import json
import struct
import sys
from typing import BinaryIO

PROTOCOL = 1


def send(out: BinaryIO, header: dict, payload: bytes = b"") -> None:
    header = dict(header, bin=len(payload))
    h = json.dumps(header, separators=(",", ":")).encode()
    out.write(struct.pack("<I", len(h)))
    out.write(h)
    if payload:
        out.write(payload)
    out.flush()


def _read_exact(inp: BinaryIO, n: int) -> bytes:
    chunks, got = [], 0
    while got < n:
        b = inp.read(n - got)
        if not b:
            raise EOFError
        chunks.append(b)
        got += len(b)
    return b"".join(chunks)


def recv(inp: BinaryIO) -> tuple[dict, bytes]:
    n = struct.unpack("<I", _read_exact(inp, 4))[0]
    header = json.loads(_read_exact(inp, n))
    k = int(header.get("bin", 0))
    return header, (_read_exact(inp, k) if k else b"")


def stdio() -> tuple[BinaryIO, BinaryIO]:
    """The worker's ends: binary stdin and stdout (anything a simulator prints goes to stderr instead)."""
    out = sys.stdout.buffer
    sys.stdout = sys.stderr
    return sys.stdin.buffer, out
