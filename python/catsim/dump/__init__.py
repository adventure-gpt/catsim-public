"""catSim's game-data dumper: reads your own Rocket League install (files on disk only) and
writes the data catSim loads. See docs/DUMPER.md.

    python -m catsim.dump --out catsim_data [--game FOLDER] [--keys KEYFILE]
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import install, versions
from .arena import soccar_pieces, write_arena
from .items import dump_items
from .tables import all_tables, write_tables
from .upk import load_keys, open_package

DATA_FORMAT = 1


def default_keys() -> list[bytes]:
    """Keys come only from the user: --keys, or CATSIM_RL_KEYS naming a key file."""
    env = os.environ.get("CATSIM_RL_KEYS")
    return load_keys(env) if env else []


def dump(out: Path, game: str | None = None, keys: list[bytes] | None = None, allow_unknown: bool = False,
         log=print, items: bool = True) -> dict:
    """Write the data folder. `items`: also read every car item's package for the random-draw model (a few
    minutes; without it, matches can't use the model's animated-parts timers)."""
    inst = install.resolve(game)
    log(f"Rocket League: {inst.path} ({inst.edition}, {inst.source}{', ' + inst.version if inst.version else ''})")
    fp = versions.fingerprint(inst.cooked, inst.exe)
    log(f"executable version {fp.exe_version or '?'}; " + (f"known: {fp.label}" if fp.label else "NOT a known version"))
    versions.require_known(fp, allow_unknown)
    keys = default_keys() if keys is None else keys
    if not keys:
        raise SystemExit("No package key given. The game's packages are encrypted and catSim ships no key: "
                         "pass --keys FILE (one AES-256 key per line, base64 or hex). See docs/DUMPER.md.")
    out = Path(out)
    stadium = open_package(inst.cooked / "Stadium_P.upk", keys)
    pieces = soccar_pieces(stadium)
    if len(pieces) != 16:
        raise SystemExit(f"expected the soccar arena's 16 collision meshes, found {len(pieces)}: unsupported layout")
    files = write_arena(out / "soccar", pieces)
    files.append(write_tables(out, all_tables(inst.cooked, keys, log=log)))
    if items:
        from .aes import BACKEND
        slow = "" if BACKEND == "cryptography" else (" - without the cryptography package this takes an hour or more: "
                                                      "pip install cryptography")
        log("reading the car items' packages for the random-draw model (a few minutes; --skip-items to leave out)" + slow)
        files.append(dump_items(inst.cooked, keys, out, log=log))
    manifest = {
        "format": DATA_FORMAT,
        "game": {"edition": inst.edition, "source": inst.source, "launcher_version": inst.version,
                 "exe_version": fp.exe_version, "known_version": fp.label,
                 "packages": {k: v for k, v in fp.packages.items()}},
        "files": {f.relative_to(out).as_posix(): versions.sha256(f) for f in files},
    }
    (out / "catsim_data.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    log(f"wrote {len(files)} files to {out}")
    return manifest
