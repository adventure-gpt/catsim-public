"""catSim's game-data dumper: reads your own Rocket League install (files on disk only) and
writes the data catSim loads. See docs/DUMPER.md.

    python -m catsim.dump --out catsim_data [--game FOLDER] [--keys KEYFILE]
    python -m catsim.dump --check catsim_data           (is a data folder the checked data?)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .. import __version__
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
    data_check = versions.check_data(out)
    manifest = {
        "format": DATA_FORMAT,
        "dumper": f"catsim {__version__}",
        "game": {"edition": inst.edition, "source": inst.source, "launcher_version": inst.version,
                 "exe_version": fp.exe_version, "known_version": fp.label,
                 "packages": {k: v for k, v in fp.packages.items()}},
        "checked_data": data_check.label,
        "files": {f.relative_to(out).as_posix(): versions.sha256(f) for f in files},
    }
    (out / "catsim_data.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    log(f"wrote {len(files)} files to {out}")
    log(f"data: {data_check.describe()}")
    if fp.label and not data_check.label:
        # A known version always gives the checked data: anything else is the dumper's own fault.
        raise versions.DataMismatch(
            f"{fp.label} should give the checked data, but this dump {data_check.problems()}. That is a bug in this "
            f"copy of the dumper (catsim {__version__}), not in your game: don't use this data, and please report it "
            f"at {versions.ISSUES}")
    if not data_check.label:
        log("nobody has checked these numbers against the game: catSim may not match it with this data")
    return manifest


def check(folder: Path, log=print) -> bool:
    """Say what a data folder is: which game and dumper made it, whether its files are still the ones written, and
    whether it holds the checked data. True when it does."""
    folder = Path(folder)
    mf = folder / "catsim_data.json"
    if mf.is_file():
        m = json.loads(mf.read_text(encoding="utf-8"))
        g = m.get("game", {})
        log(f"made by: {m.get('dumper') or 'a dumper older than catsim 0.1.2 (it does not say which)'}, "
            f"data format {m.get('format')}")
        log(f"from: {g.get('edition', '?')} {g.get('launcher_version') or ''} (executable {g.get('exe_version') or '?'}); "
            + (f"known: {g['known_version']}" if g.get("known_version") else "not a known version"))
        changed = [n for n, h in m.get("files", {}).items()
                   if not (folder / n).is_file() or versions.sha256(folder / n) != h]
        if changed:
            log(f"changed or missing since the dump: {', '.join(changed)}")
    else:
        log(f"no catsim_data.json in {folder}: not a folder the dumper wrote, or incomplete")
    c = versions.check_data(folder)
    log(f"data: {c.describe()}")
    return c.label is not None
