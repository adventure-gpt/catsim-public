"""python -m catsim.dump --out DIR [--game FOLDER] [--keys FILE] [--list] [--skip-items] [--allow-unknown-version]
python -m catsim.dump --check DIR"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import check, dump, install
from .upk import PackageError, load_keys
from .versions import DataMismatch, UnknownVersion


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m catsim.dump", description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("catsim_data"), help="folder to write (default ./catsim_data)")
    ap.add_argument("--game", help="the Rocket League folder (default: found from Steam or Epic)")
    ap.add_argument("--keys", type=Path, help="key file for the game's packages (default: $CATSIM_RL_KEYS)")
    ap.add_argument("--list", action="store_true", help="list the installs found and stop")
    ap.add_argument("--check", type=Path, metavar="DIR",
                    help="say whether a data folder holds the checked data (and which game and dumper made it), and stop")
    ap.add_argument("--skip-items", action="store_true",
                    help="leave out the car items (quicker; the random-draw model's animated parts then stay off)")
    ap.add_argument("--allow-unknown-version", action="store_true",
                    help="dump from a game version the dumper doesn't know; it then says whether the data equals the checked data")
    a = ap.parse_args(argv)
    if a.list:
        found = install.find_installs()
        for i in found:
            print(f"{i.edition:7} {i.path}  ({i.source}{', ' + i.version if i.version else ''})")
        if not found:
            print("no Rocket League install found")
        return 0 if found else 1
    if a.check:
        return 0 if check(a.check) else 1
    try:
        dump(a.out, a.game, load_keys(a.keys) if a.keys else None, a.allow_unknown_version, items=not a.skip_items)
    except (install.NotAnInstall, UnknownVersion, DataMismatch, PackageError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
