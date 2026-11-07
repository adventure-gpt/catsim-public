"""Which game versions the dumper knows, and how a game folder and a data folder are fingerprinted.

A version is known when every package the dumper reads has a hash listed here. Hashes are of
whole files, so they say nothing about the files' contents and carry no game data. The
packages are the same files in the Steam and Epic editions as far as we know; the executable's
own version number (read from its version resource) is reported but is not part of the check.

The data the checked versions give is fingerprinted the same way: CHECKED_DATA has the hash of every
physics file the dumper writes from them. A dump from a game update the dumper doesn't know yet can be
compared with it (the dumper does that itself, and `--check` does it for any data folder): game updates
often change packages without changing any physics number, and then the dump is the checked data.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path

# The packages read, and the dumps that were checked bit for bit against catSim's reference data.
PACKAGES = ("Stadium_P.upk", "UF_Night_P.upk", "Startup.upk", "TAGame.upk", "GameInfo_Soccar_SF.upk",
            "Body_Octane_SF.upk", "Body_Venom_SF.upk", "Body_Force_SF.upk", "Body_Vanquish_SF.upk", "Body_MuscleCar_SF.upk",
            "Body_Darkcar_SF.upk", "Body_Orion_SF.upk", "body_cured_SF.upk", "body_skunkz_SF.upk", "body_future_SF.upk",
            "body_aa_SF.upk", "body_hero_SF.upk", "body_pete_SF.upk", "body_smoke_SF.upk", "body_pixie_SF.upk",
            "body_bb_SF.upk", "body_lemonade_SF.upk", "body_tritip_SF.upk")

KNOWN = {
    # label: {package: sha256 prefix}
    # Stadium_P, UF_Night_P, Startup and TAGame changed from 25914621; the dumper writes the same data (CHECKED_DATA).
    "steam build 26349479 (engine 10897), checked 2026-11-06": {
        'Stadium_P.upk': '7d496c9238f50a26', 'UF_Night_P.upk': '42183be346cf3730', 'Startup.upk': '343efb7f02734cd9',
        'TAGame.upk': '599bbd96ae751d77', 'GameInfo_Soccar_SF.upk': '9cd807e130f9795a', 'Body_Octane_SF.upk': '3359efe77733d8aa',
        'Body_Venom_SF.upk': '0866c1686cf7820a', 'Body_Force_SF.upk': '39682105e7a1a262', 'Body_Vanquish_SF.upk': '2edc8f5b66131d51',
        'Body_MuscleCar_SF.upk': '754da1d5b8b6c2a2', 'Body_Darkcar_SF.upk': '2f4553f467668ba8', 'Body_Orion_SF.upk': '486c5cf1be02c0ad',
        'body_cured_SF.upk': '1ebeba31aad1fb5a', 'body_skunkz_SF.upk': 'b10dc3efb1b025fc', 'body_future_SF.upk': '33c555c4e21704bc',
        'body_aa_SF.upk': '082b4c98e77473e8', 'body_hero_SF.upk': 'e0c9449c56fcc599', 'body_pete_SF.upk': 'e5d51b16536f76e7',
        'body_smoke_SF.upk': '681fb3212dd08adb', 'body_pixie_SF.upk': 'b933c250cc8e4961', 'body_bb_SF.upk': '312b107feec4448f',
        'body_lemonade_SF.upk': '5b3085d5d3115d30', 'body_tritip_SF.upk': 'f0141182c8770b8b',
    },
    "steam build 25914621 (engine 10897), checked 2026-11-04": {
        'Stadium_P.upk': 'fa18d43557e6cec7', 'UF_Night_P.upk': 'c0014a583c1b3f15', 'Startup.upk': 'c4050f0a05cf3042',
        'TAGame.upk': '6294bbe9ed83f759', 'GameInfo_Soccar_SF.upk': '9cd807e130f9795a', 'Body_Octane_SF.upk': '3359efe77733d8aa',
        'Body_Venom_SF.upk': '0866c1686cf7820a', 'Body_Force_SF.upk': '39682105e7a1a262', 'Body_Vanquish_SF.upk': '2edc8f5b66131d51',
        'Body_MuscleCar_SF.upk': '754da1d5b8b6c2a2', 'Body_Darkcar_SF.upk': '2f4553f467668ba8', 'Body_Orion_SF.upk': '486c5cf1be02c0ad',
        'body_cured_SF.upk': '1ebeba31aad1fb5a', 'body_skunkz_SF.upk': 'b10dc3efb1b025fc', 'body_future_SF.upk': '33c555c4e21704bc',
        'body_aa_SF.upk': '082b4c98e77473e8', 'body_hero_SF.upk': 'e0c9449c56fcc599', 'body_pete_SF.upk': 'e5d51b16536f76e7',
        'body_smoke_SF.upk': '681fb3212dd08adb', 'body_pixie_SF.upk': 'b933c250cc8e4961', 'body_bb_SF.upk': '312b107feec4448f',
        'body_lemonade_SF.upk': '5b3085d5d3115d30', 'body_tritip_SF.upk': 'f0141182c8770b8b',
    },
    "steam build 25641912 (engine 10897), checked 2026-11-04": {
        'Stadium_P.upk': '206ff6708565fd8e', 'UF_Night_P.upk': 'dbed668db1ca9b65', 'Startup.upk': '062682f1c9c5bca1',
        'TAGame.upk': 'ceda0affb4986cfa', 'GameInfo_Soccar_SF.upk': '8e2fcc7405c080cb', 'Body_Octane_SF.upk': 'bedf7fc0d64ab2c6',
        'Body_Venom_SF.upk': 'daa2950e28ee42bf', 'Body_Force_SF.upk': '03f77f543886ec9f', 'Body_Vanquish_SF.upk': '6de6470e8cd98059',
        'Body_MuscleCar_SF.upk': '935fdadec008ff47', 'Body_Darkcar_SF.upk': 'f0a89b3a20ca19e6', 'Body_Orion_SF.upk': '58909940386c1664',
        'body_cured_SF.upk': '4d1cad42846370d6', 'body_skunkz_SF.upk': '0a4559cfaa485364', 'body_future_SF.upk': '5a79638cbd13c3f3',
        'body_aa_SF.upk': '177c1896c164b87b', 'body_hero_SF.upk': 'ff8b5a782174e84e', 'body_pete_SF.upk': '1a62f3b38aba7c69',
        'body_smoke_SF.upk': '17a8de2432b73e77', 'body_pixie_SF.upk': 'b4aa55a7c93b9d45', 'body_bb_SF.upk': 'ab62838ca047a14c',
        'body_lemonade_SF.upk': '5db5abe80a6385fa', 'body_tritip_SF.upk': 'e5a702c47001de47',
    },
}

# The physics files the dumper writes. items.txt is not one: new car items come with most game updates, and they
# feed only the random-draw model.
DATA_FILES = ("tables.txt", "soccar/arena.txt") + tuple(f"soccar/mesh_{i}.cmf" for i in range(16))

CHECKED_DATA = {
    # label: {file: sha256 prefix}, for the data format in catsim.dump.DATA_FORMAT.
    # Every checked build above gives exactly these files.
    "the checked data of steam builds 25641912, 25914621 and 26349479 (data format 1)": {
        'tables.txt': '265125629e021293', 'soccar/arena.txt': '4802ff560609bb06',
        'soccar/mesh_0.cmf': '8764d43b87ba134c', 'soccar/mesh_1.cmf': '9d42e5db7cde5c7b', 'soccar/mesh_2.cmf': '762ec145db3c317d',
        'soccar/mesh_3.cmf': '0db5c559287bcd25', 'soccar/mesh_4.cmf': 'cde687837642fb35', 'soccar/mesh_5.cmf': '76e8bacda25a7804',
        'soccar/mesh_6.cmf': '22b1576b37f9f187', 'soccar/mesh_7.cmf': 'bf779e51b0e0ff17', 'soccar/mesh_8.cmf': '3350567082d9a133',
        'soccar/mesh_9.cmf': '4864911a7f0a3add', 'soccar/mesh_10.cmf': '36cc44e14a0e500a', 'soccar/mesh_11.cmf': '7f4468be0b0835a4',
        'soccar/mesh_12.cmf': '952a90a0ff27c273', 'soccar/mesh_13.cmf': 'b441f782f07e81c2', 'soccar/mesh_14.cmf': '085f7533c53a73da',
        'soccar/mesh_15.cmf': '21c1d1eb450f9e09',
    },
}
# Only a prefix of each hash is kept: enough to tell versions apart.
PREFIX = 16
ISSUES = "https://github.com/adventure-gpt/catsim-public/issues"


class UnknownVersion(Exception):
    pass


class DataMismatch(Exception):
    pass


@dataclass
class Fingerprint:
    packages: dict[str, str]
    exe_version: str
    label: str | None


@dataclass
class DataCheck:
    label: str | None                                   # the checked data the folder holds, or None
    differ: list[str] = field(default_factory=list)     # files that differ from the closest checked data
    missing: list[str] = field(default_factory=list)    # physics files the folder doesn't have

    def problems(self) -> str:
        parts = []
        if self.differ:
            parts.append("differs from the checked data in " + ", ".join(self.differ))
        if self.missing:
            parts.append("has no " + ", ".join(self.missing))
        return "; it ".join(parts)

    def describe(self) -> str:
        return f"the same files as {self.label}" if self.label else f"NOT the checked data: it {self.problems()}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def exe_version(exe: Path) -> str:
    """The executable's file version from its version resource, found by the resource's own
    signature (0xFEEF04BD) so it works on any edition's build. Empty if absent."""
    data = exe.read_bytes()
    at = data.find(struct.pack("<I", 0xFEEF04BD))
    if at < 0 or at + 16 > len(data):
        return ""
    ms, ls = struct.unpack_from("<II", data, at + 8)
    return f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}"


def match(packages: dict[str, str]) -> str | None:
    for lab, want in KNOWN.items():
        if all(packages.get(n, "").startswith(h) for n, h in want.items()):
            return lab
    return None


def fingerprint(cooked: Path, exe: Path) -> Fingerprint:
    pk = {name: sha256(cooked / name) for name in PACKAGES}
    return Fingerprint(pk, exe_version(exe) if exe.is_file() else "", match(pk))


def nearest_known(packages: dict[str, str]) -> tuple[str, list[str]]:
    """The known version that shares the most packages with these, and the packages that differ from it."""
    best = None
    for lab, want in KNOWN.items():
        changed = [n for n in PACKAGES if not packages.get(n, "").startswith(want[n])]
        if best is None or len(changed) < len(best[1]):
            best = (lab, changed)
    return best


def require_known(fp: Fingerprint, allow_unknown: bool) -> None:
    if fp.label or allow_unknown:
        return
    near, changed = nearest_known(fp.packages)
    rows = "\n".join(f"  {n}: {fp.packages[n][:PREFIX]}" for n in changed)
    raise UnknownVersion(
        f"This Rocket League version is not one catSim's dumper knows (game executable {fp.exe_version or 'version unknown'}).\n"
        f"Closest known version: {near}. {len(changed)} of the {len(PACKAGES)} packages the dumper reads differ from it:\n"
        f"{rows}\n"
        "Game updates often change these files without changing any physics number. What to do:\n"
        "  1. Update catSim (pip install -U \"catsim-rl[dump]\"): a newer release may know this version.\n"
        "  2. If it still doesn't, run the same command with --allow-unknown-version added. The dumper checks every\n"
        "     file's structure as always, then compares what it wrote with the checked data and says which it is:\n"
        "     the same files (then you have exactly the data that was checked against the game) or which files\n"
        "     differ (then nobody has checked those numbers against the game yet).\n"
        f"  3. Either way, please tell us the version and these hashes: {ISSUES}")


def check_data(folder: Path) -> DataCheck:
    """Does this data folder hold exactly the checked data? Compares the hash of each physics file."""
    folder = Path(folder)
    have = {n: (sha256(folder / n) if (folder / n).is_file() else None) for n in DATA_FILES}
    missing = [n for n in DATA_FILES if have[n] is None]
    best = None
    for lab, want in CHECKED_DATA.items():
        differ = [n for n in DATA_FILES if have[n] is not None and not have[n].startswith(want[n])]
        if not differ and not missing:
            return DataCheck(lab)
        if best is None or len(differ) < len(best.differ):
            best = DataCheck(None, differ, missing)
    return best
