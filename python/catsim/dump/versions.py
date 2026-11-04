"""Which game versions the dumper knows, and how a game folder is fingerprinted.

A version is known when every package the dumper reads has a hash listed here. Hashes are of
whole files, so they say nothing about the files' contents and carry no game data. The
packages are the same files in the Steam and Epic editions as far as we know; the executable's
own version number (read from its version resource) is reported but is not part of the check.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

# The packages read, and the dumps that were checked bit for bit against catSim's reference data.
PACKAGES = ("Stadium_P.upk", "UF_Night_P.upk", "Startup.upk", "TAGame.upk", "GameInfo_Soccar_SF.upk",
            "Body_Octane_SF.upk", "Body_Venom_SF.upk", "Body_Force_SF.upk", "Body_Vanquish_SF.upk", "Body_MuscleCar_SF.upk",
            "Body_Darkcar_SF.upk", "Body_Orion_SF.upk", "body_cured_SF.upk", "body_skunkz_SF.upk", "body_future_SF.upk",
            "body_aa_SF.upk", "body_hero_SF.upk", "body_pete_SF.upk", "body_smoke_SF.upk", "body_pixie_SF.upk",
            "body_bb_SF.upk", "body_lemonade_SF.upk", "body_tritip_SF.upk")

KNOWN = {
    # label: {package: sha256 prefix}
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
    # Its tables and arena meshes are identical to the build above's.
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
# Only a prefix of each hash is kept: enough to tell versions apart.
PREFIX = 16


class UnknownVersion(Exception):
    pass


@dataclass
class Fingerprint:
    packages: dict[str, str]
    exe_version: str
    label: str | None


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


def fingerprint(cooked: Path, exe: Path) -> Fingerprint:
    pk = {name: sha256(cooked / name) for name in PACKAGES}
    label = None
    for lab, want in KNOWN.items():
        if all(pk[n].startswith(h) for n, h in want.items()):
            label = lab
            break
    return Fingerprint(pk, exe_version(exe) if exe.is_file() else "", label)


def require_known(fp: Fingerprint, allow_unknown: bool) -> None:
    if fp.label or allow_unknown:
        return
    rows = "\n".join(f"  {n}: {h[:PREFIX]}" for n, h in fp.packages.items())
    raise UnknownVersion(
        "This Rocket League version is not one catSim's dumper has been checked against.\n"
        f"Game executable version {fp.exe_version or '(unknown)'}; package hashes:\n{rows}\n"
        "The game may have changed its physics or its files. Re-run with --allow-unknown-version to try "
        "anyway: the dumper still checks every file's structure and refuses anything that doesn't fit, "
        "but nobody has checked that version's numbers against the real game.")
