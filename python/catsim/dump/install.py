"""Finding Rocket League on disk: the Steam and Epic Games editions, on Windows and Linux.

Only files are read: Steam's registry key and library list, Epic's install manifests, and the
game folder itself. Nothing is launched and no process is touched.

    find_installs()          every install found, best first
    check_game_dir(path)     is this folder a Rocket League install? (raises with the reason)
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

STEAM_APP_ID = "252950"
EPIC_APP_NAME = "Sugar"          # Rocket League's catalog name in Epic's launcher manifests
EXE = Path("Binaries") / "Win64" / "RocketLeague.exe"
COOKED = Path("TAGame") / "CookedPCConsole"
MARKER = COOKED / "Stadium_P.upk"


@dataclass
class Install:
    path: Path
    edition: str      # "steam", "epic" or "unknown"
    source: str       # how it was found
    version: str = "" # the launcher's own build label, if it gives one

    @property
    def cooked(self) -> Path:
        return self.path / COOKED

    @property
    def exe(self) -> Path:
        return self.path / EXE


class NotAnInstall(Exception):
    pass


def check_game_dir(path: str | os.PathLike) -> Path:
    p = Path(path).expanduser()
    if not p.is_dir():
        raise NotAnInstall(f"{p} is not a folder")
    # Accept the game folder itself or anything inside it (people paste Binaries/Win64 or CookedPCConsole).
    for cand in [p, *p.parents]:
        if (cand / MARKER).is_file():
            if not (cand / EXE).is_file():
                raise NotAnInstall(f"{cand} has the game's packages but no {EXE.as_posix()}")
            return cand
    raise NotAnInstall(f"{p} does not look like Rocket League: no {MARKER.as_posix()} in it or above it")


# ---------------------------------------------------------------- Steam

def _vdf_paths(text: str) -> list[str]:
    """Library folders from libraryfolders.vdf (both the old and the new layout)."""
    out = [m.group(1) for m in re.finditer(r'"path"\s+"([^"]+)"', text)]
    out += [m.group(1) for m in re.finditer(r'^\s*"\d+"\s+"([^"]+)"', text, re.M)]   # pre-2021 layout
    return [s.replace("\\\\", "\\") for s in out]


def _acf_value(text: str, key: str) -> str | None:
    m = re.search(r'"%s"\s+"([^"]*)"' % re.escape(key), text)
    return m.group(1) if m else None


def steam_roots(env: dict | None = None, home: Path | None = None) -> list[Path]:
    env = os.environ if env is None else env
    home = Path.home() if home is None else home
    roots: list[Path] = []
    if sys.platform == "win32" and env is os.environ:
        roots += _steam_from_registry()
    pf86 = env.get("ProgramFiles(x86)")
    if pf86:
        roots.append(Path(pf86) / "Steam")
    roots += [home / ".steam" / "steam", home / ".steam" / "root", home / ".local" / "share" / "Steam",
              home / ".var" / "app" / "com.valvesoftware.Steam" / ".local" / "share" / "Steam",
              home / "snap" / "steam" / "common" / ".local" / "share" / "Steam"]
    seen, out = set(), []
    for r in roots:
        key = str(r).lower()
        if key not in seen and r.is_dir():
            seen.add(key)
            out.append(r)
    return out


def _steam_from_registry() -> list[Path]:
    out = []
    try:
        import winreg
    except ImportError:
        return out
    for hive, key, val in ((winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam", "SteamPath"),
                           (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam", "InstallPath"),
                           (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam", "InstallPath")):
        try:
            with winreg.OpenKey(hive, key) as k:
                out.append(Path(winreg.QueryValueEx(k, val)[0]))
        except OSError:
            pass
    return out


def find_steam(roots: list[Path] | None = None) -> list[Install]:
    found = []
    libs: list[Path] = []
    for root in (steam_roots() if roots is None else roots):
        libs.append(root)
        for vdf in (root / "steamapps" / "libraryfolders.vdf", root / "config" / "libraryfolders.vdf"):
            if vdf.is_file():
                libs += [Path(p) for p in _vdf_paths(vdf.read_text(encoding="utf-8", errors="replace"))]
    seen = set()
    for lib in libs:
        acf = lib / "steamapps" / f"appmanifest_{STEAM_APP_ID}.acf"
        if not acf.is_file():
            continue
        text = acf.read_text(encoding="utf-8", errors="replace")
        game = lib / "steamapps" / "common" / (_acf_value(text, "installdir") or "rocketleague")
        key = str(game).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            found.append(Install(check_game_dir(game), "steam", f"Steam library {lib}", _acf_value(text, "buildid") or ""))
        except NotAnInstall:
            pass
    return found


# ---------------------------------------------------------------- Epic

def epic_data_dirs(env: dict | None = None, home: Path | None = None) -> list[Path]:
    """Folders that hold Epic's launcher data: ProgramData on Windows, and the same folder inside
    Wine/Proton prefixes on Linux (Lutris, Bottles and plain Wine keep a drive_c)."""
    env = os.environ if env is None else env
    home = Path.home() if home is None else home
    out = []
    pd = env.get("ProgramData") or env.get("ALLUSERSPROFILE")
    if pd:
        out.append(Path(pd) / "Epic")
    elif sys.platform == "win32":
        out.append(Path(r"C:\ProgramData\Epic"))
    for prefix in [home / ".wine", *sorted((home / "Games").glob("*")), *sorted((home / ".local" / "share" / "lutris" / "runners").glob("*"))]:
        cand = prefix / "drive_c" / "ProgramData" / "Epic"
        if cand.is_dir():
            out.append(cand)
    return [p for p in out if p.is_dir()]


def _is_rl(item: dict) -> bool:
    return item.get("AppName") == EPIC_APP_NAME or "rocket league" in str(item.get("DisplayName", "")).lower()


def find_epic(data_dirs: list[Path] | None = None, home: Path | None = None) -> list[Install]:
    found, seen = [], set()

    def add(path: str, source: str, version: str) -> None:
        if not path:
            return
        try:
            game = check_game_dir(path)
        except NotAnInstall:
            return
        key = str(game).lower()
        if key not in seen:
            seen.add(key)
            found.append(Install(game, "epic", source, version))

    for epic in (epic_data_dirs() if data_dirs is None else data_dirs):
        for item in sorted((epic / "EpicGamesLauncher" / "Data" / "Manifests").glob("*.item")):
            try:
                data = json.loads(item.read_text(encoding="utf-8-sig", errors="replace"))
            except (OSError, ValueError):
                continue
            if _is_rl(data):
                add(data.get("InstallLocation", ""), f"Epic manifest {item.name}", str(data.get("AppVersionString", "")))
        dat = epic / "UnrealEngineLauncher" / "LauncherInstalled.dat"
        if dat.is_file():
            try:
                data = json.loads(dat.read_text(encoding="utf-8-sig", errors="replace"))
            except (OSError, ValueError):
                data = {}
            for row in data.get("InstallationList", []):
                if _is_rl(row):
                    add(row.get("InstallLocation", ""), "Epic LauncherInstalled.dat", str(row.get("AppVersion", "")))
    # Linux launchers for the Epic edition (Heroic, legendary) keep their own list.
    home = Path.home() if home is None else home
    for f in (home / ".config" / "legendary" / "installed.json",
              home / ".config" / "heroic" / "legendaryConfig" / "legendary" / "installed.json",
              home / ".var" / "app" / "com.heroicgameslauncher.hgl" / "config" / "heroic" / "legendaryConfig" / "legendary" / "installed.json"):
        if f.is_file():
            try:
                data = json.loads(f.read_text(encoding="utf-8", errors="replace"))
            except (OSError, ValueError):
                continue
            for name, row in (data.items() if isinstance(data, dict) else []):
                if name == EPIC_APP_NAME or _is_rl(row if isinstance(row, dict) else {}):
                    add(row.get("install_path", ""), f"legendary list {f}", str(row.get("version", "")))
    return found


def find_installs() -> list[Install]:
    """Every install found on this machine: Steam first, then Epic."""
    out, seen = [], set()
    for inst in find_steam() + find_epic():
        key = str(inst.path).lower()
        if key not in seen:
            seen.add(key)
            out.append(inst)
    env_dir = os.environ.get("CATSIM_RL_DIR")
    if env_dir:
        out.insert(0, Install(check_game_dir(env_dir), "unknown", "CATSIM_RL_DIR"))
    return out


def resolve(game: str | None) -> Install:
    """The install to dump from: an explicit folder, or the first one found."""
    if game:
        p = check_game_dir(game)
        for inst in find_installs():
            if inst.path.resolve() == p.resolve():
                return inst
        return Install(p, "unknown", "given on the command line")
    found = find_installs()
    if not found:
        raise NotAnInstall("Rocket League was not found. Pass the game folder with --game "
                           "(the folder that holds Binaries and TAGame).")
    return found[0]
