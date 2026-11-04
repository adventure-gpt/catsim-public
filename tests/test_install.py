"""Finding the game: Steam and Epic layouts built in a temporary folder (no real install needed)."""
import json
from pathlib import Path

import pytest

from catsim.dump import install


def fake_game(root: Path) -> Path:
    (root / "TAGame" / "CookedPCConsole").mkdir(parents=True)
    (root / "TAGame" / "CookedPCConsole" / "Stadium_P.upk").write_bytes(b"\xc1\x83\x2a\x9e")
    (root / "Binaries" / "Win64").mkdir(parents=True)
    (root / "Binaries" / "Win64" / "RocketLeague.exe").write_bytes(b"MZ")
    return root


def test_check_game_dir_accepts_subfolders(tmp_path):
    game = fake_game(tmp_path / "rocketleague")
    assert install.check_game_dir(game) == game
    assert install.check_game_dir(game / "Binaries" / "Win64") == game
    assert install.check_game_dir(game / "TAGame" / "CookedPCConsole") == game


def test_check_game_dir_rejects(tmp_path):
    with pytest.raises(install.NotAnInstall):
        install.check_game_dir(tmp_path)
    game = fake_game(tmp_path / "rl")
    (game / "Binaries" / "Win64" / "RocketLeague.exe").unlink()
    with pytest.raises(install.NotAnInstall, match="RocketLeague.exe"):
        install.check_game_dir(game)


def test_steam_second_library(tmp_path):
    steam = tmp_path / "Steam"
    (steam / "steamapps").mkdir(parents=True)
    lib = tmp_path / "Games" / "SteamLibrary"
    game = fake_game(lib / "steamapps" / "common" / "rocketleague")
    (steam / "steamapps" / "libraryfolders.vdf").write_text(
        '"libraryfolders"\n{\n\t"0"\n\t{\n\t\t"path"\t\t"%s"\n\t}\n\t"1"\n\t{\n\t\t"path"\t\t"%s"\n\t}\n}\n'
        % (str(steam).replace("\\", "\\\\"), str(lib).replace("\\", "\\\\")))
    (lib / "steamapps" / "appmanifest_252950.acf").write_text(
        '"AppState"\n{\n\t"appid"\t\t"252950"\n\t"buildid"\t\t"123"\n\t"installdir"\t\t"rocketleague"\n}\n')
    found = install.find_steam([steam])
    assert [(f.path, f.edition, f.version) for f in found] == [(game, "steam", "123")]


def test_epic_manifest_and_launcher_list(tmp_path):
    epic = tmp_path / "ProgramData" / "Epic"
    manifests = epic / "EpicGamesLauncher" / "Data" / "Manifests"
    manifests.mkdir(parents=True)
    game = fake_game(tmp_path / "Epic Games" / "rocketleague")
    other = tmp_path / "Epic Games" / "SomethingElse"
    other.mkdir(parents=True)
    (manifests / "AAAA.item").write_text(json.dumps({"AppName": "Fortnite", "DisplayName": "Fortnite",
                                                     "InstallLocation": str(other)}))
    (manifests / "BBBB.item").write_text(json.dumps({"AppName": "Sugar", "DisplayName": "Rocket League",
                                                     "InstallLocation": str(game), "AppVersionString": "2.61"}),
                                         encoding="utf-8-sig")
    found = install.find_epic([epic], home=tmp_path)
    assert [(f.path, f.edition, f.version) for f in found] == [(game, "epic", "2.61")]

    # The launcher's own list alone is enough too.
    (manifests / "BBBB.item").unlink()
    (epic / "UnrealEngineLauncher").mkdir(parents=True)
    (epic / "UnrealEngineLauncher" / "LauncherInstalled.dat").write_text(json.dumps(
        {"InstallationList": [{"InstallLocation": str(game), "AppName": "Sugar", "AppVersion": "2.61-x"}]}))
    found = install.find_epic([epic], home=tmp_path)
    assert [(f.path, f.source) for f in found] == [(game, "Epic LauncherInstalled.dat")]


def test_epic_manifest_pointing_nowhere_is_skipped(tmp_path):
    epic = tmp_path / "Epic"
    m = epic / "EpicGamesLauncher" / "Data" / "Manifests"
    m.mkdir(parents=True)
    (m / "X.item").write_text(json.dumps({"AppName": "Sugar", "InstallLocation": str(tmp_path / "gone")}))
    (m / "Y.item").write_text("not json")
    assert install.find_epic([epic], home=tmp_path) == []


def test_epic_on_linux_via_legendary(tmp_path):
    game = fake_game(tmp_path / "Games" / "Heroic" / "rocketleague")
    cfg = tmp_path / ".config" / "heroic" / "legendaryConfig" / "legendary"
    cfg.mkdir(parents=True)
    (cfg / "installed.json").write_text(json.dumps({"Sugar": {"install_path": str(game), "version": "v1"}}))
    found = install.find_epic([], home=tmp_path)
    assert [(f.path, f.edition) for f in found] == [(game, "epic")]


def test_epic_inside_a_wine_prefix(tmp_path):
    game = fake_game(tmp_path / ".wine" / "drive_c" / "Program Files" / "Epic Games" / "rocketleague")
    m = tmp_path / ".wine" / "drive_c" / "ProgramData" / "Epic" / "EpicGamesLauncher" / "Data" / "Manifests"
    m.mkdir(parents=True)
    (m / "Z.item").write_text(json.dumps({"AppName": "Sugar", "InstallLocation": str(game)}))
    dirs = install.epic_data_dirs(env={}, home=tmp_path)
    assert [f.path for f in install.find_epic(dirs, home=tmp_path)] == [game]


def test_steam_roots_on_linux_layouts(tmp_path):
    (tmp_path / ".local" / "share" / "Steam").mkdir(parents=True)
    roots = install.steam_roots(env={}, home=tmp_path)
    assert roots == [tmp_path / ".local" / "share" / "Steam"]
