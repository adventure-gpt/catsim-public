# The game-data dumper

catSim ships no data from the game. `catsim-dump` (or `python -m catsim.dump`) makes it from your own copy.

```
catsim-dump --list                                     # which installs it can see
catsim-dump --out catsim_data --keys keys.txt          # find the game, check it, write catsim_data/ (about 3 minutes)
catsim-dump --out catsim_data --keys keys.txt --skip-items                  # about 10 s, without the car items
catsim-dump --out catsim_data --keys keys.txt --game "D:\Games\rocketleague"  # say where the game is
```

## What it does, and what it never does

- It reads files on disk only: the packages under `TAGame\CookedPCConsole` and the executable's version resource.
  It never starts the game, never reads a running game's memory, and never writes to the game's folder.
- It checks the game's version before reading anything else, and stops with a clear message on a version nobody
  has checked (`--allow-unknown-version` tries anyway; the structural checks below still apply).
- It checks the structure of everything it reads and refuses to write anything that doesn't fit, rather than guess.

## Finding the game

In this order, the first one found wins (`--list` shows them all):

| Edition | Where it looks |
|---|---|
| Steam, Windows | Steam's registry key (`SteamPath` / `InstallPath`), then every library in `steamapps\libraryfolders.vdf`, each checked for `appmanifest_252950.acf` |
| Steam, Linux (including Proton) | `~/.steam/steam`, `~/.local/share/Steam`, the Flatpak and Snap Steam folders, and their libraries the same way |
| Epic, Windows | `C:\ProgramData\Epic\EpicGamesLauncher\Data\Manifests\*.item` (the one whose `AppName` is `Sugar` or whose name is Rocket League: its `InstallLocation`), then `C:\ProgramData\Epic\UnrealEngineLauncher\LauncherInstalled.dat` |
| Epic, Linux | Heroic's and legendary's `installed.json`, and Epic's manifests inside Wine prefixes (`~/.wine`, `~/Games/*`, Lutris) |
| Any | `--game FOLDER` or the `CATSIM_RL_DIR` environment variable: the folder holding `Binaries` and `TAGame` (any folder inside it works too) |

## The key file

The game's packages keep their tables encrypted (AES-256). The keys are not stored as plain bytes in the game's
files, and catSim does not include them. Use the `keys.txt` from
[CrunchyRL/RLUPKTools](https://github.com/CrunchyRL/RLUPKTools), the Rocket League modding community's package
tools (also listed on the [Rocket League Modding Wiki](https://rocketleaguemoddingwiki.github.io/pages/software.html)). Its format: one key per line,
base64 or hex, `#` for comments. The dumper tries each key and uses the one that decrypts the package's chunk
table correctly; a wrong key file fails cleanly.

`--keys FILE`, or set `CATSIM_RL_KEYS=FILE`.

Newer items may need keys your file doesn't have. Those items are listed in `items.txt` as not readable, and a
match that uses one runs without the animated parts' timers (see below); everything else is unaffected.

## What it writes

```
catsim_data/
  catsim_data.json        what was dumped, from which install and version, and each file's sha256
  soccar/arena.txt        the 16 arena pieces: mesh name, mirror, placement
  soccar/mesh_0.cmf ...   the pieces' triangle meshes (physics units, the game's vertex and triangle order)
  tables.txt              car presets, steer curves, bodies, pads, goals, planes, pickup boxes
  items.txt               every car item's animated parts, and the maps' (for the random-draw model)
```

These files come from your copy of the game: keep them to yourself.

### Where each piece comes from

| Data | Packages |
|---|---|
| Arena meshes, planes, goals, pads | `Stadium_P` (and `UF_Night_P` for its pad heights; the dumper checks it has the same meshes and Stadium_P's last four planes) |
| Car presets and steer curves | `Startup`, `TAGame`, `GameInfo_Soccar_SF`, and presets kept in body packages |
| Car bodies | the 19 body packages (`Body_*_SF`, `body_*_SF`) |
| Car items | `TAGame`'s product database, then each item's own package (about 7,900 products) |

docs/PROVENANCE.md lists every value, how it is read, and the few documented rules where the files don't say.

### The car items and the random-draw model

The game draws random numbers outside physics too, from the same stream that decides kickoff spawn choices:
every animated (skeletal) mesh in the world draws one about once a second. Which items are animated decides how
many draws a frame makes, so the dumper reads each item: the body's mesh, whether the wheels' meshes are
skeletal, each antenna's and topper's attachments (and an antenna's stick), the goal explosion's effects actor,
and the animated parts placed in the two maps.

Each line of `items.txt` says how many animated parts the item adds, and `ok`, or why the model has no rule for
it (for example an attachment with a particle effect). `python -m catsim.draw` shows the settings for a match:

```
python -m catsim.draw --data catsim_data --map UF_Night_P --car car_id=23,wheels_id=4842,antenna_id=12019,hat_id=12697,goal_explosion_id=12660
{"wheels": [4], "extras": [1], "map": [5], "goal_mesh": [1], "goal": [3, 7], "demolish": [4, 6], "reset": [2, 7], "respawn": [2, 3]}
```

`Arena.start_game_rng` does this for you (docs/API.md).

## Versions checked

| Game build | Edition | Result |
|---|---|---|
| Steam build 25914621 (executable 1.0.10897.0), Windows 11 | Steam | Every table and mesh bit-exact against the development engine (two corrected values, docs/PROVENANCE.md) |
| Steam build 25641912 (executable 1.0.10897.0), Windows 11 | Steam | Tables and arena meshes identical to the above |
| Epic build ++Prime+Update60.1-CL-530866 (executable 1.0.10897.0), Windows 11 | Epic | Found by `--list` through the launcher's install manifest; all 18 data files (arena meshes and every table) byte-identical to the Steam dump of the same engine version |

To check another Epic install: run `catsim-dump --list` (it should show an `epic` line), then
`catsim-dump --out epic_data --keys keys.txt --allow-unknown-version`, and compare `epic_data/catsim_data.json`'s
file hashes with a Steam dump of the same game version.
