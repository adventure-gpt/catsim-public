"""The frame random-draw model: what a match draws from the game's random stream outside physics.

The game draws its random numbers from one stream (the C runtime's rand()). Physics uses it for kickoff spawn
choices; many other things draw from the same stream every frame. To replay a real game exactly, catSim models
those draws:

- every skeletal (animated) mesh component's level-of-detail timer: one draw when it is due, then it waits
  0.75 to 1.25 s (the car bodies, skeletal wheels, skeletal antennas and toppers, the map's animated parts, a
  goal explosion's animated part);
- effects when they start: a boost pickup 5 draws, a jump 6, a dodge 7, a goal explosion 7, a demolition 6;
- 2 draws at the end of every frame, and the camera's draws at each countdown;
- the respawn spot choice after a demolition;
- the order of all of these inside a frame.

This module works out the model's settings for a match from the data folder (`items.txt`, written by the
dumper from your own game) and the cars' loadouts. The draw counts and timings above are rules measured in the
running game; which parts each item adds is read from the item's package. A match is covered when every item
of every car is one the dumped data describes; otherwise `settings()` raises NotCovered with the reason, and the
model of the components stays off.

Known limit: a few things also draw from the stream and can't be predicted from the game's state: the online
services' timers, the online-data cache timer, the crowd-chant timer and the post-match save timers. When one of
them fires, the replay's random stream shifts. `Arena.start_game_rng(extra_timers=...)` takes such timers' phases
when you know them (for example from a start snapshot).
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

# Rules measured in the running game (frame ends counted from the event's frame, numbers drawn).
GOAL = (3, 7)          # a goal explosion's effects
DEMOLISH = (4, 6)      # a demolition's effects
RESPAWN = (2, 3)       # a respawned car's new body joins the update list at the 2nd frame end, its other parts at the 3rd
# A kickoff restart draws 2, plus 7 per car, in its own frame; each car's new body makes its first level-of-detail
# draw through the component model (measured with skeletal wheels: in the next frame).
RESET = (2, 7)


class NotCovered(Exception):
    """The dumped data doesn't describe this match's items, so the component model can't be turned on."""


@dataclass
class Loadout:
    """A car's items by product id, with RLBot's field names (0 = the default item). Colours don't matter here."""
    car_id: int = 23
    decal_id: int = 0
    wheels_id: int = 0
    boost_id: int = 0
    antenna_id: int = 0
    hat_id: int = 0
    paint_finish_id: int = 0
    custom_finish_id: int = 0
    engine_audio_id: int = 0
    trails_id: int = 0
    goal_explosion_id: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "Loadout":
        names = {f.name for f in fields(cls)}
        return cls(**{k: int(v) for k, v in d.items() if k in names})


SLOT_OF = {"car_id": "body", "decal_id": "decal", "wheels_id": "wheels", "boost_id": "boost", "antenna_id": "antenna",
           "hat_id": "topper", "paint_finish_id": "paint_finish", "custom_finish_id": "paint_finish",
           "engine_audio_id": "engine_audio", "trails_id": "trail", "goal_explosion_id": "goal_explosion"}


@dataclass
class Item:
    id: int
    slot: str
    package: str
    skeletal: int
    status: str


class Items:
    """items.txt: every product the dumper read, and the maps' skeletal components."""

    def __init__(self, path: str | Path):
        self.items: dict[int, Item] = {}
        self.maps: dict[str, int] = {}
        section = None
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line or line.startswith("catsim-items"):
                continue
            if line.startswith("["):
                section = line.strip("[]")
                continue
            w = line.split()
            if section == "items" and len(w) == 5:
                self.items[int(w[0])] = Item(int(w[0]), w[1], w[2], int(w[3]), w[4])
            elif section == "maps" and len(w) == 2:
                self.maps[w[0]] = int(w[1])

    @classmethod
    def in_folder(cls, data_dir: str | Path) -> "Items":
        p = Path(data_dir) / "items.txt"
        if not p.is_file():
            raise NotCovered(f"{p} not found: re-run the dumper without --skip-items")
        return cls(p)


def _item(items: Items, field: str, pid: int) -> Item | None:
    if pid == 0:
        return None                     # the default item: static parts only (measured with the default loadout)
    it = items.items.get(pid)
    want = SLOT_OF[field]
    if it is None:
        raise NotCovered(f"{field} {pid}: not in the dumped product list")
    if it.slot != want:
        raise NotCovered(f"{field} {pid} ({it.package}) is a {it.slot}, not a {want}")
    if it.status != "ok":
        raise NotCovered(f"{field} {pid} ({it.package}): {it.status.replace('_', ' ')}")
    return it


def car_parts(items: Items, lo: Loadout) -> tuple[int, int, bool]:
    """(skeletal wheels, other skeletal parts, skeletal goal explosion) for one car."""
    body = _item(items, "car_id", lo.car_id)
    if body is None or body.skeletal != 1:
        raise NotCovered(f"car body {lo.car_id}: expected one skeletal car mesh")
    wheels = _item(items, "wheels_id", lo.wheels_id)
    extras = 0
    for f in ("antenna_id", "hat_id"):
        it = _item(items, f, getattr(lo, f))
        extras += it.skeletal if it else 0
    for f in ("decal_id", "boost_id", "paint_finish_id", "custom_finish_id", "engine_audio_id", "trails_id"):
        _item(items, f, getattr(lo, f))
    goal = _item(items, "goal_explosion_id", lo.goal_explosion_id)
    return (wheels.skeletal if wheels else 0), extras, bool(goal and goal.skeletal)


def settings(items: Items, map_name: str, loadouts: list[Loadout]) -> dict:
    """The component model's settings for a match on `map_name` with these cars (in the order they joined)."""
    if map_name not in items.maps:
        raise NotCovered(f"map {map_name}: not in the dumped data")
    if not loadouts:
        raise NotCovered("no cars")
    parts = [car_parts(items, lo) for lo in loadouts]
    if len(set(parts)) != 1:
        raise NotCovered("the cars' items differ in their animated parts; the model takes one loadout shape for every car")
    wheels, extras, goal_mesh = parts[0]
    return {"wheels": [wheels], "extras": [extras], "map": [items.maps[map_name]], "goal_mesh": [int(goal_mesh)],
            "goal": list(GOAL), "demolish": list(DEMOLISH), "reset": list(RESET), "respawn": list(RESPAWN)}


def main(argv=None) -> int:
    """python -m catsim.draw --data catsim_data --map UF_Night_P --car car_id=23,wheels_id=4842,... [--car ...]"""
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(prog="python -m catsim.draw", description="The draw model's settings for a match.")
    ap.add_argument("--data", default="catsim_data")
    ap.add_argument("--map", default="Stadium_P")
    ap.add_argument("--car", action="append", default=[], help="one car's items: field=id,field=id (RLBot names)")
    a = ap.parse_args(argv)
    los = [Loadout.from_dict(dict(kv.split("=", 1) for kv in c.split(",") if kv)) for c in a.car] or [Loadout()]
    try:
        print(json.dumps(settings(Items.in_folder(a.data), a.map, los)))
    except NotCovered as e:
        print(f"not covered: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
