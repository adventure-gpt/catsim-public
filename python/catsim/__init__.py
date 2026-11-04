"""catSim: a bit-exact Rocket League physics simulator, with an API shaped like RocketSim's Python bindings.

    import catsim as rs                      # most RocketSim code runs with this one line changed
    rs.init("catsim_data")                   # the folder catSim's dumper wrote from your copy of the game
    arena = rs.Arena(rs.GameMode.SOCCAR)
    car = arena.add_car(rs.Team.BLUE, rs.CarConfig(rs.CarConfig.OCTANE))
    car.set_controls(rs.CarControls(throttle=1.0, boost=True))
    arena.step(120)
    print(car.get_state().pos, arena.ball.get_state().pos)

Differences from RocketSim are listed in docs/API.md.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

__all__ = ["init", "Arena", "Ball", "Car", "CarConfig", "CarControls", "CarState", "BallState", "BoostPad",
           "BoostPadState", "GameMode", "Team", "Vec", "RotMat", "Angle", "Loadout", "__version__"]
__version__ = "0.1.0"

_lib_mod = None
_data_dir: Path | None = None


def __getattr__(name):
    if name == "Loadout":                        # imported on first use, so `python -m catsim.draw` runs cleanly
        from .draw import Loadout
        return Loadout
    raise AttributeError(f"module 'catsim' has no attribute {name!r}")


def _lib():
    global _lib_mod
    if _lib_mod is None:
        import importlib
        _lib_mod = importlib.import_module("catsim._native")
    return _lib_mod


def init(data_dir: str | os.PathLike | None = None, silent: bool = False) -> None:
    """Load the game data. `data_dir`: the folder catSim's dumper wrote (default: $CATSIM_DATA, else ./catsim_data).
    RocketSim's `init(collision_meshes_path)` takes the same place in a script."""
    global _data_dir
    d = Path(data_dir or os.environ.get("CATSIM_DATA") or "catsim_data")
    if not d.is_dir():
        raise FileNotFoundError(f"{d} is not a folder: run `python -m catsim.dump --out {d} --keys KEYFILE` first")
    _lib().init(str(d))
    _data_dir = d
    if not silent:
        print(f"catSim: game data loaded from {d}")


def _need_init() -> Path:
    if _data_dir is None:
        init(silent=True)
    return _data_dir


# ---------------------------------------------------------------- small value types

class Vec:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def as_tuple(self):
        return (self.x, self.y, self.z)

    def as_numpy(self):
        import numpy as np
        return np.array(self.as_tuple(), dtype=np.float32)

    def __iter__(self):
        return iter(self.as_tuple())

    def __getitem__(self, i):
        return self.as_tuple()[i]

    def __add__(self, o):
        return Vec(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o):
        return Vec(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, k):
        return Vec(self.x * k, self.y * k, self.z * k)

    def dot(self, o):
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, o):
        return Vec(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)

    def length(self):
        return math.sqrt(self.dot(self))

    def __eq__(self, o):
        return isinstance(o, Vec) and self.as_tuple() == o.as_tuple()

    def __repr__(self):
        return f"Vec({self.x:g}, {self.y:g}, {self.z:g})"


class RotMat:
    """Forward, right and up axes (Unreal's frame: x forward, y right, z up)."""
    __slots__ = ("forward", "right", "up")

    def __init__(self, forward: Vec | None = None, right: Vec | None = None, up: Vec | None = None):
        self.forward = forward or Vec(1, 0, 0)
        self.right = right or Vec(0, 1, 0)
        self.up = up or Vec(0, 0, 1)

    @staticmethod
    def get_identity():
        return RotMat()

    def as_angle(self) -> "Angle":
        f, r, u = self.forward, self.right, self.up
        return Angle(math.atan2(f.y, f.x), math.atan2(f.z, math.hypot(f.x, f.y)), math.atan2(-r.z, u.z))

    def __repr__(self):
        return f"RotMat(forward={self.forward}, right={self.right}, up={self.up})"


@dataclass
class Angle:
    """Yaw, pitch, roll in radians (Unreal's convention)."""
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    def to_rot_mat(self) -> RotMat:
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        cr, sr = math.cos(self.roll), math.sin(self.roll)
        return RotMat(Vec(cp * cy, cp * sy, sp),
                      Vec(sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp),
                      Vec(-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp))


class GameMode(IntEnum):
    SOCCAR = 0


class Team(IntEnum):
    BLUE = 0
    ORANGE = 1


@dataclass
class CarControls:
    throttle: float = 0.0
    steer: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    jump: bool = False
    boost: bool = False
    handbrake: bool = False

    def row(self):
        return [self.throttle, self.steer, self.pitch, self.yaw, self.roll,
                1.0 if self.jump else 0.0, 1.0 if self.boost else 0.0, 1.0 if self.handbrake else 0.0]


class CarConfig:
    """A car's hitbox and wheels: one of the game's handling presets, or a car body by name or product id."""
    OCTANE = "Octane"
    DOMINUS = "MuscleCar"
    PLANK = "Darkcar"
    BREAKOUT = "LongCar"
    HYBRID = "Hybrid"
    MERC = "Merc"

    def __init__(self, preset: str | int = OCTANE):
        self.name = str(preset)

    @property
    def hitbox_size(self) -> Vec:
        _need_init()
        return Vec(*_lib().preset_hitbox(self.name)[0])

    @property
    def hitbox_pos_offset(self) -> Vec:
        _need_init()
        return Vec(*_lib().preset_hitbox(self.name)[1])

    def __repr__(self):
        return f"CarConfig({self.name!r})"


@dataclass
class BallState:
    pos: Vec = field(default_factory=lambda: Vec(0, 0, 93.15))
    vel: Vec = field(default_factory=Vec)
    ang_vel: Vec = field(default_factory=Vec)


@dataclass
class CarState:
    pos: Vec = field(default_factory=lambda: Vec(0, 0, 17))
    rot_mat: RotMat = field(default_factory=RotMat)
    vel: Vec = field(default_factory=Vec)
    ang_vel: Vec = field(default_factory=Vec)
    boost: float = 33.3
    is_on_ground: bool = True
    has_jumped: bool = False
    has_double_jumped: bool = False
    has_flipped: bool = False
    is_jumping: bool = False
    is_flipping: bool = False
    is_supersonic: bool = False
    is_demoed: bool = False
    demo_respawn_timer: float = 0.0
    handbrake_val: float = 0.0
    wheels_with_contact: int = 0
    flip_time: float = 0.0
    air_time: float = 0.0
    air_time_since_jump: float = 0.0
    time_spent_boosting: float = 0.0
    last_controls: CarControls = field(default_factory=CarControls)


@dataclass
class BoostPadState:
    is_active: bool = True
    cooldown: float = 0.0


class BoostPad:
    def __init__(self, arena: "Arena", index: int, pos: Vec, is_big: bool):
        self._arena, self.index, self.pos, self.is_big = arena, index, pos, is_big

    def get_pos(self) -> Vec:
        return self.pos

    def get_state(self) -> BoostPadState:
        row = self._arena._native_arena().arena_row()
        n = len(self._arena._pads)
        return BoostPadState(row[9 + self.index] > 0.5, row[9 + n + self.index])


# ---------------------------------------------------------------- the arena

class Ball:
    RADIUS = 91.25

    def __init__(self, arena: "Arena"):
        self._arena = arena

    def get_state(self) -> BallState:
        r = self._arena._native_arena().arena_row()
        return BallState(Vec(*r[0:3]), Vec(*r[3:6]), Vec(*r[6:9]))

    def set_state(self, s: BallState, sleep_at_rest: bool = False) -> None:
        self._arena._native_arena().set_ball([*s.pos, *s.vel, *s.ang_vel], sleep_at_rest)

    def get_radius(self) -> float:
        return self.RADIUS


class Car:
    def __init__(self, arena: "Arena", id: int, team: Team, config: CarConfig, loadout=None):
        self._arena, self.id, self.team, self.config = arena, id, Team(team), config
        self.loadout = loadout
        self.controls = CarControls()

    @property
    def _slot(self) -> int:
        return self._arena._cars.index(self)

    def set_controls(self, c: CarControls) -> None:
        self.controls = c

    def get_controls(self) -> CarControls:
        return self.controls

    def get_state(self) -> CarState:
        n = self._arena._native_arena()
        i = self._slot
        r, h = n.car_row(i), n.hidden_row(i)
        rot = RotMat(Vec(*h[0:3]), Vec(*h[3:6]), Vec(*h[6:9]))
        return CarState(pos=Vec(*r[2:5]), rot_mat=rot, vel=Vec(*r[5:8]), ang_vel=Vec(*r[8:11]), boost=r[17],
                        is_on_ground=r[18] > 0.5, has_jumped=r[19] > 0.5, has_double_jumped=r[20] > 0.5,
                        has_flipped=r[21] > 0.5, is_jumping=r[22] > 0.5, is_flipping=r[23] > 0.5,
                        is_supersonic=r[26] > 0.5, is_demoed=r[27] > 0.5, demo_respawn_timer=r[28],
                        handbrake_val=r[30], wheels_with_contact=int(r[32]), flip_time=h[9],
                        air_time=h[11], air_time_since_jump=h[12], time_spent_boosting=h[13],
                        last_controls=self.controls)

    def set_state(self, s: CarState) -> None:
        a = s.rot_mat.as_angle()
        self._arena._native_arena().set_car(self._slot, [*s.pos, *s.vel, *s.ang_vel, a.yaw, a.pitch, a.roll, s.boost])

    def demolish(self) -> None:
        raise NotImplementedError("catSim demolishes cars only through the game's own rules (a supersonic hit)")

    def __repr__(self):
        return f"Car(id={self.id}, team={self.team.name}, config={self.config})"


class Arena:
    """One soccar arena at 120 ticks a second.

    catSim builds its physics world in the game's order once its cars are known, so adding or removing a car
    rebuilds the world and starts a kickoff (RocketSim keeps the current state)."""

    def __init__(self, game_mode: GameMode = GameMode.SOCCAR, tick_rate: float = 120.0, round_flow: bool = False,
                 map: str = "Stadium_P"):
        if GameMode(game_mode) != GameMode.SOCCAR:
            raise ValueError("catSim simulates soccar only")
        if tick_rate != 120.0:
            raise ValueError("catSim runs at the game's own 120 ticks a second")
        self.game_mode = GameMode.SOCCAR
        self.tick_rate = 120.0
        self.tick_time = 1.0 / 120.0
        self.round_flow = round_flow
        self.map = map
        self.draw_model: dict | None = None
        self.draw_model_note = "not started"
        self._cars: list[Car] = []
        self._next_id = 1
        self._native = None
        self._ticks_before = 0
        self.ball = Ball(self)
        self._goal_cb = self._bump_cb = self._touch_cb = None
        self._data = _need_init()
        self._pads = [BoostPad(self, i, Vec(*p), big) for i, (p, big) in enumerate(_lib().pad_locations(map))]

    # -- building
    def _native_arena(self):
        if self._native is None:
            cars = [(c.config.name, int(c.team)) for c in self._cars]
            self._native = _lib().NativeArena(cars, str(self._data), self.round_flow, self.map)
        return self._native

    def _rebuild(self):
        if self._native is not None:
            self._ticks_before += self._native.tick_count
        self._native = None

    def add_car(self, team: Team, config: CarConfig | None = None, loadout=None) -> Car:
        """`loadout`: the car's items (a `catsim.Loadout`, or a dict with RLBot's names and product ids), for the
        random-draw model; its `car_id` picks the body when no `config` is given."""
        from .draw import Loadout
        if isinstance(loadout, dict):
            loadout = Loadout.from_dict(loadout)
        if config is None and loadout is not None:
            config = CarConfig(loadout.car_id)
        car = Car(self, self._next_id, team, config or CarConfig(), loadout)
        self._next_id += 1
        self._cars.append(car)
        self._rebuild()
        return car

    def remove_car(self, car: Car) -> None:
        self._cars.remove(car)
        self._rebuild()

    def get_cars(self) -> list[Car]:
        return list(self._cars)

    def get_car_from_id(self, id: int) -> Car | None:
        return next((c for c in self._cars if c.id == id), None)

    def get_boost_pads(self) -> list[BoostPad]:
        return list(self._pads)

    @property
    def tick_count(self) -> int:
        return self._ticks_before + (self._native.tick_count if self._native is not None else 0)

    @property
    def score(self) -> tuple[int, int]:
        return tuple(self._native_arena().score)

    # -- callbacks (RocketSim's shapes; `data` is passed back unchanged)
    def set_goal_score_callback(self, cb, data=None):
        self._goal_cb = (cb, data) if cb else None

    def set_car_bump_callback(self, cb, data=None):
        self._bump_cb = (cb, data) if cb else None

    def set_car_ball_collision_callback(self, cb, data=None):
        self._touch_cb = (cb, data) if cb else None

    # -- running
    def step(self, ticks: int = 1) -> None:
        n = self._native_arena()
        rows = [c.controls.row() for c in self._cars]
        if not (self._goal_cb or self._bump_cb or self._touch_cb):
            n.step(rows, ticks)
            return
        for _ in range(ticks):
            goal = n.step(rows, 1)
            ev = n.events()
            for i, e in enumerate(ev):
                if self._touch_cb and e[0] > 0.5:
                    self._touch_cb[0](self, self._cars[i], self._touch_cb[1])
                if self._bump_cb and e[3] > 0.5:
                    victim = next((self._cars[j] for j, f in enumerate(ev) if f[2] > 0.5), None)
                    self._bump_cb[0](self, self._cars[i], victim, e[1] > 0.5, self._bump_cb[1])
            if goal >= 0 and self._goal_cb:
                self._goal_cb[0](self, Team(goal), self._goal_cb[1])

    # -- replaying real games: the game's frame timing and its random-number stream
    def step_frame(self, steps: int, frame_time: float, render_alpha: float) -> int:
        """One frame of the game's frame loop: `steps` physics ticks (0 to 8) with the cars' controls, the frame's
        real duration (s) and its render alpha. Returns the team that scored, or -1."""
        return self._native_arena().step_frame([c.controls.row() for c in self._cars], steps, frame_time, render_alpha)

    def _loadouts(self) -> list:
        from .draw import Loadout
        bodies = {name.lower(): pid for pid, name, _ in _lib().body_names()}
        out = []
        for c in self._cars:
            if c.loadout is not None:
                out.append(c.loadout)
                continue
            pid = bodies.get(c.config.name.lower()) or (int(c.config.name) if c.config.name.isdigit() else None)
            if pid is None:
                from .draw import NotCovered
                raise NotCovered(f"car {c.id} is a handling preset, not a car body: give it a loadout")
            out.append(Loadout(car_id=pid))
        return out

    def start_game_rng(self, crt_state: int, world_time: float, motion_fx: bool = True, director_camera: bool = True,
                       extra_timers=()) -> dict | None:
        """Turn on the game's random-number stream (`crt_state`: the C runtime's rand() state, `world_time`: the
        game's clock in seconds) and the model of everything else that draws from it. The animated parts' timers
        come from the data folder's items.txt and the cars' loadouts; when the data doesn't cover the match they
        stay off (`draw_model_note` says why) and the rest of the model still runs. `extra_timers`: timers the
        model can't predict, if you know them, as (next due time, period, random extra period). Returns the parts'
        settings, or None."""
        from .draw import Items, NotCovered, settings
        import warnings
        try:
            self.draw_model = settings(Items.in_folder(self._data), self.map, self._loadouts())
            self.draw_model_note = "covered"
        except NotCovered as e:
            self.draw_model, self.draw_model_note = None, str(e)
            warnings.warn(f"catSim: the random-draw model runs without the animated parts' timers: {e}", stacklevel=2)
        self._native_arena().start_game_rng(int(crt_state) & 0xFFFFFFFF, float(world_time), self.draw_model, motion_fx,
                                            director_camera, [tuple(map(float, x)) for x in extra_timers])
        return self.draw_model

    @property
    def crt_state(self) -> int | None:
        """The game's random-number state, while the random-draw model runs."""
        return self._native_arena().crt_state

    def draw_parts(self) -> list[tuple[int, int]]:
        """The animated parts' timers in update order: (car index, or -1 for the map and -2 for the goal explosion;
        part 0 body, 1 wheel, 2 other part, 3 goal explosion, 4 map)."""
        return list(self._native_arena().draw_parts())

    def reset_kickoff(self, seed: int | None = None) -> None:
        """Cars to kickoff spots, ball to the middle. The spots are chosen the game's way; `seed` is accepted for
        RocketSim compatibility and ignored."""
        self._native_arena().reset_kickoff()

    def is_ball_probably_going_in(self, max_time: float = 2.0) -> bool:
        for row in self._native_arena().predict_ball(int(max_time * 120), 1):
            if abs(row[1]) > 5124.25:
                return True
        return False

    def ball_prediction(self, ticks: int = 360, every: int = 1) -> list[BallState]:
        """The ball's future path (cars left out), `ticks` ticks ahead, one state every `every` ticks."""
        return [BallState(Vec(*r[0:3]), Vec(*r[3:6]), Vec(*r[6:9])) for r in self._native_arena().predict_ball(ticks, every)]

    def clone(self) -> "Arena":
        """A new arena with the same cars and the same ball and car states (hidden timers start fresh)."""
        a = Arena(self.game_mode, self.tick_rate, self.round_flow, self.map)
        for c in self._cars:
            a.add_car(c.team, CarConfig(c.config.name), c.loadout).set_controls(c.controls)
        a.ball.set_state(self.ball.get_state())
        for mine, theirs in zip(self._cars, a._cars):
            st = mine.get_state()
            if not st.is_demoed:
                theirs.set_state(st)
        return a
