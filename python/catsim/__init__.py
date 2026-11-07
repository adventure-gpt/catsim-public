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
           "BoostPadState", "GameMode", "Team", "Vec", "RotMat", "Angle", "Loadout", "CarHidden", "Clock", "WheelContact",
           "BodyState", "RoundState", "ContactPoint", "Manifold", "ContactMemory", "__version__"]
__version__ = "0.1.3"

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


# ---------------------------------------------------------------- hidden state (what a packet doesn't show)

@dataclass
class Clock:
    """A game component (jump, dodge, boost, ...): on or off, and the seconds since it last changed."""
    active: bool = False
    time: float = 0.0


@dataclass
class WheelContact:
    """A wheel's contact: touching, the surface normal, and what it touches: "arena", "car" or "ball"."""
    contact: bool = False
    normal: Vec = field(default_factory=lambda: Vec(0, 0, 1))
    ground: str = "arena"


_GROUNDS = ("arena", "car", "ball")


@dataclass
class CarHidden:
    """Everything about a car beyond its body that the next ticks read: the game's flags, its components' clocks, the
    input it holds, its wheels' contacts and its hit gates. `Car.get_hidden()` / `set_hidden()`; docs/API.md says what
    each field is and how RocketSim v3's `CarState` fields map onto them. Times in seconds; frames are physics frames
    (`Arena.physics_frame`)."""
    driving: bool = True
    on_ground: bool = True
    jumped: bool = False
    double_jumped: bool = False
    supersonic: bool = False
    supersonic_time: float = 0.0
    jump: Clock = field(default_factory=Clock)
    jump_released: bool = False
    double_jump: Clock = field(default_factory=Clock)
    dodge: Clock = field(default_factory=Clock)
    dodge_torque: Vec = field(default_factory=Vec)
    dodge_direction: Vec = field(default_factory=Vec)
    auto_flip: Clock = field(default_factory=Clock)
    auto_flip_right: bool = False
    auto_flip_duration: float = 0.0
    boost: Clock = field(default_factory=Clock)
    boost_amount: float = 0.333
    air_control: Clock = field(default_factory=Clock)
    handbrake: float = 0.0
    held_input: CarControls = field(default_factory=lambda: CarControls())
    wheels: list = field(default_factory=lambda: [WheelContact() for _ in range(4)])
    world_contact: bool = False
    world_contact_normal: Vec = field(default_factory=Vec)
    bump_victim: int | None = None          # the index (in arena.get_cars()) of the car it last bumped
    bump_time: float = 0.0                  # when, in physics time (Arena.physics_frame / 120)
    ball_impact_frame: int = 0              # the frame of its last ball hit with the extra impulse
    ball_touch_frame: int = 0

    def to_row(self) -> list:
        b = lambda x: 1.0 if x else 0.0
        r = [b(self.driving), b(self.on_ground), b(self.jumped), b(self.double_jumped), b(self.supersonic),
             self.supersonic_time, b(self.jump.active), self.jump.time, b(self.jump_released), b(self.double_jump.active),
             self.double_jump.time, b(self.dodge.active), self.dodge.time, *self.dodge_torque, *self.dodge_direction,
             b(self.auto_flip.active), self.auto_flip.time, b(self.auto_flip_right), self.auto_flip_duration,
             b(self.boost.active), self.boost.time,
             self.boost_amount, b(self.air_control.active), self.air_control.time, self.handbrake, *self.held_input.row()]
        r += [b(w.contact) for w in self.wheels]
        for w in self.wheels:
            r += list(w.normal)
        r += [b(self.world_contact), *self.world_contact_normal]
        r += [-1.0 if self.bump_victim is None else float(self.bump_victim), self.bump_time, float(self.ball_impact_frame),
              float(self.ball_touch_frame)]
        r += [float(_GROUNDS.index(w.ground)) for w in self.wheels]
        return r

    @staticmethod
    def from_row(r) -> "CarHidden":
        b = lambda k: r[k] != 0.0
        v = lambda k: Vec(r[k], r[k + 1], r[k + 2])
        held = CarControls(r[29], r[30], r[31], r[32], r[33], r[34] > 0.5, r[35] > 0.5, r[36] > 0.5)
        return CarHidden(b(0), b(1), b(2), b(3), b(4), r[5], Clock(b(6), r[7]), b(8), Clock(b(9), r[10]), Clock(b(11), r[12]),
                         v(13), v(16), Clock(b(19), r[20]), b(21), r[22], Clock(b(23), r[24]), r[25], Clock(b(26), r[27]), r[28],
                         held, [WheelContact(b(37 + w), v(41 + 3 * w), _GROUNDS[int(r[61 + w])]) for w in range(4)], b(53), v(54),
                         None if r[57] < 0 else int(r[57]), r[58], int(r[59]), int(r[60]))


@dataclass
class BodyState:
    """A rigid body exactly as the physics holds it, in Bullet units (1 = 50 uu): no rounding anywhere, so setting a
    body you got back gives the same next ticks. `basis` is the rotation matrix by rows (9 numbers)."""
    origin: Vec = field(default_factory=Vec)
    basis: tuple = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    lin_vel: Vec = field(default_factory=Vec)
    ang_vel: Vec = field(default_factory=Vec)
    asleep: bool = False
    deactivation_time: float = 0.0

    def to_row(self) -> list:
        return [*self.origin, *self.basis, *self.lin_vel, *self.ang_vel, 1.0 if self.asleep else 0.0, self.deactivation_time]

    @staticmethod
    def from_row(r) -> "BodyState":
        return BodyState(Vec(r[0], r[1], r[2]), tuple(r[3:12]), Vec(r[12], r[13], r[14]), Vec(r[15], r[16], r[17]),
                         r[18] != 0.0, r[19])


@dataclass
class RoundState:
    """The round's hidden state. `phase`: 0 countdown, 1 playing, 2 after a goal, 3 ended; `phase_left`: countdown ticks
    or post-goal seconds left. `countdown_clock`: the game's own countdown when it runs: (seconds shown, start timer's
    elapsed seconds or -1, update timer's elapsed seconds or -1)."""
    physics_frame: int = 0
    phase: int = 1
    phase_left: float = 0.0
    countdown_clock: tuple | None = None
    ball_has_been_hit: bool = False
    ball_held: bool = False
    game_time: float = 300.0
    score: tuple = (0, 0)


class ContactPoint:
    """One cached contact point, every field exact (`fields()` names them; `row` holds them in that order). The usual
    ones as attributes: `local_a` (on the manifold's body0, in its frame), `local_b` (on body1, in its frame),
    `world_a`, `world_b`, `normal` (on body1, world axes) and `distance`, all in Bullet units."""
    __slots__ = ("row",)

    def __init__(self, row):
        self.row = [float(x) for x in row]

    @staticmethod
    def fields() -> list:
        return _lib().contact_point_fields()

    local_a = property(lambda s: Vec(*s.row[0:3]))
    local_b = property(lambda s: Vec(*s.row[3:6]))
    world_a = property(lambda s: Vec(*s.row[6:9]))
    world_b = property(lambda s: Vec(*s.row[9:12]))
    normal = property(lambda s: Vec(*s.row[12:15]))
    distance = property(lambda s: s.row[15])

    def __eq__(self, o):
        return isinstance(o, ContactPoint) and self.row == o.row

    def __repr__(self):
        return f"ContactPoint(local_a={self.local_a}, local_b={self.local_b}, normal={self.normal}, distance={self.distance})"


@dataclass
class Manifold:
    """A contact manifold: up to 4 cached points between `body0` and `body1` ("ball", "car<i>" with i the car's place in
    `get_cars()`, or "world<k>", `Arena.world_parts()`'s k-th). `owner` is "pair" (`index` into the pairs, `child` the
    compound children or -1) or "predictive" (`index` into the last step's continuous-collision contacts)."""
    owner: str
    index: int
    child: tuple
    body0: str
    body1: str
    points: list = field(default_factory=list)


@dataclass
class ContactMemory:
    """What a step reads from the last one besides the bodies: each moving body's broadphase box ((body, min, max), as the
    last bounds update left it), the broadphase's overlapping pairs in their order ((body0, body1, dispatched), body0 the
    lower broadphase handle) and the manifolds in the dispatcher's order."""
    boxes: list = field(default_factory=list)
    pairs: list = field(default_factory=list)
    manifolds: list = field(default_factory=list)


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
    hidden: CarHidden | None = None      # filled by get_state; set_state applies it when present


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

    def set_state(self, s: BoostPadState) -> None:
        """Taken with `cooldown` seconds left, or available."""
        cd = float(s.cooldown)
        self._arena._native_arena().set_pad(self.index, (not s.is_active) and cd > 0, cd, 0.0)

    def get_timer(self) -> tuple:
        """Exact: (taken, the timer's length, its elapsed seconds)."""
        return tuple(self._arena._native_arena().pad(self.index))

    def set_timer(self, taken: bool, length: float, elapsed: float) -> None:
        self._arena._native_arena().set_pad(self.index, bool(taken), float(length), float(elapsed))


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

    def get_body(self) -> "BodyState | None":
        """The ball's body exactly (Bullet units), or None while there is no ball (after a goal's explosion)."""
        r = self._arena._native_arena().ball_body()
        return None if r is None else BodyState.from_row(r)

    def set_body(self, b: "BodyState") -> None:
        """Write the ball's body exactly: no rounding, asleep as given."""
        self._arena._native_arena().set_ball_body(b.to_row())

    def set_held(self, held: bool) -> None:
        """Hold the ball still where it is, as the game does at a kickoff while every car is held."""
        self._arena._native_arena().set_ball_held(bool(held))


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
                        last_controls=self.controls, hidden=self.get_hidden())

    def set_state(self, s: CarState) -> None:
        """Position, rotation (through the game's integer rotator, as its own state setting does), velocities and boost
        (a whole number); then `s.hidden`, when present. For an exact body use `set_body`."""
        a = s.rot_mat.as_angle()
        n = self._arena._native_arena()
        n.set_car(self._slot, [*s.pos, *s.vel, *s.ang_vel, a.yaw, a.pitch, a.roll, s.boost])
        if s.hidden is not None:
            row = s.hidden.to_row()
            if abs(s.boost - s.hidden.boost_amount * 100.0) > 1e-3:      # the public boost was changed: it wins
                row[25] = n.car_hidden(self._slot)[25]
            n.set_car_hidden(self._slot, row)

    def get_hidden(self) -> "CarHidden | None":
        """The car's hidden state, or None while it is demolished."""
        r = self._arena._native_arena().car_hidden(self._slot)
        return None if r is None else CarHidden.from_row(r)

    def set_hidden(self, h: "CarHidden") -> None:
        """Set the hidden state. The held input goes through the game's input store (8-bit axes), with no jump press."""
        self._arena._native_arena().set_car_hidden(self._slot, h.to_row())

    def get_body(self) -> "BodyState | None":
        """The car's body exactly (Bullet units), or None while it is demolished."""
        r = self._arena._native_arena().car_body(self._slot)
        return None if r is None else BodyState.from_row(r)

    def set_body(self, b: "BodyState") -> None:
        """Write the car's body exactly (no rounding of the rotation). A demolished car gets its body back."""
        self._arena._native_arena().set_car_body(self._slot, b.to_row())

    def dodge_from_input(self, forward: float, right: float):
        """The dodge torque and direction the game gives this car for a dodge input (forward, right): (Vec, Vec), for
        `CarHidden.dodge_torque` and `dodge_direction`. RocketSim v3's `flip_rel_torque` (x, y) is (-right, forward)."""
        t, d = self._arena._native_arena().dodge_from_input(self._slot, float(forward), float(right))
        return Vec(*t), Vec(*d)

    def demolish_now(self, respawn_in: float = 3.0) -> None:
        """Take the car's body away as a demolition does; it respawns after `respawn_in` seconds (with round flow)."""
        self._arena._native_arena().set_car_demolished(self._slot, float(respawn_in))

    def set_driving(self, on: bool) -> None:
        """Whether the car takes input (the game turns it off during countdowns and after goals)."""
        self._arena._native_arena().set_driving(self._slot, bool(on))

    def add_boost(self, amount: float) -> None:
        """Add boost as a pickup does (0 to 100)."""
        self._arena._native_arena().add_boost(self._slot, float(amount) / 100.0)

    def queue_respawn(self, pos: "Vec", rotator: tuple) -> None:
        """Where this car's next respawn goes (uu; pitch, yaw, roll with 65536 a turn); otherwise the game's choice."""
        self._arena._native_arena().queue_respawn(self._slot, list(pos), [int(x) for x in rotator])

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

    def copy(self) -> "Arena":
        """An exact copy: bodies, contact caches, every timer. Stepping both with the same controls gives the same
        result bit for bit. (Callbacks are not copied.)"""
        a = Arena.__new__(Arena)
        a.__dict__.update(self.__dict__)
        a._cars = []
        for c in self._cars:
            twin = Car(a, c.id, c.team, c.config, c.loadout)
            twin.controls = c.controls
            a._cars.append(twin)
        a.ball = Ball(a)
        a._pads = [BoostPad(a, p.index, p.pos, p.is_big) for p in self._pads]
        a._goal_cb = a._bump_cb = a._touch_cb = None
        a._native = self._native_arena().copy()
        return a

    @property
    def physics_frame(self) -> int:
        """The physics frame number (one more each tick); `CarHidden`'s hit gates count in it."""
        return self._native_arena().physics_frame

    def get_round(self) -> "RoundState":
        r = self._native_arena().round_hidden()
        return RoundState(r[0], r[1], r[2], None if r[3] is None else tuple(r[3]), r[4], r[5], r[6], tuple(r[7]))

    def set_round(self, r: "RoundState") -> None:
        cc = r.countdown_clock
        cc = None if cc is None else (int(cc[0]), float(cc[1]), float(cc[2]))
        self._native_arena().set_round_hidden(int(r.physics_frame), int(r.phase), float(r.phase_left), cc,
                                              bool(r.ball_has_been_hit), bool(r.ball_held), float(r.game_time),
                                              [int(x) for x in r.score])

    def get_contacts(self) -> "ContactMemory":
        """The contact memory: pairs and manifolds with their cached points (docs/API.md, "Contact memory")."""
        boxes, pairs, mans = self._native_arena().contacts()
        return ContactMemory([(b, Vec(*lo), Vec(*hi)) for b, lo, hi in boxes], [tuple(p) for p in pairs],
                             [Manifold(o, i, (c0, c1), b0, b1, [ContactPoint(r) for r in pts]) for o, i, c0, c1, b0, b1, pts in mans])

    def set_contacts(self, m: "ContactMemory") -> None:
        """Replace the contact memory. The pairs should be the bodies whose broadphase boxes overlap now (their order is
        what this sets); each manifold is made by its pair's own collision algorithm, in the order given."""
        self._native_arena().set_contacts([(str(b), list(lo), list(hi)) for b, lo, hi in m.boxes],
                                          [(str(a), str(b), bool(d)) for a, b, d in m.pairs],
                                          [(x.owner, int(x.index), int(x.child[0]), int(x.child[1]), x.body0, x.body1,
                                            [p.row if isinstance(p, ContactPoint) else list(p) for p in x.points]) for x in m.manifolds])

    def body_order(self) -> list:
        """The moving bodies ("ball", "car<i>") in the collision world's object order. Each tick the islands are
        numbered in this order and the solver takes the manifolds sorted by island, so it decides which of a body's
        contacts is solved first. It has a history: a body that leaves (a demolished car) is replaced by the last one,
        and a new body goes on the end."""
        return list(self._native_arena().body_order())

    def set_body_order(self, order) -> None:
        """Put the moving bodies in this object order (`body_order()`'s layout, each of them once)."""
        self._native_arena().set_body_order([str(x) for x in order])

    def world_parts(self) -> list:
        """The arena's planes and meshes ("world<k>" is the k-th): (kind, broadphase handle)."""
        return [tuple(w) for w in self._native_arena().world_parts()]

    def contact_point(self, body0: str, body1: str, local_a, local_b, normal, distance: float) -> "ContactPoint":
        """A contact point as the game adds one, from the point on each body (in its frame), the normal on body1 and the
        distance (Bullet units); the friction, restitution and the rest are made the game's way."""
        return ContactPoint(self._native_arena().contact_point(body0, body1, list(local_a), list(local_b), list(normal), float(distance)))

    def use_game_countdown(self, on: bool = True) -> None:
        """With `round_flow`, run later kickoffs' countdowns on the game's own clock (a 1 s start, then 3, 2, 1)."""
        self._native_arena().use_game_countdown(bool(on))

    def set_pad_rule(self, on: bool) -> None:
        """The game's own boost pad pickups on (the default) or off (then pads change only by `BoostPad.set_state` and
        boost only by `Car.add_boost`)."""
        self._native_arena().set_pad_rule(bool(on))

    def queue_kickoff(self, spots) -> None:
        """Where the next kickoff puts the cars: one (Vec uu, (pitch, yaw, roll)) per car in order; otherwise the game's
        own choice."""
        self._native_arena().queue_kickoff([(list(p), [int(x) for x in r]) for p, r in spots])

    def clone(self) -> "Arena":
        """A new arena with the same cars, ball and car states, bodies exact and hidden state included (contact caches
        start empty: `copy()` keeps those too)."""

        a = Arena(self.game_mode, self.tick_rate, self.round_flow, self.map)
        for c in self._cars:
            a.add_car(c.team, CarConfig(c.config.name), c.loadout).set_controls(c.controls)
        a.ball.set_state(self.ball.get_state())
        for mine, theirs in zip(self._cars, a._cars):
            st = mine.get_state()
            if not st.is_demoed:
                theirs.set_state(st)
                theirs.set_body(mine.get_body())
        b = self.ball.get_body()
        if b is not None:
            a.ball.set_body(b)
        return a
