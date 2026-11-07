"""Level C for simulators without the game's match flow: the harness runs the flow, the simulator the physics.

A whole recorded game needs more than physics: countdowns, kickoffs, goals and restarts, demolitions and respawns. A
simulator whose adapter replays the round itself (the "game" request, simbench.game) does all of that and gets the
strict level C. Every other simulator gets this runner, through the requests every adapter has (new, set, step,
bodies). The harness supplies the flow, each piece an assist named in the results:

  - each kickoff: the cars and the ball are put where the recording has them at the countdown's first packet row (the
    packet's own rounding; velocities as the packet shows them, the ball at rest), and the cars are held (may-drive off)
    for as long as the recording's countdown lasts;
  - after a goal, the next kickoff the same way, when the recording has it (the simulator's own goal, if it has one,
    doesn't matter: the ball's rows during a goal's replay are not scored);
  - a respawn: the car is put where the recording's packet first shows it again;
  - a kickoff or a respawn puts a new car there (nothing carried over: jumps, flips, boosting, timers, wheel contacts)
    with the game's spawn boost (33.3; a packet shows 33.33);
  - a game recorded at float precision: the same, and at its first float row (a few ticks after the first countdown)
    and at each later kickoff's start the bodies become the game's own floats, word for word; the rest of each car's
    state stays what the simulator built from the same controls (a Bullet-units adapter takes the words themselves, a
    uu one the uu values its x 0.02 turns into those words, where they exist, and is compared in uu: the words x 50);
  - an adapter that takes them (caps "tick_frames") gets each tick's frame row: the frame's render alpha and
    duration (the game tests boost pads once per rendered frame, at its render fraction);
  - each rendered frame: its physics ticks (the frame's tick count at 120 Hz; a frame with no tick runs none), with the
    frame's controls; a jump held through a countdown is not a press when the cars go (the game's input is read all
    through the countdown, so the button is let go first, as in the game).

The simulator does what it models: physics, boost and pads, demolitions. Scored on bodies only (the adapters report
bodies, not boost or pads): each car's and the ball's position, velocity and angular velocity at the packet's rounding,
and the rotation within one unit of the packet's integer rotator (the harness converts the matrix itself).
"""
from __future__ import annotations

import math

import numpy as np

from .game import BALL_W, CAR_W, PADS, q_pos, q_vel, row_ticks
from .state import TICK_CAR, TICK_FRAME

F32 = np.float32

# The game's car bodies (public product ids, as RLBot names them) by their hitbox, in RocketSim's preset names.
PRESET_OF_BODY = {23: "Octane", 4906: "Octane", 5039: "Octane", 403: "Dominus", 9388: "Dominus", 22: "Breakout",
                  8669: "Breakout", 10697: "Breakout", 7901: "Breakout", 10698: "Breakout", 31: "Hybrid", 9427: "Hybrid",
                  7012: "Hybrid", 803: "Plank", 24: "Plank", 7052: "Plank", 30: "Merc", 4780: "Merc", 4782: "Psyclops"}

# The game's boost at a spawn (a kickoff or a respawn): its config's 0.333 of a full tank. A packet shows it rounded
# (33.33), so the harness hands the game's own amount instead.
SPAWN_BOOST = 33.3

# A spawned car is a new car: nothing from before (jumps, flips, boosting, timers, wheel contacts) carries over.
FRESH_CAR = {"on_ground": False, "wheels_contact": [False] * 4, "has_jumped": False, "has_double_jumped": False,
             "has_flipped": False, "is_jumping": False, "is_flipping": False, "is_auto_flipping": False,
             "is_supersonic": False, "is_boosting": False, "is_demoed": False, "jump_ticks": 0, "flip_time": 0.0,
             "air_time": 0.0, "air_time_since_jump": 0.0, "handbrake_val": 0.0, "supersonic_grace_timer": 0.0,
             "time_since_boosted": 0.0, "boosting_time": 0.0, "auto_flip_timer": 0.0, "bump_cooldown_timer": 0.0,
             "demo_respawn_timer": 0.0}

ASSISTS = ["the harness supplies only the match flow the simulator lacks: each kickoff's start (cars and ball where the "
           "recording's countdown starts them, as its packet shows them: positions to 0.01 uu, velocities to 0.01 uu/s; at "
           "float precision, the bodies then set to the game's own floats at the first float row and at each later kickoff's "
           "start; a new car with the game's spawn boost, 33.3) and the hold through the countdown, the "
           "restart after each goal when the recording has it, and each demolished car's respawn spot from the recording",
           "the simulator runs everything it models itself: physics, boost and pads, demolitions and their respawn timers, "
           "goals (its own goal, if it has one, changes nothing the harness does)",
           "each frame's physics ticks counted from the recording", "scored on bodies only (no boost or pads)"]


def rot_vectors(rot):
    """Unreal rotators (pitch, yaw, roll in 65536ths of a turn) -> forward and up unit vectors (float64)."""
    r = np.asarray(rot, np.float64) * (math.pi / 32768.0)
    cp, sp, cy, sy, cr, sr = np.cos(r[..., 0]), np.sin(r[..., 0]), np.cos(r[..., 1]), np.sin(r[..., 1]), np.cos(r[..., 2]), np.sin(r[..., 2])
    fwd = np.stack([cp * cy, cp * sy, sp], -1)
    up = np.stack([-cr * sp * cy - sr * sy, -cr * sp * sy + sr * cy, cr * cp], -1)
    return fwd, up


def rotator(fwd, right, up):
    """A rotation's forward / right / up -> the Unreal rotator, rounded to whole units."""
    pitch = math.atan2(fwd[2], math.hypot(fwd[0], fwd[1]))
    yaw = math.atan2(fwd[1], fwd[0])
    cp, sp, cy, sy = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    # roll from the up vector against the rotation with roll 0
    up0 = (-sp * cy, -sp * sy, cp)
    right0 = (-sy, cy, 0.0)
    roll = math.atan2(up[0] * right0[0] + up[1] * right0[1] + up[2] * right0[2], up[0] * up0[0] + up[1] * up0[1] + up[2] * up0[2])
    k = 32768.0 / math.pi
    return [round(pitch * k), round(yaw * k), round(roll * k)]


def _state_car(row):
    """A truth car row [present, pos, vel, ang vel, rotator, boost] -> a simbench state car at a spawn (a kickoff's
    start, a respawn): a new car with the game's spawn boost."""
    f, u = rot_vectors(row[10:13])
    right = np.cross(u, f)
    return {**FRESH_CAR, "pos": [float(x) for x in row[1:4]], "vel": [float(x) for x in row[4:7]],
            "ang_vel": [float(x) for x in row[7:10]], "fwd": f.tolist(), "right": right.tolist(), "up": u.tolist(),
            "boost": SPAWN_BOOST}


def uu_of_bt(bt) -> np.ndarray:
    """Bullet units -> the uu floats a uu simulator (RocketSim's x 0.02 into Bullet units) turns into the same Bullet
    words: x 50, then the nearest neighbour that survives, where one does (not every word has one)."""
    bt = np.asarray(bt, np.float32)
    k = np.float32(0.02)
    uu = (bt * np.float32(50.0)).astype(np.float32)
    bad = (uu * k).astype(np.float32) != bt
    for step in (1, -1, 2, -2, 3, -3):
        if not bad.any():
            break
        cand = uu.copy()
        for _ in range(abs(step)):
            cand = np.where(bad, np.nextafter(cand, np.float32(np.inf if step > 0 else -np.inf)), cand).astype(np.float32)
        ok = bad & ((cand * k).astype(np.float32) == bt)
        uu = np.where(ok, cand, uu)
        bad &= ~ok
    return uu


def _float_bodies(words, present, boost=None):
    """A float row's bodies (the game's own Bullet words: origin, basis by rows, linear and angular velocity) as a
    simbench state that sets them exactly: the words themselves (`body_bt`) for adapters in Bullet units, and uu values
    that RocketSim's conversion turns back into the same words for the others. Bodies only (everything else as the simulator has it), unless
    `boost` is given: then a whole start (a car the row doesn't have is demolished, a still car on its wheels)."""
    def body(w):
        w = np.asarray(w, np.float32)
        basis = w[3:12].reshape(3, 3)
        return {"pos": uu_of_bt(w[0:3]).astype(float).tolist(), "vel": uu_of_bt(w[12:15]).astype(float).tolist(),
                "ang_vel": w[15:18].astype(float).tolist(), "fwd": basis[:, 0].astype(float).tolist(),
                "right": basis[:, 1].astype(float).tolist(), "up": basis[:, 2].astype(float).tolist(),
                "body_bt": w.astype(float).tolist() + [0.0, 0.0]}
    cars = []
    for k in range(len(words) - 1):
        if not present[k]:
            cars.append(None if boost is None else {"is_demoed": True, "demo_respawn_timer": 3.0})
            continue
        c = body(words[k])
        if boost is not None:
            c["boost"] = float(boost[k])
            if float(np.abs(np.asarray(words[k][12:15], np.float32)).max()) * 50.0 <= 1.0:
                c.update({"on_ground": True, "wheels_contact": [True] * 4})
        cars.append(c)
    ball = body(words[-1])
    if boost is not None:
        ball["asleep"] = False
    return {"cars": cars, "ball": ball}


def _given_float(rec, i, units):
    """Float row i as the harness hands it over, in the simulator's units (what it reports holding those words)."""
    out = np.zeros((rec["float_body"].shape[1], 19), F32)
    out[:, 0] = rec["float_present"][i] > 0
    out[:, 1:] = rec["float_body"][i]
    if units == "uu":
        out[:, 1:4] = (out[:, 1:4] * F32(50.0)).astype(F32)
        out[:, 13:16] = (out[:, 13:16] * F32(50.0)).astype(F32)
    out[out[:, 0] == 0] = 0.0
    return out


def birth(rec: dict, data) -> dict | None:
    """The first kickoff's bodies as the game makes them: the recording's own start (its restart's spots, the ticks the
    restart runs before the cars are born, each car's birth delay) put through catSim's spawn rules from the dump (the
    same spot table and body placement catSim's own run uses). Per car its body at birth (the game's Bullet words, at
    rest) and the physics ticks after the restart it is born at; the ball at the restart. None when catSim or its data
    isn't there, or the recording has no restart."""
    ops = rec.get("ops_C") or rec.get("ops")
    head = ops[:next((i for i, o in enumerate(ops) if o[0] == "frames"), len(ops))]
    start = next((o[1] for o in head if o[0] == "restart"), None)
    if start is None or not data:
        return None
    try:
        import catsim
        catsim.init(str(data), silent=True)
        lib = catsim._lib()
    except Exception:  # noqa: BLE001
        return None
    cfg = next(o[1] for o in head if o[0] == "arena")
    spot = lambda s: ([float(s[0]), float(s[1]), float(s[2])], [0, int(s[3]), 0])
    a = lib.NativeArena([(str(c["body"]), int(c["team"])) for c in cfg["cars"]], str(data), True,
                        cfg.get("map", "Stadium_P"), False, int(cfg.get("countdown_ticks", 2)))
    for o in head:
        if o[0] == "start_at_drop":
            a.start_at_drop(None if o[1] is None else [spot(s) for s in o[1]])
        elif o[0] == "restart":
            # every car born at the restart, before any tick: its body as the game places it
            a.restart(None if o[1].get("spots") is None else [spot(s) for s in o[1]["spots"]], 0)
    nc = len(cfg["cars"])
    cars = [a.car_body(k) for k in range(nc)]
    ball = a.ball_body()
    if ball is None or any(c is None for c in cars):
        return None
    delay = next((list(o[1]) for o in head if o[0] == "birth_delay"), [0] * nc)
    spots = start.get("spots") or next((o[1] for o in head if o[0] == "start_at_drop"), None)
    return {"cars": [np.asarray(c[:18], F32) for c in cars], "ball": np.asarray(ball[:18], F32),
            "delay": [int(x) for x in delay], "steps": int(start.get("steps", 0)),
            "actor": [[float(s[0]), float(s[1]), float(s[2]), int(s[3])] for s in spots] if spots else None}


def _born(words, boost=SPAWN_BOOST):
    """A new car at its birth: its body words, at rest, nothing carried over."""
    c = _float_bodies(np.stack([words, words]), np.array([True, True]), np.full(2, boost))["cars"][0]
    c.update(FRESH_CAR)
    c["boost"] = boost
    c["vel"] = [0.0, 0.0, 0.0]
    return c


def _view_of(bodies, units, nc):
    """Adapter bodies (cars then the ball, 18 floats) -> one row of the scored view (simbench.game's layout)."""
    k = 50.0 if units == "bt" else 1.0
    out = np.full(nc * CAR_W + BALL_W + PADS, np.nan, np.float64)
    for b in range(nc + 1):
        x = bodies[b].astype(np.float64)
        present = bool(np.any(x != 0.0))
        at = b * CAR_W if b < nc else nc * CAR_W
        out[at] = 1.0 if present else 0.0
        if not present:
            continue
        out[at + 1:at + 4] = x[0:3] * k
        out[at + 4:at + 7] = x[3:6] * k
        out[at + 7:at + 10] = x[6:9]
        out[at + 10:at + 13] = rotator(x[9:12], x[12:15], x[15:18])
    return out


def _float_of(bodies, units, nc):
    """Adapter bodies -> [present, origin, basis by rows, linear velocity, angular velocity] in the simulator's own units
    (a uu simulator's position and velocity stay in uu: game.score_float compares them with the game's words in uu)."""
    out = np.zeros((nc + 1, 19), F32)
    for b in range(nc + 1):
        x = bodies[b]
        if not np.any(x != 0.0):
            continue
        out[b, 0] = 1.0
        out[b, 1:4] = x[0:3]
        out[b, 4:13] = np.stack([x[9:12], x[12:15], x[15:18]], -1).reshape(9)
        out[b, 13:16] = x[3:6]
        out[b, 16:19] = x[6:9]
    return out


def run(worker, rec: dict, units: str, fview: np.ndarray | None = None, births: dict | None = None) -> np.ndarray:
    """Run one recording through the harness's flow. Returns the view: per compared row, cars then the ball (NaN where
    the simulator reports nothing: boost, pads). `fview` (float rows x bodies x 19), when given, gets the bodies after
    each frame that has a float row. `births` (from `birth`): the first kickoff starts from the game's own spawn
    instead of the countdown's first packet row."""
    meta = rec["meta"]
    nc = int(meta["cars"])
    ops = rec.get("ops_C") or rec.get("ops")
    arena_op = next(o for o in ops if o[0] == "arena")
    cars = [{"preset": PRESET_OF_BODY.get(int(c["body"]), "Octane"), "body": int(c["body"]), "team": int(c["team"])}
            for c in arena_op[1]["cars"]]
    hdr, _ = worker.request({"op": "new", "cars": cars, "map": meta["map"]})
    arena = hdr["arena"]
    tc, tb, ph = rec["truth_cars"], rec["truth_ball"], rec["truth_phase"]
    J = len(ph)
    view = np.full((J, nc * CAR_W + BALL_W + PADS), np.nan, np.float64)
    judge = rec["frame_judge"].astype(np.int64)
    steps = rec["frame_steps"].astype(np.int64)
    ctl = np.asarray(rec["frame_controls"], F32)
    # the kickoffs: the first row of each countdown; the respawns: a car back after rows without it, outside a countdown
    kick = {r for r in range(J) if ph[r] == 1 and (r == 0 or ph[r - 1] != 1)}
    back = {}
    for k in range(nc):
        p = tc[:, k, 0] > 0.5
        for r in np.nonzero(p[1:] & ~p[:-1])[0] + 1:
            if int(r) not in kick:
                back.setdefault(int(r), []).append(k)
    # the next compared row's phase for every frame (whether the cars may drive in it)
    nxt = np.full(len(judge), -1, np.int64)
    after = -1
    for f in range(len(judge) - 1, -1, -1):
        if judge[f] >= 0:
            after = judge[f]
        nxt[f] = after
    started = False
    last = None
    first_kick = min(kick) if kick else -1
    release = next((r for r in range(J) if ph[r] != 1), J)
    if births is not None:
        # the first countdown's cars come from their births, not from the packet rows
        back = {r: v for r, v in back.items() if r >= release}
    frow = np.full(len(judge), -1, np.int64)
    if fview is not None:
        frow[rec["float_frame"].astype(np.int64)] = np.arange(len(rec["float_frame"]))
    latch = np.zeros(nc, bool)             # jump held since a countdown: not a press until let go
    # an adapter that takes each tick's frame row (the suite's layout): the frame's render alpha (its last tick's render
    # fraction, the one before when the recording has none) and duration; the earlier ticks of a frame of several get
    # -1 as their fraction (no render frame ends there: the game tests boost pads once a frame)
    tick_frames = bool((getattr(worker, "hello", None) or {}).get("caps", {}).get("tick_frames"))
    alpha_s = rec.get("frame_alpha")
    ftime = rec.get("frame_time")
    alpha = 0.5

    def frame_rows(f, n):
        nonlocal alpha
        a = float(alpha_s[f]) if alpha_s is not None else float("nan")
        if math.isfinite(a):
            alpha = a
        dt = float(ftime[f]) if ftime is not None and math.isfinite(float(ftime[f])) and float(ftime[f]) > 0 else n / 120.0
        rows = np.zeros((n, TICK_FRAME), F32)
        rows[:, 0], rows[:, 1], rows[0, 2] = alpha, dt, n
        rows[:, 3], rows[-1, 3], rows[:, 4] = -1.0, alpha, -1.0
        return rows
    # a float-precision game: the match runs from its first kickoff as at packet precision, and at its first float row
    # the bodies become the game's own floats, word for word (the rest of each car's state is what the simulator built
    # through the countdown, from the same controls)
    float_start = int(np.nonzero(frow >= 0)[0][0]) if fview is not None and (frow >= 0).any() else -1

    def to_floats(f):
        i = int(frow[f])
        whole = not started
        jr = int(nxt[f]) if nxt[f] >= 0 else 0
        boost = (np.full(nc, SPAWN_BOOST) if ph[jr] == 1 else tc[jr, :, 13]) if whole else None
        worker.request({"op": "set", "arena": arena,
                        "state": _float_bodies(rec["float_body"][i], rec["float_present"][i] > 0, boost)})
        _, data = worker.request({"op": "bodies", "arena": arena})
        return np.frombuffer(data, F32).reshape(nc + 1, 18)
    def place(r):
        # where the recording's countdown starts: everything put there instead of run
        st = {"cars": [_state_car(tc[r, k]) if tc[r, k, 0] > 0.5 else {"is_demoed": True, "demo_respawn_timer": 3.0}
                       for k in range(nc)],
              "ball": {"pos": [float(x) for x in tb[r, 1:4]], "vel": [0.0, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0],
                       "fwd": [1.0, 0.0, 0.0], "right": [0.0, 1.0, 0.0], "up": [0.0, 0.0, 1.0], "asleep": False}}
        worker.request({"op": "set", "arena": arena, "state": st})
        _, data = worker.request({"op": "bodies", "arena": arena})
        return np.frombuffer(data, F32).reshape(nc + 1, 18)
    # a countdown's first row that the recording compares before any frame ends there (a recording that starts with
    # it): put there before the first frame after it
    framed = set(int(x) for x in judge if x >= 0)
    early = {}
    for r0 in sorted(kick - framed - ({first_kick} if births is not None else set())):
        later = np.nonzero(judge > r0)[0]
        if len(later):
            early.setdefault(int(later[0]), []).append(r0)
    unborn = []
    left = []
    if births is not None:
        # the restart: the ball where the game puts it, no car yet; its own ticks; then the cars born by then
        hold = {"is_demoed": True, "demo_respawn_timer": 1000.0}
        r0 = max(first_kick, 0)
        st = {"cars": [{**_born(births["cars"][k]), **hold} for k in range(nc)],
              # the ball held on its spot through the countdown, where the packet shows it (its body rests there)
              "ball": {"pos": [float(x) for x in tb[r0, 1:4]], "vel": [0.0, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0],
                       "fwd": [1.0, 0.0, 0.0], "right": [0.0, 1.0, 0.0], "up": [0.0, 0.0, 1.0], "asleep": False}}
        worker.request({"op": "set", "arena": arena, "state": st})
        unborn = list(range(nc))
        left = [d - births["steps"] for d in births["delay"]]
        if births["steps"] > 0:
            inp = np.zeros((nc, TICK_CAR), F32)
            worker.request({"op": "step", "arena": arena, "cars": nc, "ticks": births["steps"]}, inp.tobytes())
        started = True

    just_born = []

    def born_now():
        # the cars whose birth has come (the game makes them at a frame's end, after its ticks)
        nonlocal unborn, just_born
        now = [k for k in unborn if left[k] <= 0]
        if now:
            worker.request({"op": "set", "arena": arena, "state": {
                "cars": [_born(births["cars"][k]) if k in now else None for k in range(nc)], "ball": None}})
            unborn = [k for k in unborn if k not in now]
        just_born = now
        _, data = worker.request({"op": "bodies", "arena": arena})
        return np.frombuffer(data, F32).reshape(nc + 1, 18)

    def seen(bodies):
        # the packet after a car's birth frame shows its spawn actor (the body's place reaches the actor with the
        # next physics tick): the harness's own spawn, as the recording's spot gives it
        v = _view_of(bodies, units, nc)
        if births is not None and births.get("actor"):
            for k in just_born:
                x, y, z, yaw = births["actor"][k]
                at = k * CAR_W
                v[at + 1:at + 4] = [x, y, z]
                v[at + 4:at + 10] = 0.0
                v[at + 10:at + 13] = [0, yaw, 0]
        return v
    # the rows the recording compares between frames: after how many frames each comes
    judged_at = {}
    nframes = 0
    for op in ops:
        if op[0] == "frames":
            nframes = int(op[1]) + int(op[2])
        elif op[0] == "judge":
            judged_at.setdefault(nframes, []).append(int(op[1]))
    if births is not None:
        last = born_now()
        if first_kick not in framed and first_kick >= 0:
            view[first_kick] = seen(last)
    for f in range(len(judge)):
        r = int(judge[f])
        for rj in judged_at.get(f, ()):
            if last is not None and started:
                view[rj] = seen(last)
        just_born = []                     # a birth shows as the spawn actor only in the frame it happens in
        for r0 in early.get(f, ()):
            last = place(r0)
            started = True
            view[r0] = _view_of(last, units, nc)
        if f == float_start and not started:
            # a float row before any kickoff: the simulator starts there
            last = to_floats(f)
            started = True
            latch = ctl[f][:, 5] > 0.5     # a jump held in the frame before the start is not a new press
            fview[frow[f]] = _given_float(rec, int(frow[f]), units)
            if r >= 0:
                view[r] = _view_of(last, units, nc)
            continue
        if r in kick and not (births is not None and r == first_kick):
            # this frame ends where the recording's countdown starts: put everything there instead of running it
            last = place(r)
            started = True
            if fview is not None and frow[f] >= 0:
                # at float precision, a later kickoff's start is the game's own floats (the first one: see to_floats)
                last = to_floats(f)
                fview[frow[f]] = _given_float(rec, int(frow[f]), units)
            view[r] = _view_of(last, units, nc)
            continue
        if not started:
            continue
        if steps[f] > 0:
            drive = 0.0 if (nxt[f] >= 0 and ph[nxt[f]] == 1) else 1.0
            inp = np.zeros((nc, TICK_CAR), F32)
            inp[:, :8] = ctl[f]
            inp[:, 8] = drive
            held = ctl[f][:, 5] > 0.5
            if drive == 0.0:
                latch |= held
            latch &= held
            inp[latch, 5] = 0.0
            req = {"op": "step", "arena": arena, "cars": nc, "ticks": int(steps[f])}
            body = inp.tobytes()
            if tick_frames:
                req["tick_frames"] = True
                body += frame_rows(f, int(steps[f])).tobytes()
            _, data = worker.request(req, body)
            last = np.frombuffer(data, F32).reshape(nc + 1, 18)
        if unborn:
            left = [x - int(steps[f]) for x in left]
            last = born_now()
        if births is not None and r == first_kick:
            # the ball where the countdown's first packet row shows it, held there as at packet precision
            worker.request({"op": "set", "arena": arena, "state": {"cars": [None] * nc, "ball": {
                "pos": [float(x) for x in tb[r, 1:4]], "vel": [0.0, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0],
                "fwd": [1.0, 0.0, 0.0], "right": [0.0, 1.0, 0.0], "up": [0.0, 0.0, 1.0], "asleep": False}}})
            _, data = worker.request({"op": "bodies", "arena": arena})
            last = np.frombuffer(data, F32).reshape(nc + 1, 18)
        if f == float_start:
            last = to_floats(f)
        if r >= 0:
            if r in back and last is not None:
                # the respawned cars only: everything else (the ball too) stays as the simulator has it
                st = {"cars": [_state_car(tc[r, k]) if k in back[r] else None for k in range(nc)], "ball": None}
                worker.request({"op": "set", "arena": arena, "state": st})
                _, data = worker.request({"op": "bodies", "arena": arena})
                last = np.frombuffer(data, F32).reshape(nc + 1, 18)
            if last is not None:
                view[r] = seen(last)
        if fview is not None and frow[f] >= 0 and last is not None:
            # the start row is the harness's own input (as a kickoff's start row is at packet precision)
            fview[frow[f]] = _given_float(rec, int(frow[f]), units) if f == float_start else _float_of(last, units, nc)
    # a compared row the recording judges after its last frame (the match's end)
    for rj in judged_at.get(len(judge), ()):
        if last is not None:
            view[rj] = seen(last)
    worker.request({"op": "close", "arena": arena})
    return view


def score(rec: dict, view: np.ndarray) -> dict:
    """Bodies only: each body there or not as the packet says, position / velocity / angular velocity at the packet's
    rounding, the rotation within one unit."""
    nc = int(rec["meta"]["cars"])
    J = len(rec["truth_phase"])
    sc = view[:, :nc * CAR_W].reshape(J, nc, CAR_W)
    sb = view[:, nc * CAR_W:nc * CAR_W + BALL_W]
    tc, tb, ph = rec["truth_cars"], rec["truth_ball"], rec["truth_phase"]
    seen = ~np.isnan(view[:, 0])
    off = np.zeros((J, nc + 1), bool)

    def body_off(s, t):
        bad = (np.abs(q_pos(np.nan_to_num(s[..., 1:4])) - t[..., 1:4]) > .002).any(-1)
        bad |= (np.abs(q_vel(np.nan_to_num(s[..., 4:7])) - t[..., 4:7]) > .002).any(-1)
        bad |= (np.abs(q_vel(np.nan_to_num(s[..., 7:10]), 1e4) - t[..., 7:10]) > 2e-5).any(-1)
        d = np.abs(np.nan_to_num(s[..., 10:13]) - t[..., 10:13])
        bad |= np.minimum(d, np.abs(d - 65536)).max(-1) > 1
        return bad
    pt = tc[:, :, 0] > 0.5
    off[:, :nc] = (body_off(sc, tc) & pt) | ((np.nan_to_num(sc[:, :, 0]) > 0.5) != pt)
    pb = tb[:, 0] > 0.5
    rot_known = ~np.isnan(tb[:, 10])
    tb2 = np.where(np.isnan(tb), 0.0, tb)
    sb2 = sb.copy()
    sb2[~rot_known, 10:13] = 0.0
    off[:, nc] = (body_off(sb2, tb2) & pb) | ((np.nan_to_num(sb[:, 0]) > 0.5) != pb)
    judged = (ph != 4) & seen
    bad = off.any(-1)
    idx = np.nonzero(bad & judged)[0]
    ticks = row_ticks(rec)
    first = None
    if len(idx):
        r = int(idx[0])
        tick = ticks.get(r)
        first = {"compared_row": int(judged[:r].sum()), "of": int(judged.sum()), "tick": tick,
                 "seconds": None if tick is None else round(tick / 120.0, 2),
                 "bodies": [("ball" if b == nc else f"car{b}") for b in np.nonzero(off[r])[0]], "pads": False, "boost": False,
                 "phase": {1: "countdown", 2: "kickoff", 3: "active"}.get(int(ph[r]), str(int(ph[r])))}
    return {"compared_rows": int(judged.sum()), "exact_rows": int((~bad & judged).sum()),
            "exact_pct": 100.0 * float((~bad & judged).sum()) / max(int(judged.sum()), 1),
            "exact_to_the_end": first is None, "first_drift": first, "bodies_only": True}


HORIZONS = (1.0, 5.0, 30.0)


def kickoff_errors(rec: dict, view: np.ndarray) -> dict:
    """The RocketGPU full-state suite's normalized error (per car: the square root of the sum of squares of the car's and the
    ball's terms, position / 2 uu, velocity / 3 uu/s, angular velocity / 0.05 rad/s, forward / 0.05, up / 0.05; capped at
    10) 1, 5 and 30 s after each kickoff (the countdown's end), averaged over the cars and the kickoffs that last that long
    without a goal. Returns {seconds: [sum, count]}."""
    nc = int(rec["meta"]["cars"])
    J = len(rec["truth_phase"])
    ph = rec["truth_phase"]
    tc, tb = rec["truth_cars"].astype(np.float64), rec["truth_ball"].astype(np.float64)
    sc = view[:, :nc * CAR_W].reshape(J, nc, CAR_W)
    sb = view[:, nc * CAR_W:nc * CAR_W + BALL_W]
    ticks = row_ticks(rec)
    order = sorted((t, r) for r, t in ticks.items() if t is not None)
    t_of = np.full(J, -1, np.int64)
    for t, r in order:
        t_of[r] = t
    starts = [r for r in range(1, J) if ph[r] == 2 and ph[r - 1] == 1]
    out = {str(h): [0.0, 0] for h in HORIZONS}

    def terms(s, t):
        if np.isnan(s[1]) or np.isnan(t[1]):
            return None
        fs, us = rot_vectors(s[10:13])
        ft, ut = rot_vectors(t[10:13])
        return [np.linalg.norm(s[1:4] - t[1:4]) / 2.0, np.linalg.norm(s[4:7] - t[4:7]) / 3.0, np.linalg.norm(s[7:10] - t[7:10]) / 0.05,
                np.linalg.norm(fs - ft) / 0.05, np.linalg.norm(us - ut) / 0.05]
    for r0 in starts:
        t0 = t_of[r0]
        if t0 < 0:
            continue
        for h in HORIZONS:
            want = t0 + int(round(h * 120))
            r = r0
            while r < J and (t_of[r] < want or t_of[r] < 0):
                if ph[r] in (4, 1) and r > r0:
                    r = J
                    break
                r += 1
            if r >= J or ph[r] in (4, 1) or np.isnan(view[r, 0]):
                continue
            bt = terms(sb[r], tb[r]) if tb[r, 0] > 0.5 and not np.isnan(tb[r, 10]) else None
            for k in range(nc):
                if tc[r, k, 0] < 0.5:
                    continue
                ct = terms(sc[r, k], tc[r, k]) if sc[r, k, 0] > 0.5 else None
                e = 10.0 if ct is None else math.sqrt(sum(x * x for x in ct + (bt or [])))
                out[str(h)][0] += min(e, 10.0)
                out[str(h)][1] += 1
    return out
