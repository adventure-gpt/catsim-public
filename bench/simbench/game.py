"""Whole recorded games (levels C and D): the harness's side of the "game" request, and its score.

A recording (bench/recordings/*.npz) holds one real game from its first kickoff to its end:

  ops_C, ops_D   the round as a list of steps, in order (below), for each level the recording has: the arena and the
                 first kickoff's bodies, then the frames, with whatever the level gives in between
  frame_steps    per rendered frame, the physics ticks it ran (0, 1 or more)
  frame_time     per frame, its real duration in seconds (NaN: the tick's 1/120 s)
  frame_alpha    per frame, its render alpha (NaN: unchanged)
  frame_controls per frame, each car's 8 controls (throttle, steer, pitch, yaw, roll, jump, boost, handbrake)
  frame_judge    per frame, the compared packet row that follows it (-1: none)
  truth_*        the compared packet rows: per car [present, pos, vel, ang vel, rotator pitch / yaw / roll (units of
                 65536 a turn), boost], the ball [present, pos, vel, ang vel, rotator], the pads' active flags, the phase

The ops, in plain words (an adapter runs them in order; what it can't take, it says, and the level is "not supported"):

  ["arena", {cars: [{body, team}], map, countdown_ticks}]   a soccar arena with these car bodies (the game's product
                                                           ids), as built, before any kickoff
  ["start_at_drop", spots]          the first kickoff's bodies: each car's actor at [x, y, z, yaw units], ball in the middle
  ["restart", {spots, steps}]       a restart as the game makes one, `steps` ticks in (the first kickoff's bodies born)
  ["birth_delay", [ticks per car]]  cars whose bodies came later (players who joined one after another)
  ["next_handles", [ball, cars...]] the broadphase handles the next restart's bodies take
  ["physics_frame", n]              the physics frame counter
  ["ball_at_rest", asleep]          the ball at rest on its spot
  ["countdown", [length, default, shown, update_elapsed, start_elapsed, warmup, next_spawn_index]]  the game's own
                                    countdown clock as it stands
  ["match_seconds", s]              the match clock
  ["scripted", on] / ["pad_rule", on]   the game's own round flow / pad pickups on or off
  ["queue_kickoff", spots]          level C: where the next kickoff puts the cars (the game's random choice, given)
  ["queue_respawn", car, spot]      level C: where the car's next respawn goes (given)
  ["spawn_selector", {primary, secondary, next_spawn_index}]  level D: the kickoff spots the game chooses from
  ["random_stream", {crt_state, world_time, mesh, motion_fx, director_camera, timers}]  level D: the game's random
                                    number stream at the start, and what draws from it (the sim predicts every choice)
  ["round_gate", started, steps]    the coming frame starts the round, or a countdown is held that long
  ["frames", first, count]          run these frames
  ["save"] / ["restore"]            the end-of-match publication rule: a match that ends in a frame publishes the state
                                    before it
  ["judge", row]                    compare now (otherwise after the frame `frame_judge` names)

The reply: per compared row, what a packet would show before its rounding: per car [present, pos, vel, ang vel,
rotator, boost], the ball [present, pos, vel, ang vel, rotator], each pad's active flag.

Scored, as RLBot packets show the game: positions rounded to 0.01 uu, velocities to 0.01 uu/s and angular velocities to
0.0001 rad/s the game's way (then within 0.002 / 0.002 / 0.00002, float noise), rotations as the game's integer rotator
(equal), each body there or not as in the packet, every pad's state, each car's boost during countdowns (within 0.05).
A row is exact when all of that is; the first row that isn't is where the game and the simulator part.
"""
from __future__ import annotations

import json

import numpy as np

F32 = np.float32
CAR_W, BALL_W, PADS = 14, 13, 34


def width(nc: int) -> int:
    return nc * CAR_W + BALL_W + PADS


def load_meta(path) -> dict:
    with np.load(path) as z:
        return json.loads(z["meta"].tobytes())


def levels(meta: dict) -> list:
    return meta.get("levels") or [meta["level"]]


def load(path) -> dict:
    z = np.load(path)
    rec = {k: z[k] for k in z.files}
    rec["meta"] = json.loads(rec["meta"].tobytes())
    for k in [k for k in rec if k.startswith("ops")]:
        rec[k] = json.loads(rec[k].tobytes())
    return rec


def pack(rec: dict, level: str | None = None, float_rows: bool = False) -> tuple[dict, bytes]:
    """The "game" request. With `float_rows` (a float-precision recording), the payload also names, per frame, the
    float row that follows it (-1 none), and the reply carries [present, 18 Bullet words] per body for each float row
    after the packet view."""
    nc = int(rec["meta"]["cars"])
    n = len(rec["frame_steps"])
    level = level or levels(rec["meta"])[0]
    level = {"Cf": "C", "Df": "D"}.get(level, level)
    ops = rec.get(f"ops_{level}") or rec["ops"]
    req = {"op": "game", "level": level, "map": rec["meta"]["map"], "cars": nc, "frames": n,
           "judges": int(len(rec["truth_phase"])), "ops": ops, "loadouts": rec["meta"].get("loadouts")}
    payload = (rec["frame_steps"].astype(F32).tobytes() + rec["frame_time"].astype(F32).tobytes()
               + rec["frame_alpha"].astype(F32).tobytes() + rec["frame_judge"].astype(F32).tobytes()
               + np.ascontiguousarray(rec["frame_controls"], F32).tobytes())
    if float_rows and "float_frame" in rec:
        ff = np.full(n, -1.0, F32)
        ff[rec["float_frame"].astype(np.int64)] = np.arange(len(rec["float_frame"]), dtype=F32)
        req["float_rows"] = int(len(rec["float_frame"]))
        payload += ff.tobytes()
    return req, payload


def unpack(req: dict, payload: bytes) -> dict:
    """The worker's side: the frame arrays from a game request's payload."""
    n, nc = int(req["frames"]), int(req["cars"])
    f = np.frombuffer(payload, F32)
    out = {"steps": f[0:n].astype(np.int32), "time": f[n:2 * n], "alpha": f[2 * n:3 * n], "judge": f[3 * n:4 * n].astype(np.int32),
           "controls": f[4 * n:4 * n + n * nc * 8].reshape(n, nc, 8)}
    at = 4 * n + n * nc * 8
    out["float"] = f[at:at + n].astype(np.int32) if req.get("float_rows") else None
    return out


def score_float(rec: dict, fview: np.ndarray) -> dict:
    """Float rows: each body there or not as in the game and, when both have it, its 18 Bullet words bit-equal."""
    nc = int(rec["meta"]["cars"])
    R = len(rec["float_frame"])
    v = np.asarray(fview, F32).reshape(R, nc + 1, 19)
    sim_p, sim_w = v[..., 0] > 0.5, np.ascontiguousarray(v[..., 1:]).view(np.uint32)
    p = rec["float_present"] > 0
    w = np.ascontiguousarray(rec["float_body"], F32).view(np.uint32)
    off = (sim_p != p) | (p & sim_p & (sim_w != w).any(-1))
    bad = off.any(-1)
    judged = rec["float_phase"] != 4
    idx = np.nonzero(bad & judged)[0]
    steps = np.cumsum(rec["frame_steps"].astype(np.int64))
    first = None
    if len(idx):
        r = int(idx[0])
        tick = int(steps[int(rec["float_frame"][r])])
        first = {"row": r, "of": int(judged.sum()), "tick": tick, "seconds": round(tick / 120.0, 2),
                 "bodies": [("ball" if b == nc else f"car{b}") for b in np.nonzero(off[r])[0]],
                 "phase": PHASE.get(int(rec["float_phase"][r]), str(int(rec["float_phase"][r])))}
    return {"compared_rows": int(judged.sum()), "exact_rows": int((~bad & judged).sum()),
            "exact_pct": 100.0 * float((~bad & judged).sum()) / max(int(judged.sum()), 1), "exact_to_the_end": first is None,
            "first_drift": first, "body_rows": int((p & sim_p).sum()), "body_rows_exact": int((p & sim_p & ~off).sum()),
            "goal_rows": int((~judged).sum()), "goal_rows_exact": int((~bad & ~judged).sum())}


def q_pos(x):
    return np.floor(x.astype(F32) * F32(100.0) + F32(0.5)) * 0.01


def q_vel(v, scale=100.0):
    k = np.trunc(v.astype(F32) * F32(scale))
    return k / scale + np.sign(k) * 0.1 / scale


# RLBot's match phases, as the packets number them
PHASE = {0: "inactive", 1: "countdown", 2: "kickoff", 3: "active", 4: "goal scored", 5: "replay", 6: "paused", 7: "ended"}


def row_ticks(rec: dict) -> dict:
    """Each compared row's physics tick, counted from the recording's first frame."""
    ops = next((rec[k] for k in ("ops_C", "ops_D", "ops") if k in rec), [])
    steps = np.cumsum(rec["frame_steps"].astype(np.int64))
    out, now = {}, 0
    for op in ops:
        if op[0] == "frames":
            i0, n = int(op[1]), int(op[2])
            j = rec["frame_judge"][i0:i0 + n]
            for k in np.nonzero(j >= 0)[0]:
                out[int(j[k])] = int(steps[i0 + k])
            now = int(steps[i0 + n - 1])
        elif op[0] == "judge":
            out[int(op[1])] = now
    return out


def score(rec: dict, view: np.ndarray) -> dict:
    nc = int(rec["meta"]["cars"])
    J = len(rec["truth_phase"])
    view = view.reshape(J, width(nc))
    sc = view[:, :nc * CAR_W].reshape(J, nc, CAR_W)
    sb = view[:, nc * CAR_W:nc * CAR_W + BALL_W]
    sp = view[:, nc * CAR_W + BALL_W:] > 0.5
    tc, tb, tp, ph = rec["truth_cars"], rec["truth_ball"], rec["truth_pads"] > 0, rec["truth_phase"]
    present_t = tc[:, :, 0] > 0.5
    off = np.zeros((J, nc + 1), bool)
    off[:, :nc] = ((np.abs(q_pos(sc[:, :, 1:4]) - tc[:, :, 1:4]) > .002).any(-1) | (np.abs(q_vel(sc[:, :, 4:7]) - tc[:, :, 4:7]) > .002).any(-1)
                   | (np.abs(q_vel(sc[:, :, 7:10], 1e4) - tc[:, :, 7:10]) > 2e-5).any(-1))
    d = np.abs(sc[:, :, 10:13] - tc[:, :, 10:13])
    off[:, :nc] |= np.minimum(d, np.abs(d - 65536)).max(-1) > 0
    off[:, :nc] &= present_t
    ball_t = tb[:, 0] > 0.5
    off[:, nc] = ball_t & ((np.abs(q_pos(sb[:, 1:4]) - tb[:, 1:4]) > .002).any(-1) | (np.abs(q_vel(sb[:, 4:7]) - tb[:, 4:7]) > .002).any(-1)
                           | (np.abs(q_vel(sb[:, 7:10], 1e4) - tb[:, 7:10]) > 2e-5).any(-1))
    off[:, :nc] |= (sc[:, :, 0] > 0.5) != present_t
    off[:, nc] |= (sb[:, 0] > 0.5) != ball_t
    pads_off = (sp != tp).any(-1)
    boost_off = (ph == 1) & ((np.abs(sc[:, :, 13] - tc[:, :, 13]) > 0.05) & present_t).any(-1)
    has_rot = ~np.isnan(tb[:, 10])
    rot_ok = np.zeros(J, bool)
    rot_ok[has_rot] = ((sb[has_rot, 10:13].astype(np.int64) % 65536) == (tb[has_rot, 10:13].astype(np.int64) % 65536)).all(-1)
    bad = off.any(-1) | pads_off | boost_off
    judged = ph != 4                                  # a goal's replay rows are scored apart (the ball is gone)
    first = None
    idx = np.nonzero(bad & judged)[0]
    if len(idx):
        r = int(idx[0])
        bodies = [("ball" if b == nc else f"car{b}") for b in np.nonzero(off[r])[0]]
        # the physics tick of the drift, counted from the recording's first frame
        tick = row_ticks(rec).get(r)
        first = {"compared_row": int(judged[:r].sum()), "of": int(judged.sum()), "tick": tick,
                 "seconds": None if tick is None else round(tick / 120.0, 2), "bodies": bodies, "pads": bool(pads_off[r]),
                 "boost": bool(boost_off[r]), "phase": PHASE.get(int(ph[r]), str(int(ph[r])))}
    goal_rows = ~judged
    return {"compared_rows": int(judged.sum()), "exact_rows": int((~bad & judged).sum()),
            "exact_pct": 100.0 * float((~bad & judged).sum()) / max(int(judged.sum()), 1),
            "exact_to_the_end": first is None, "first_drift": first,
            "exact_rows_before_first_drift": int(judged[:idx[0]].sum()) if len(idx) else int(judged.sum()),
            "goal_rows": int(goal_rows.sum()), "goal_rows_exact": int((~bad & goal_rows).sum()),
            "ball_rotation_compared": int(has_rot.sum()), "ball_rotation_exact": int(rot_ok.sum()),
            "body_rows_off": [int(x) for x in off[judged].sum(0)]}
