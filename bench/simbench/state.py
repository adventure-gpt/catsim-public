"""The simulator-neutral state the harness hands to adapters, and what each accuracy level gives.

A state is plain JSON: `{"cars": [car, ...], "ball": ball, "pads": [[x, y, cooldown], ...], "tick": n}`. Units are the
game's: positions uu, velocities uu/s, angular velocities rad/s, rotations as forward / right / up unit vectors, boost
0 to 100. Which keys a car carries depends on the level (below); an adapter sets every key it can and ignores the rest.

Level B "v3": the public state, RocketSim v3's whole `CarState` (the full-state suite's "public" rows): pos, fwd, right,
    up, vel, ang_vel, boost, on_ground, has_jumped, has_double_jumped, has_flipped, is_supersonic, is_demoed,
    demo_respawn_timer, flip_rel_torque, jump_ticks, flip_time, is_flipping, is_jumping, jump_hold_broken, air_time,
    air_time_since_jump, handbrake_val, supersonic_grace_timer, time_since_boosted, boosting_time, is_boosting,
    is_auto_flipping, auto_flip_timer, auto_flip_torque_scale, bump_cooldown_timer, bump_victim (car index, -1 none),
    last_extra_hit_tick (-1 none; ticks counted like `tick`), wheels_contact (4 bools, RocketSim's wheel order: front
    left, front right, back left, back right), prev_controls (8 numbers: the controls the car held last tick), and the
    ball's `asleep`.
Level B "packet" (a variant of B): only the fields an RLBot v5 packet lists, unrounded: pos, fwd, right, up, vel,
    ang_vel, boost, is_supersonic, is_demoed, demo_respawn_timer, has_jumped, has_double_jumped, has_flipped,
    air_state (0 on the ground, 1 jumping, 2 double jumping, 3 dodging, 4 in the air), dodge_timeout (seconds left to
    dodge, -1 none), dodge_elapsed, dodge_dir (forward, right), last_input (the 8 controls of the last tick); the
    ball's pos, rotation and velocities. Contacts, timers and gates are left for the simulator to guess.
Level "rlpr": an RLPR recording's tick record (RocketSim's rlpr_metric): the packet's fields with exact floats plus
    jump_time, flip_time, is_jumping, is_flipping, flip_rel_torque, double_jumped_or_flipped, has_flip, wheels_contact,
    wheel_normals, world_contact, world_contact_normal, prev_controls, boosting_time, is_boosting, handbrake_val.
Level A "full": everything the game holds, in the full-state suite's own field layout: `start["full"]` names byte ranges
    of the request's payload (`{"field": [offset, length]}`, the suite's `s.*` arrays). Only adapters that map that
    layout take it.

Every level also gives `next_controls` (per car, the 8 controls of the first tick to come): an adapter may use them to
guess what a car was holding when the level doesn't say (`prev_controls`).

A ball: pos, fwd, right, up, vel, ang_vel, asleep. Bodies come back as 18 float32 per body, cars in order then the
ball: pos, vel, ang_vel, forward, right, up, with pos and vel in the adapter's own `units` ("uu", or "bt" = uu / 50, as
the simulator holds them, so that no conversion rounds them). An absent body (a demolished car) is 18 zeros.
"""
from __future__ import annotations

PACKET_CAR = ("pos", "fwd", "right", "up", "vel", "ang_vel", "boost", "is_supersonic", "is_demoed", "demo_respawn_timer",
              "has_jumped", "has_double_jumped", "has_flipped", "air_state", "dodge_timeout", "dodge_elapsed", "dodge_dir",
              "last_input")
V3_CAR = ("pos", "fwd", "right", "up", "vel", "ang_vel", "boost", "on_ground", "has_jumped", "has_double_jumped",
          "has_flipped", "is_supersonic", "is_demoed", "demo_respawn_timer", "flip_rel_torque", "jump_ticks", "flip_time",
          "is_flipping", "is_jumping", "jump_hold_broken", "air_time", "air_time_since_jump", "handbrake_val",
          "supersonic_grace_timer", "time_since_boosted", "boosting_time", "is_boosting", "is_auto_flipping",
          "auto_flip_timer", "auto_flip_torque_scale", "bump_cooldown_timer", "bump_victim", "last_extra_hit_tick",
          "wheels_contact", "prev_controls")
BALL = ("pos", "fwd", "right", "up", "vel", "ang_vel", "asleep")

LEVELS = {"A": "full", "B": "v3", "Bp": "packet"}


def packet_of(c: dict) -> dict:
    """RocketSim v3 CarState fields -> what an RLBot v5 packet shows for that car."""
    on = bool(c.get("on_ground"))
    air = 0 if on else 1 if c.get("is_jumping") else 3 if c.get("is_flipping") else 4
    rel = c.get("flip_rel_torque", (0.0, 0.0, 0.0))
    can = c.get("has_jumped") and not (c.get("has_double_jumped") or c.get("has_flipped"))
    out = {k: c[k] for k in ("pos", "fwd", "right", "up", "vel", "ang_vel", "boost", "is_supersonic", "is_demoed",
                             "demo_respawn_timer", "has_jumped", "has_double_jumped", "has_flipped") if k in c}
    out.update(air_state=air, dodge_elapsed=float(c.get("flip_time", 0.0)) if c.get("is_flipping") else 0.0,
               dodge_dir=[float(rel[1]), float(-rel[0])] if c.get("is_flipping") else [0.0, 0.0],
               dodge_timeout=max(0.0, 1.25 - float(c.get("air_time_since_jump", 0.0))) if (can and not on) else -1.0,
               last_input=list(c.get("prev_controls", [0.0] * 8)))
    return out


# A tick's inputs per car: the 8 controls, then whether the car may drive (the game holds cars during countdowns),
# then the boost the game gave it from a pad on that tick (0 to 1; 0 none). Frame timing, when a benchmark gives it,
# comes per tick as [the render alpha of its frame, the frame's duration in s, the number of physics ticks in the frame
# on the frame's first tick (0 on its later ticks), and the recording's own two values for the tick: its render fraction
# (-1 on a frame's earlier ticks) and its frame start time (-1 when not recorded)]: the game runs 0, 1 or more ticks per
# rendered frame.
TICK_CAR = 10
TICK_FRAME = 5
BODY = 18

# RocketSim's standard car bodies by preset, in the order RocketSim v3 numbers them.
PRESETS = ("Octane", "Dominus", "Plank", "Breakout", "Hybrid", "Merc", "Psyclops")


def pick(car: dict, fields) -> dict:
    return {k: car[k] for k in fields if k in car}


def v3_from_packet(c: dict) -> dict:
    """A packet-level car as RocketSim v3 CarState fields (the kits' default guess; simbench_kit::v3_from_packet in Rust).
    Cars that already have v3 fields come back unchanged."""
    if "air_state" not in c:
        return c
    o = dict(c)
    air = int(c["air_state"])
    o.update(on_ground=air == 0, is_jumping=air == 1, is_flipping=air == 3)
    d = c.get("dodge_dir", [0.0, 0.0])
    if air == 3:
        o["flip_time"] = c.get("dodge_elapsed", 0.0)
        o["flip_rel_torque"] = [-d[1], d[0], 0.0]
    else:
        o["flip_time"] = 10.0      # not dodging: guess the last dodge (if any) ended long ago (its clock gates the next)
    if "last_input" in c:
        li = c["last_input"]
        o["prev_controls"] = li
        boosting = li[6] > 0.5 and c.get("boost", 0.0) > 0
        o.update(handbrake_val=1.0 if li[7] > 0.5 else 0.0, is_boosting=boosting, boosting_time=0.5 if boosting else 0.0,
                 time_since_boosted=0.0 if boosting else 0.5)
    if c.get("dodge_timeout", -1.0) >= 0:
        o["air_time_since_jump"] = 1.25 - c["dodge_timeout"]
    return o
