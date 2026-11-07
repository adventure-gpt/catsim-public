"""catSim through its Python package (`pip install catsim-rl`): the same engine as the native `catsim` worker, one
arena at a time through the Python binding. For speed the native worker is catSim's fast path; this adapter shows what
the Python API alone gives (and is what RocketSim-style Python scripts get)."""
from __future__ import annotations

import math
import os

import numpy as np

from simbench.kit import Adapter
from simbench.state import v3_from_packet

# RocketSim's preset names -> catSim's car bodies (by product id) or presets
PRESET = {"Octane": "Octane", "Dominus": "MuscleCar", "Plank": "Darkcar", "Breakout": "LongCar", "Hybrid": "Hybrid",
          "Merc": "Merc", "Psyclops": "Psyclops"}
WHEEL_FROM_V3 = (1, 0, 3, 2)      # catSim's wheel w is RocketSim's wheel WHEEL_FROM_V3[w] (left and right swapped)
F32 = np.float32


def f32(x) -> float:
    return float(F32(x))


def basis_rows(fwd, right, up) -> list:
    return [fwd[0], right[0], up[0], fwd[1], right[1], up[1], fwd[2], right[2], up[2]]


class CatsimPy(Adapter):
    name = "catsim-py"
    device = "cpu"
    units = "bt"
    caps = {"levels": ["packet", "v3", "rlpr"], "frame_timing": True, "game_flow": True, "rng_stream": True, "spot_choices": True}
    choices = [
        "bodies written exactly as given (position x 0.02, velocity x 0.02, the rotation matrix as is)",
        "sets every hidden field the start gives: jump, flip, boost and supersonic clocks, the dodge torque from "
        "flip_rel_torque, wheel contacts, held controls, bump and ball-hit gates, demolitions",
        "can't set: contact manifolds' cached points and the broadphase pair order (catSim makes them at the first tick)",
        "pad pickups from the recording (the game's own pad rule off), cars held where the recording holds them",
        "whole games: catSim's own round flow (countdowns, goals, restarts, pickups, demolitions), the "
        "recording's frames as the game ran them; C takes the kickoff and respawn spots given, D its own choices from "
        "the random stream's start (the draw model from the data folder's items)",
    ]

    def setup(self, env):
        import catsim
        self.rs = catsim
        self.lib = catsim._lib()
        data = env.get("catsim_data") or os.environ.get("CATSIM_DATA")
        catsim.init(data, silent=True)
        self.data = data
        self.version = f"catsim-rl {catsim.__version__}"
        self.about = "catSim's Python package, one arena at a time"

    def new_arena(self, cars, map="Stadium_P"):
        names = [(str(c["body"]) if c.get("body") else PRESET[c["preset"]], int(c["team"])) for c in cars]
        a = self.lib.NativeArena(names, str(self.data), False, map)
        a.set_pad_rule(False)
        return {"a": a, "n": len(cars), "rows": None}

    # ------------------------------------------------------------ state in

    def set_state(self, arena, st):
        a, n = arena["a"], arena["n"]
        tick = int(st.get("tick", 0))
        frame = a.physics_frame
        for i, c in enumerate(st["cars"]):
            if c is None:
                continue
            c = v3_from_packet(c)
            if c.get("is_demoed"):
                a.set_car_demolished(i, f32(c.get("demo_respawn_timer", 3.0)))
                continue
            if "body_bt" in c:
                a.set_car_body(i, list(c["body_bt"]))
            else:
                p, v = c["pos"], c["vel"]
                a.set_car_body(i, [f32(F32(p[0]) * F32(0.02)), f32(F32(p[1]) * F32(0.02)), f32(F32(p[2]) * F32(0.02)),
                                   *basis_rows(c["fwd"], c["right"], c["up"]),
                                   f32(F32(v[0]) * F32(0.02)), f32(F32(v[1]) * F32(0.02)), f32(F32(v[2]) * F32(0.02)),
                                   *c["ang_vel"], 0.0, 0.0])
            h = list(a.car_hidden(i))
            self._hidden(a, i, h, c, tick, frame)
            if "prev_controls" not in c and "next_controls" in st:
                h[29:37] = [float(x) for x in st["next_controls"][i]]      # guess: already holding the first tick's
            a.set_car_hidden(i, h)
        b = st["ball"]
        if b is None:                   # a ball given as null stays as it is
            pass
        elif "body_bt" in b:
            a.set_ball_body(list(b["body_bt"]))
        else:
            p, v = b["pos"], b["vel"]
            asleep = bool(b.get("asleep", False))
            a.set_ball_body([f32(F32(p[0]) * F32(0.02)), f32(F32(p[1]) * F32(0.02)), f32(F32(p[2]) * F32(0.02)),
                             *basis_rows(b.get("fwd", (1, 0, 0)), b.get("right", (0, 1, 0)), b.get("up", (0, 0, 1))),
                             f32(F32(v[0]) * F32(0.02)), f32(F32(v[1]) * F32(0.02)), f32(F32(v[2]) * F32(0.02)),
                             *b["ang_vel"], 1.0 if asleep else 0.0, 0.0])
        for x, y, cd in st.get("pads", []):
            i = self._pad_index(x, y)
            a.set_pad(i, cd > 0, float(cd), 0.0)

    def _pad_index(self, x, y):
        if not hasattr(self, "_pads"):
            self._pads = self.lib.pad_locations("Stadium_P")
        return min(range(len(self._pads)), key=lambda k: (self._pads[k][0][0] - x) ** 2 + (self._pads[k][0][1] - y) ** 2)

    def _hidden(self, a, i, h, c, tick, frame):
        """RocketSim v3 / RLPR field meanings -> catSim's (docs/API.md, 'RocketSim v3's CarState in catSim')."""
        B = lambda k, d=False: 1.0 if c.get(k, d) else 0.0
        if "on_ground" in c:
            h[1] = B("on_ground")
        if "has_jumped" in c:       # the game's own jump flag: set by a jump, a double jump and a dodge alike
            h[2] = 1.0 if (c.get("has_jumped") or c.get("has_double_jumped") or c.get("has_flipped")) else 0.0
        if "has_double_jumped" in c or "has_flipped" in c:
            h[3] = 1.0 if (c.get("has_double_jumped") or c.get("has_flipped")) else 0.0
        if "double_jumped_or_flipped" in c:
            h[3] = 1.0 if (c["double_jumped_or_flipped"] and not c.get("has_flip", False)) else 0.0
        if "is_supersonic" in c:
            h[4] = B("is_supersonic")
        if "supersonic_grace_timer" in c:
            h[5] = f32(c["supersonic_grace_timer"])
        if "is_jumping" in c:
            h[6] = B("is_jumping")
            if "jump_time" in c:
                h[7] = f32(c["jump_time"])
            elif c.get("is_jumping"):
                h[7] = f32(c.get("jump_ticks", 0) / 120.0)
            elif "air_time_since_jump" in c:
                h[7] = f32(c["air_time_since_jump"])
        if "jump_hold_broken" in c:
            h[8] = B("jump_hold_broken")
        if "is_flipping" in c:
            h[11] = B("is_flipping")
        if "flip_time" in c:
            h[12] = f32(c["flip_time"])
        if "flip_rel_torque" in c:
            rx, ry = c["flip_rel_torque"][0], c["flip_rel_torque"][1]
            if rx or ry:
                t, d = a.dodge_from_input(i, f32(ry), f32(-rx))
                h[13:16], h[16:19] = list(t), list(d)
        if "is_auto_flipping" in c:
            h[19] = B("is_auto_flipping")
            if c.get("is_auto_flipping"):
                left = f32(c.get("auto_flip_timer", 0.0))
                h[20], h[22] = 1.0 / 120.0, f32(left + 1.0 / 120.0)
                h[21] = 1.0 if c.get("auto_flip_torque_scale", 1.0) > 0 else 0.0
        if "is_boosting" in c:
            h[23] = B("is_boosting")
            h[24] = f32(c.get("boosting_time", 0.0) if c.get("is_boosting") else c.get("time_since_boosted", 0.0))
        if "boost" in c:
            h[25] = f32(F32(c["boost"]) / F32(100.0))
        if "handbrake_val" in c:
            h[28] = f32(c["handbrake_val"])
        if "prev_controls" in c:
            h[29:37] = [float(x) for x in c["prev_controls"]]
        up = (0.0, 0.0, 1.0)      # a contact normal the level doesn't give: straight up (measured best on B2 starts)
        if "wheels_contact" in c:
            for w in range(4):
                h[37 + w] = 1.0 if c["wheels_contact"][WHEEL_FROM_V3[w]] else 0.0
                normal = c["wheel_normals"][WHEEL_FROM_V3[w]] if "wheel_normals" in c else up
                h[41 + 3 * w:44 + 3 * w] = [float(x) for x in normal]
        elif "on_ground" in c:     # a packet doesn't show the wheels: all four touching when the car is on the ground
            for w in range(4):
                h[37 + w] = h[1]
                h[41 + 3 * w:44 + 3 * w] = list(up)
        if "world_contact" in c:
            h[53] = B("world_contact")
            h[54:57] = [float(x) for x in c.get("world_contact_normal", (0, 0, 0))]
        if "bump_cooldown_timer" in c:
            left = float(c["bump_cooldown_timer"])
            v = int(c.get("bump_victim", -1))
            h[57] = float(v) if v >= 0 and left > 0 else -1.0
            h[58] = f32(F32(1.0 / 120.0) * F32(frame) - F32(0.25 - left)) if left > 0 else -10.0
        if "last_extra_hit_tick" in c:
            hit = int(c["last_extra_hit_tick"])
            h[59] = float(frame - (tick - hit)) if hit >= 0 else -100000.0

    # ------------------------------------------------------------ stepping

    def _drive(self, arena, inputs):
        a, n = arena["a"], arena["n"]
        anyone = False
        for i in range(n):
            drive = bool(inputs[i, 8] >= 0.5)
            anyone |= drive
            a.set_driving(i, drive)
        a.set_ball_held(n > 0 and not anyone)

    def _credit(self, arena, inputs):
        """The game's pad credits: a small pad adds 0.12 (the float the game adds), a big one fills the car."""
        a = arena["a"]
        for i in range(arena["n"]):
            g = float(inputs[i, 9])
            if g <= 0:
                continue
            m = round(g / 0.12)
            if 1 <= m <= 4 and abs(g - 0.12 * m) < 1e-3:
                for _ in range(m):
                    a.add_boost(i, f32(0.12))
            else:
                a.add_boost(i, 1.0)

    def step(self, arena, inputs, ticks=1, frame=None):
        self._drive(arena, inputs)
        arena["a"].step(inputs[:, :8].tolist(), ticks)
        self._credit(arena, inputs)

    def step_frame(self, arena, ticks, frame_time, alpha):
        """A game frame of several ticks: the bodies after each tick from exact copies run that far, then the frame."""
        a = arena["a"]
        self._drive(arena, ticks[0])
        rows = ticks[0][:, :8].tolist()
        alpha = min(max(float(alpha), 0.0), 1.0)
        out = []
        for j in range(1, len(ticks)):
            c = {"a": a.copy(), "n": arena["n"]}
            c["a"].step_frame(rows, j, float(frame_time), alpha)
            out.append(self.bodies(c))
        a.step_frame(rows, len(ticks), float(frame_time), alpha)
        out.append(self.bodies(arena))
        for t in ticks:
            self._credit(arena, t)
        return np.stack(out)

    # ------------------------------------------------------------ whole recorded games (levels C and D)

    def game(self, req, fr):
        """Run a whole recorded game's ops (simbench.game) on catSim's own round flow; return what each compared packet
        row would show. Every op is a catSim public API call (docs/API.md, 'Replaying a recording from its start')."""
        lib = self.lib
        nc = int(req["cars"])
        width = nc * 14 + 13 + 34
        out = np.zeros((int(req["judges"]), width), np.float32)
        a = saved = None
        spot = lambda s: ([float(s[0]), float(s[1]), float(s[2])], [0, int(s[3]), 0])
        steps, ftime, alpha, judge, ctl = fr["steps"], fr["time"], fr["alpha"], fr["judge"], fr["controls"]
        frow = fr.get("float")
        fout = np.zeros((int(req.get("float_rows") or 0), nc + 1, 19), np.float32)

        def float_row(r):
            # each body exactly as the physics holds it: present, origin, basis by rows, linear and angular velocity
            for b in range(nc + 1):
                x = a.car_body(b) if b < nc else a.ball_body()
                if x is not None:
                    fout[r, b, 0] = 1.0
                    fout[r, b, 1:] = np.asarray(x[:18], np.float32)
        for op in req["ops"]:
            name = op[0]
            if name == "arena":
                cfg = op[1]
                a = lib.NativeArena([(str(c["body"]), int(c["team"])) for c in cfg["cars"]], str(self.data), True,
                                    cfg.get("map", "Stadium_P"), False, int(cfg.get("countdown_ticks", 2)))
            elif name == "start_at_drop":
                a.start_at_drop(None if op[1] is None else [spot(s) for s in op[1]])
            elif name == "restart":
                sp = op[1].get("spots")
                a.restart(None if sp is None else [spot(s) for s in sp], int(op[1].get("steps", 0)))
            elif name == "scripted":
                a.set_scripted(bool(op[1]))
            elif name == "pad_rule":
                a.set_pad_rule(bool(op[1]))
            elif name == "next_handles":
                a.set_next_handles([int(x) for x in op[1]])
            elif name == "birth_delay":
                a.set_birth_delay([int(x) for x in op[1]])
            elif name == "physics_frame":
                a.physics_frame = int(op[1])
            elif name == "ball_at_rest":
                a.set_ball_at_rest(bool(op[1]))
            elif name == "countdown":
                v = op[1]
                a.set_countdown(int(v[0]), int(v[1]), int(v[2]), float(v[3]), float(v[4]), bool(v[5]), int(v[6]))
            elif name == "match_seconds":
                a.set_match_seconds(float(op[1]))
            elif name == "queue_kickoff":
                a.queue_kickoff([spot(s) for s in op[1]])
            elif name == "queue_respawn":
                loc, rot = spot(op[2])
                a.queue_respawn(int(op[1]), loc, rot)
            elif name == "spawn_selector":
                conv = lambda team: [([r[0], r[1], r[2]], [int(r[3]), int(r[4]), int(r[5])]) for r in team]
                s = op[1]
                a.set_spawn_selector([conv(t) for t in s["primary"]], [conv(t) for t in s["secondary"]], int(s["next_spawn_index"]))
            elif name == "random_stream":
                s = op[1]
                a.start_game_rng(int(s["crt_state"]), float(s["world_time"]), s.get("mesh"), bool(s.get("motion_fx", True)),
                                 bool(s.get("director_camera", True)), [tuple(map(float, t[:3])) for t in s.get("timers", [])])
            elif name == "round_gate":
                a.round_gate(bool(op[1]), int(op[2]))
            elif name == "crt_state":
                raise RuntimeError("this recording sets the random state frame by frame, which catSim's public API doesn't take")
            elif name == "save":
                saved = a.copy()
            elif name == "restore":
                a = saved
            elif name == "judge":
                out[int(op[1])] = a.packet_view()
            elif name == "frames":
                for k in range(int(op[1]), int(op[1]) + int(op[2])):
                    rows = ctl[k].tolist()
                    n = int(steps[k])
                    if np.isnan(ftime[k]) and n == 1:
                        a.step(rows, 1)                          # one tick of 1/120 s (a frame of its own)
                    elif np.isnan(ftime[k]):
                        a.step_frame(rows, n, float(np.float32(1.0 / 120.0) * np.float32(n)), 0.0)
                    else:
                        al = float(alpha[k]) if not np.isnan(alpha[k]) else 0.0
                        a.step_frame(rows, n, float(ftime[k]), al)
                    if judge[k] >= 0:
                        out[judge[k]] = a.packet_view()
                    if frow is not None and frow[k] >= 0:
                        float_row(int(frow[k]))
            else:
                raise RuntimeError(f"unknown game op {name}")
        return np.concatenate([out.reshape(-1), fout.reshape(-1)]) if len(fout) else out

    def bodies(self, arena):
        a, n = arena["a"], arena["n"]
        out = np.zeros((n + 1, 18), np.float32)
        for i in range(n):
            r = a.car_body(i)
            if r is not None:
                out[i] = body18(r)
        r = a.ball_body()
        if r is not None:
            out[n] = body18(r)
        return out


def body18(r) -> np.ndarray:
    """Bullet body row -> pos, vel (Bullet units, exactly as held), ang vel, forward, right, up (the basis' columns)."""
    o = np.empty(18, np.float32)
    o[0:3] = r[0:3]
    o[3:6] = r[12:15]
    o[6:9] = r[15:18]
    m = r[3:12]
    o[9:12] = (m[0], m[3], m[6])
    o[12:15] = (m[1], m[4], m[7])
    o[15:18] = (m[2], m[5], m[8])
    return o
