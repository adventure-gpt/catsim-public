"""RocketSim v2 (the C++ RocketSim) through its Python package: `pip install rocketsim` (imports as `RocketSim`).

Its fastest published path for many arenas is `Arena.multi_step(arenas, ticks)` (the package steps the arenas on its own
threads); one arena runs with `arena.step`. State in through `car.set_state` / `ball.set_state` with every CarState field
the level gives; bodies out through `get_state`. The package reads RocketSim's collision meshes (`collision_meshes/`).
"""
from __future__ import annotations

import time

import numpy as np

from simbench.kit import Adapter
from simbench.state import v3_from_packet


class RocketSimV2(Adapter):
    name = "rocketsim-v2"
    device = "cpu"
    units = "uu"
    # Accuracy levels are left off: v2's CarState clocks mean other things than v3's (its flip and jump timers), so a start
    # mapped from the suite's v3 rows isn't v2's best shot; set_state below is ready for a mapping someone checks.
    caps = {"levels": [], "frame_timing": False, "game_flow": False, "rng_stream": False, "spot_choices": False,
            "threads": True}
    choices = [
        "accuracy not run: v2's CarState clocks (flip, jump) mean other things than v3's, and no checked mapping exists yet",
        "sets every CarState field the level gives (jump and flip clocks, flip torque, boost, supersonic and handbrake "
        "state, auto-flip, world contact, wheel count, last controls, demolition); the ball through BallState",
        "batched speed through Arena.multi_step (its own threads; the thread count isn't settable from Python)",
        "takes no frame timing and no contact manifolds; per-car Python calls set controls and read states",
        "accuracy: the ball's rotation turned on (ArenaConfig.no_ball_rot = False); speed: its default (the ball's rotation "
        "left out, a little less work)",
    ]

    def setup(self, env):
        import RocketSim as rs
        import importlib.metadata as md
        self.rs = rs
        rs.init(env["collision_meshes"])
        self.version = f"rocketsim {md.version('rocketsim')} (PyPI)"
        self.about = "RocketSim v2 (C++) through its Python package"
        self.presets = {"Octane": rs.CarConfig.OCTANE, "Dominus": rs.CarConfig.DOMINUS, "Plank": rs.CarConfig.PLANK,
                        "Breakout": rs.CarConfig.BREAKOUT, "Hybrid": rs.CarConfig.HYBRID, "Merc": rs.CarConfig.MERC}

    def new_arena(self, cars, map="Stadium_P", ball_rotation=True):
        rs = self.rs
        cfg = rs.ArenaConfig()
        cfg.no_ball_rot = not ball_rotation          # v2 leaves the ball's rotation out by default; accuracy turns it on
        a = rs.Arena(rs.GameMode.SOCCAR, config=cfg)
        out = []
        for c in cars:
            team = rs.Team.BLUE if int(c["team"]) == 0 else rs.Team.ORANGE
            out.append(a.add_car(team, rs.CarConfig(self.presets.get(c["preset"], rs.CarConfig.OCTANE))))
        return {"a": a, "cars": out}

    def _vec(self, v):
        return self.rs.Vec(float(v[0]), float(v[1]), float(v[2]))

    def set_state(self, arena, st):
        rs = self.rs
        next_ctl = st.get("next_controls")
        for k, (car, c) in enumerate(zip(arena["cars"], st["cars"])):
            c = v3_from_packet(c)
            s = car.get_state()
            s.pos, s.vel, s.ang_vel = self._vec(c["pos"]), self._vec(c["vel"]), self._vec(c["ang_vel"])
            if "fwd" in c:
                s.rot_mat = rs.RotMat(self._vec(c["fwd"]), self._vec(c["right"]), self._vec(c["up"]))
            for src, dst in (("boost", "boost"), ("air_time", "air_time"), ("air_time_since_jump", "air_time_since_jump"),
                             ("flip_time", "flip_time"), ("handbrake_val", "handbrake_val"), ("auto_flip_timer", "auto_flip_timer"),
                             ("auto_flip_torque_scale", "auto_flip_torque_scale"), ("demo_respawn_timer", "demo_respawn_timer"),
                             ("supersonic_grace_timer", "supersonic_time"), ("boosting_time", "time_spent_boosting"),
                             ("bump_cooldown_timer", "car_contact_cooldown_timer")):
                if src in c:
                    setattr(s, dst, float(c[src]))
            for src, dst in (("on_ground", "is_on_ground"), ("has_jumped", "has_jumped"), ("has_double_jumped", "has_double_jumped"),
                             ("has_flipped", "has_flipped"), ("is_flipping", "is_flipping"), ("is_jumping", "is_jumping"),
                             ("is_supersonic", "is_supersonic"), ("is_auto_flipping", "is_auto_flipping"), ("is_demoed", "is_demoed"),
                             ("world_contact", "has_world_contact")):
                if src in c:
                    setattr(s, dst, bool(c[src]))
            if "jump_time" in c:
                s.jump_time = float(c["jump_time"])
            elif "jump_ticks" in c:
                s.jump_time = float(c["jump_ticks"]) / 120.0
            if "flip_rel_torque" in c:
                s.flip_rel_torque = self._vec(c["flip_rel_torque"])
                s.last_rel_dodge_torque = self._vec(c["flip_rel_torque"])
            if "wheels_contact" in c:
                s.wheels_with_contact = tuple(bool(x) for x in c["wheels_contact"])
            elif "on_ground" in c:      # a packet doesn't show the wheels: all four touching when on the ground
                s.wheels_with_contact = (bool(c["on_ground"]),) * 4
            if "bump_victim" in c and int(c["bump_victim"]) >= 0:
                s.car_contact_id = arena["cars"][int(c["bump_victim"])].id
            held = c.get("prev_controls") or (next_ctl[k] if next_ctl else None)
            if held is not None:
                s.last_controls = rs.CarControls(*[float(x) for x in held[:5]], held[5] > 0.5, held[6] > 0.5, held[7] > 0.5)
            car.set_state(s)
        b = st["ball"]
        bs = arena["a"].ball.get_state()
        bs.pos, bs.vel, bs.ang_vel = self._vec(b["pos"]), self._vec(b["vel"]), self._vec(b["ang_vel"])
        if "fwd" in b:
            bs.rot_mat = rs.RotMat(self._vec(b["fwd"]), self._vec(b["right"]), self._vec(b["up"]))
        arena["a"].ball.set_state(bs)
        if st.get("pads"):
            pads = arena["a"].get_boost_pads()
            for x, y, cd in st["pads"]:
                p = min(pads, key=lambda q: (q.get_pos().x - x) ** 2 + (q.get_pos().y - y) ** 2)
                ps = p.get_state()
                ps.is_active, ps.cooldown = cd <= 0, float(max(cd, 0.0))
                p.set_state(ps)
        arena["pads_from_recording"] = bool(st.get("pads_from_recording"))

    def _controls(self, arena, inputs):
        rs = self.rs
        for car, r in zip(arena["cars"], inputs):
            if r[8] < 0.5:
                car.set_controls(rs.CarControls(steer=float(r[1])))
            else:
                car.set_controls(rs.CarControls(float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]),
                                                bool(r[5] > 0.5), bool(r[6] > 0.5), bool(r[7] > 0.5)))

    def step(self, arena, inputs, ticks=1, frame=None):
        self._controls(arena, inputs)
        arena["a"].step(int(ticks))
        if arena.get("pads_from_recording"):
            for car, r in zip(arena["cars"], inputs):
                if r[9] > 0:
                    s = car.get_state()
                    s.boost = min(100.0, s.boost + float(r[9]) * 100.0)
                    car.set_state(s)

    def bodies(self, arena):
        out = np.zeros((len(arena["cars"]) + 1, 18), np.float32)
        for k, car in enumerate(arena["cars"]):
            s = car.get_state()
            if s.is_demoed:
                continue
            out[k] = row(s)
        out[-1] = row(arena["a"].ball.get_state())
        return out

    def speed(self, job, starts, controls):
        """Batched: every arena's controls, then Arena.multi_step over all of them (the package's own threads), then every
        arena's state through get_gym_state; one arena on one thread: arena.step."""
        cars = job["cars"]
        # the package doesn't free an Arena's C++ side when Python drops it: reuse one set of arenas per roster
        key = (tuple((c["preset"], c["team"]) for c in cars), job["arenas"])
        if getattr(self, "_speed_key", None) != key:
            self._speed_arenas = None
            self._speed_arenas = [self.new_arena(cars, ball_rotation=False) for _ in range(job["arenas"])]
            self._speed_key = key
        arenas = self._speed_arenas
        n_dec, k = controls.shape[1], job["decision_ticks"]
        threaded = job["threads"] > 1
        out = []
        inp = np.zeros((len(cars), 10), np.float32)
        inp[:, 8] = 1.0
        for rep in range(job["warmup"] + job["repeats"]):
            for a, st in zip(arenas, starts):
                self.set_state(a, st)
            t0 = time.perf_counter()
            for d in range(n_dec):
                for i, a in enumerate(arenas):
                    inp[:, :8] = controls[i, d]
                    self._controls(a, inp)
                if threaded:
                    self.rs.Arena.multi_step([a["a"] for a in arenas], k)
                else:
                    for a in arenas:
                        a["a"].step(k)
                for a in arenas:
                    a["a"].get_gym_state()
            if rep >= job["warmup"]:
                out.append(time.perf_counter() - t0)
        return {"seconds": out, "threads": job["threads"] if threaded else 1}


def row(s) -> np.ndarray:
    m = s.rot_mat
    return np.asarray([s.pos.x, s.pos.y, s.pos.z, s.vel.x, s.vel.y, s.vel.z, s.ang_vel.x, s.ang_vel.y, s.ang_vel.z,
                       m.forward.x, m.forward.y, m.forward.z, m.right.x, m.right.y, m.right.z, m.up.x, m.up.y, m.up.z], np.float32)
