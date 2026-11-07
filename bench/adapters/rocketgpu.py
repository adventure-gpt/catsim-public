"""RocketGPU (github.com/thecosmicskye/RocketGPU, MIT) through its own Python module, which you build as its README says.

Point the adapter at your build's python folder: SIMBENCH_ROCKETGPU_CUDA=.../build-cuda/python (the `rocketgpu` sim) and
SIMBENCH_ROCKETGPU_CPU=.../build-cpu/python (`rocketgpu-cpu`; its README's fastest CPU build is the profile-guided one,
tools/pgo_build.sh). ROCKETGPU_COLLISION_MESH_DIR is set from the harness's collision_meshes folder.

Speed: its batched path, `BatchSim` (all arenas in one simulator): every decision the 8 ticks' controls go in one call,
`step_with_device_controls` (a device buffer on the CUDA build, a host array on the CPU build); the states stay where
the simulator keeps them (on the card for CUDA; read through its host view on the CPU build).
Accuracy: its RocketSim-style `Arena` (one arena): level A through `set_full_state` with the full-state suite's fields,
the suite's per-segment calls and mutators replayed first; level B through `set_state_v3` (RocketSim v3's CarState,
field for field) and the ball's sleep through its full state. Per tick, as the suite runs RocketGPU: cars the game holds
through `driving_disabled`, the ball held while every car is, each frame's render fraction and start time through the
mutator config, pad credits on the boost field, bodies read from the full state (Bullet units).
"""
from __future__ import annotations

import ctypes
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from simbench.kit import Adapter
from simbench.state import v3_from_packet

PRESET = {"Octane": "CAR_CONFIG_OCTANE", "Dominus": "CAR_CONFIG_DOMINUS", "Plank": "CAR_CONFIG_PLANK",
          "Breakout": "CAR_CONFIG_BREAKOUT", "Hybrid": "CAR_CONFIG_HYBRID", "Merc": "CAR_CONFIG_MERC",
          "Psyclops": "CAR_CONFIG_PSYCLOPS"}
INT = {1: "<B", 2: "<H", 4: "<I"}


def local(p):
    """A Windows path given to a worker running in WSL."""
    if p and len(p) > 2 and p[1] == ":" and sys.platform != "win32":
        return "/mnt/" + p[0].lower() + p[2:].replace("\\", "/")
    return p


class RocketGPU(Adapter):
    name = "rocketgpu"
    units = "bt"
    caps = {"levels": ["full", "v3", "packet"], "frame_timing": False, "game_flow": False, "rng_stream": False,
            "spot_choices": False, "threads": True, "threads_at_setup": True}
    choices = [
        "speed: BatchSim, one simulator for all arenas; 8 ticks' controls a call (step_with_device_controls); "
        "boost pads, car-car collisions, demos and scoring on; states stay in the simulator",
        "level A: set_full_state with every suite field, after the segment's own calls and mutators (as the suite runs it)",
        "level B: set_state_v3 (v3's CarState field for field) and the ball's sleep flag through the full state",
        "per tick: driving_disabled, the ball's 2D hold, the frame's render fraction and start time, pad credits",
    ]

    def __init__(self):
        self.cuda = os.environ.get("SIMBENCH_ROCKETGPU_BUILD", "cuda") == "cuda"
        self.name = "rocketgpu" if self.cuda else "rocketgpu-cpu"
        self.device = "gpu" if self.cuda else "cpu"

    def setup(self, env):
        build = "cuda" if self.cuda else "cpu"
        path = os.environ.get("SIMBENCH_ROCKETGPU_CUDA" if self.cuda else "SIMBENCH_ROCKETGPU_CPU")
        if not path or not Path(path).is_dir():
            raise RuntimeError(f"set SIMBENCH_ROCKETGPU_{build.upper()} to your RocketGPU {build} build's python folder")
        sys.path.insert(0, path)
        os.environ["ROCKETGPU_COLLISION_MESH_DIR"] = local(env.get("collision_meshes") or "")
        if env.get("threads"):
            os.environ["ROCKETGPU_CPU_THREADS"] = str(env["threads"])      # (fixed at its first CPU call)
        import rocketgpu_py as rp
        import rocketsim as rs
        import rocketgpu_py_cuda as views
        self.rp, self.rs, self.views = rp, rs, views
        if self.cuda and not getattr(rp, "is_cuda_build", False):
            raise RuntimeError("SIMBENCH_ROCKETGPU_CUDA points at a CPU build")
        src = Path(path).resolve().parents[1]
        commit = subprocess.run(["git", "-C", str(src), "rev-parse", "--short=10", "HEAD"], capture_output=True, text=True).stdout.strip()
        self.version = f"RocketGPU {commit or '?'} ({'CUDA' if self.cuda else 'CPU'} build)"
        self.about = f"RocketGPU's Python module, {'its CUDA build' if self.cuda else 'its CPU build'}"

    # ------------------------------------------------------------ one arena (accuracy)

    def _config(self, c):
        rp, rs = self.rp, self.rs
        cfg = c.get("config")
        if not cfg:
            return getattr(rs, PRESET.get(c["preset"], "CAR_CONFIG_OCTANE"))
        return self._obj(dict(cfg, __type__="CarConfig"))

    def _obj(self, o):
        """A value the suite's metadata wrote (objects by type name, enums by value) as this build's object."""
        rp = self.rp
        if isinstance(o, list):
            return [self._obj(x) for x in o]
        if not isinstance(o, dict):
            return o
        if "__enum__" in o:
            for m in (self.rs, rp.compat, rp):
                cls = getattr(m, o["__enum__"], None)
                if cls is not None:
                    return next(v for v in cls.__members__.values() if int(v) == o["value"])
        if "__type__" in o:
            for m in (self.rs, rp.compat, rp):
                cls = getattr(m, o["__type__"], None)
                if cls is not None:
                    x = cls()
                    for k, v in o.items():
                        if k != "__type__":
                            setattr(x, k, self._obj(v))
                    return x
            raise RuntimeError(f"this RocketGPU build has no {o['__type__']}")
        return {k: self._obj(v) for k, v in o.items()}

    def new_arena(self, cars, map="Stadium_P"):
        rp, rs = self.rp, self.rs
        cfg = rp.compat.ArenaConfig()
        cfg.no_ball_rot = False
        a = rs.Arena(rs.GameMode.SOCCAR, cfg, 120.0)
        out = [a.add_car(rs.Team.Blue if int(c["team"]) == 0 else rs.Team.Orange, self._config(c)) for c in cars]
        return {"a": a, "cars": out, "n": len(cars)}

    def _fields(self, arena):
        return arena["a"].get_full_state(["car_states", "ball_states"])

    def _put(self, st, new, key, k, fmt, *v):
        n = len(st[key]) // (len([1]) if not key.startswith("car_states.") else self._ncap(st))
        buf = new.setdefault(key, bytearray(st[key]))
        struct.pack_into(fmt, buf, k * n, *v)

    def _ncap(self, st):
        return len(st["car_states.boost"]) // 4

    def set_state(self, arena, st):
        rp, rs = self.rp, self.rs
        a = arena["a"]
        for car, c in zip(arena["cars"], st["cars"]):
            c = v3_from_packet(c)
            s = car.get_state_v3()
            v = lambda x: rp.Vec3(float(x[0]), float(x[1]), float(x[2]))
            s.pos, s.vel, s.ang_vel = v(c["pos"]), v(c["vel"]), v(c["ang_vel"])
            if "fwd" in c:
                s.rot_mat.forward, s.rot_mat.right, s.rot_mat.up = v(c["fwd"]), v(c["right"]), v(c["up"])
            for k in ("has_jumped", "has_double_jumped", "has_flipped", "is_flipping", "is_jumping", "jump_hold_broken",
                      "is_supersonic", "is_boosting", "is_auto_flipping", "is_demoed"):
                if k in c:
                    setattr(s, k, bool(c[k]))
            for k in ("flip_time", "air_time", "air_time_since_jump", "boost", "handbrake_val", "supersonic_grace_timer",
                      "time_since_boosted", "boosting_time", "auto_flip_timer", "auto_flip_torque_scale",
                      "bump_cooldown_timer", "demo_respawn_timer"):
                if k in c:
                    setattr(s, k, float(c[k]))
            if "on_ground" in c:
                s.is_on_ground = bool(c["on_ground"])
            if "jump_ticks" in c:
                s.jump_ticks = int(c["jump_ticks"])
            if "flip_rel_torque" in c:
                s.flip_rel_torque = v(c["flip_rel_torque"])
            if "wheels_contact" in c:
                s.wheels_with_contact = [bool(x) for x in c["wheels_contact"]]
            elif "on_ground" in c:
                s.wheels_with_contact = [bool(c["on_ground"])] * 4
            held = c.get("prev_controls") or (st.get("next_controls") or [None] * arena["n"])[arena["cars"].index(car)]
            if held is not None:
                ctl = rs.CarControls()
                ctl.throttle, ctl.steer, ctl.pitch, ctl.yaw, ctl.roll = (float(x) for x in held[:5])
                ctl.jump, ctl.boost, ctl.handbrake = bool(held[5] > 0.5), bool(held[6] > 0.5), bool(held[7] > 0.5)
                s.prev_controls = ctl
                s.controls = ctl
            if "bump_victim" in c:
                vi = int(c["bump_victim"])
                s.bump_cooldown_victim = arena["cars"][vi].id if 0 <= vi < arena["n"] else None
            if "last_extra_hit_tick" in c:
                h = int(c["last_extra_hit_tick"])
                s.last_extra_hit_tick = h if h >= 0 else None
            car.set_state_v3(s)
        b = st["ball"]
        bs = a.ball.get_state()
        v = lambda x: rp.Vec3(float(x[0]), float(x[1]), float(x[2]))
        bs.pos, bs.vel, bs.ang_vel = v(b["pos"]), v(b["vel"]), v(b["ang_vel"])
        if "fwd" in b:
            bs.rot_mat.forward, bs.rot_mat.right, bs.rot_mat.up = v(b["fwd"]), v(b["right"]), v(b["up"])
        a.ball.set_state(bs)
        if b.get("asleep"):
            f = self._fields(arena)
            a.set_full_state({"ball_states.sleeping": struct.pack("<I", 1)[:len(f["ball_states.sleeping"])]})
        arena["pads_from_recording"] = bool(st.get("pads_from_recording"))

    def segment(self, req, inputs, frames, full):
        if "full" not in req["start"]:
            return super().segment(req, inputs, frames, full)
        a = self.new_arena(req["cars"])
        arena = a["a"]
        ids = [c.id for c in a["cars"]]
        if req.get("ids") and ids != list(req["ids"]):
            raise RuntimeError(f"car ids {ids} are not the recording's {req['ids']}")
        for name, args, kw in req.get("calls", []):
            getattr(arena, name)(*self._obj(args), **self._obj(kw))
        if req.get("mutator"):
            m = arena.get_mutator_config()
            for k, v in req["mutator"].items():
                if k != "__type__":
                    setattr(m, k, self._obj(v))
            arena.set_mutator_config(m)
        state = {k: bytes(full[o:o + n]) for k, (o, n) in req["start"]["full"].items()}
        arena.set_full_state(state)
        a["pads_from_recording"] = True
        n = int(req["ticks"])
        out = np.zeros((n, a["n"] + 1, 18), np.float32)
        for t in range(n):
            self.step(a, inputs[t], 1, None if frames is None else frames[t], hold=not req["start"].get("timer_hold"))
            out[t] = self.bodies(a)
        return out

    def step(self, arena, inputs, ticks=1, frame=None, hold=True):
        a = arena["a"]
        st = self._fields(arena)
        new = {}
        cap = self._ncap(st)
        if hold:
            n = len(st["car_states.driving_disabled"]) // cap
            buf = new.setdefault("car_states.driving_disabled", bytearray(st["car_states.driving_disabled"]))
            for k in range(arena["n"]):
                struct.pack_into(INT[n], buf, k * n, 0 if inputs[k, 8] >= 0.5 else 1)
            n2 = len(st["ball_states.constrained_2d"])
            want = 1 if arena["n"] and not (inputs[:, 8] >= 0.5).any() else 0
            if struct.unpack_from(INT[n2], st["ball_states.constrained_2d"], 0)[0] != want:
                new["ball_states.constrained_2d"] = struct.pack(INT[n2], want)
        if new:
            a.set_full_state({k: bytes(v) for k, v in new.items()})
        if frame is not None:
            m = a.get_mutator_config()
            if m.frame_start_dt != float(frame[4]):
                m.frame_start_dt = float(frame[4])
            m.boost_pad_frame_frac = float(frame[3])
            a.set_mutator_config(m)
        for car, r in zip(arena["cars"], inputs):
            c = self.rs.CarControls()
            c.throttle, c.steer, c.pitch, c.yaw, c.roll = (float(x) for x in r[:5])
            c.jump, c.boost, c.handbrake = bool(r[5] > 0.5), bool(r[6] > 0.5), bool(r[7] > 0.5)
            car.set_controls(c)
        a.step(int(ticks))
        if arena.get("pads_from_recording") and (inputs[:, 9] > 0).any():
            st = self._fields(arena)
            buf = bytearray(st["car_states.boost"])
            for k in range(arena["n"]):
                g = float(inputs[k, 9])
                if g <= 0:
                    continue
                b = struct.unpack_from("<f", buf, 4 * k)[0]
                m = round(g / 0.12)
                struct.pack_into("<f", buf, 4 * k, min(100.0, b + 12.0 * m) if 1 <= m <= 4 and abs(g - 0.12 * m) < 1e-3 else 100.0)
            a.set_full_state({"car_states.boost": bytes(buf)})

    def bodies(self, arena):
        st = self._fields(arena)
        cap = self._ncap(st)
        out = np.zeros((arena["n"] + 1, 18), np.float32)
        def g(key, k, per):
            return np.frombuffer(st[key], np.float32)[k * per:k * per + 3]
        for k in range(arena["n"]):
            if struct.unpack_from("<B", st["car_states.is_demoed"], k * (len(st["car_states.is_demoed"]) // cap))[0]:
                continue
            per = len(st["car_states.pos_bt"]) // 4 // cap
            out[k] = np.concatenate([g("car_states.pos_bt", k, per), g("car_states.vel_bt", k, per), g("car_states.internal_ang_vel", k, per),
                                     g("car_states.basis_forward", k, per), g("car_states.basis_right", k, per), g("car_states.basis_up", k, per)])
        out[-1] = np.concatenate([g("ball_states.pos_bt", 0, 3), g("ball_states.vel_bt", 0, 3), g("ball_states.ang_vel", 0, 3),
                                  g("ball_states.basis_forward", 0, 3), g("ball_states.basis_right", 0, 3), g("ball_states.basis_up", 0, 3)])
        return out

    # ------------------------------------------------------------ many arenas (speed)

    def speed(self, job, starts, controls):
        rp, views = self.rp, self.views
        n_ar, cars = job["arenas"], job["cars"]
        nc = len(cars)
        cfg = rp.SimConfig()
        cfg.env_count, cfg.max_cars_per_env = n_ar, nc
        cfg.enable_boost_pads = cfg.enable_car_car_collisions = cfg.enable_demos = cfg.enable_scoring = 1
        cfg.tick_dt = 1.0 / 120.0
        sim = rp.BatchSim(cfg)
        cap = sim.car_capacity()
        sim.set_env_car_count([nc] * n_ar)
        sim.set_teams([rp.Team.Blue if c["team"] == 0 else rp.Team.Orange for c in cars] * n_ar)
        car_states, ball_states = self._batch_states(starts, nc)
        k = job["decision_ticks"]
        dt = views.state_dtype("controls")
        host = np.zeros((k, cap), dtype=dt)
        dev = _Device(host.nbytes) if self.cuda else None
        table = np.zeros((controls.shape[1], cap, 8), np.float32)
        table[:, :n_ar * nc] = controls.transpose(1, 0, 2, 3).reshape(controls.shape[1], n_ar * nc, 8)
        cpu_view = None if self.cuda else views.host_struct_array(sim, "car_states")
        out = []
        for rep in range(job["warmup"] + job["repeats"]):
            sim.upload_car_states(car_states)
            sim.upload_ball_states(ball_states)
            t0 = time.perf_counter()
            for d in range(controls.shape[1]):
                c = table[d]
                for i, name in enumerate(("throttle", "steer", "pitch", "yaw", "roll")):
                    host[name][:] = c[:, i]
                for i, name in ((5, "jump"), (6, "boost"), (7, "handbrake")):
                    host[name][:] = c[:, i] > 0.5
                if self.cuda:
                    dev.write(host)
                    sim.step_with_device_controls(dev.ptr, k, cap)
                else:
                    sim.step_with_device_controls(host.ctypes.data, k, cap)
                    cpu_view["pos"].copy()
            if self.cuda:
                dev.sync()
            if rep >= job["warmup"]:
                out.append(time.perf_counter() - t0)
        return {"seconds": out, "threads": int(os.environ.get("ROCKETGPU_CPU_THREADS", os.cpu_count())) if not self.cuda else 1}

    def _quat(self, f, r, u):
        """A rotation given by its forward / right / up axes as RocketGPU's quaternion (w, x, y, z)."""
        m = np.array([f, r, u], np.float64).T          # columns: forward, right, up
        t = m[0, 0] + m[1, 1] + m[2, 2]
        if t > 0:
            k = 0.5 / np.sqrt(t + 1.0)
            w, x, y, z = 0.25 / k, (m[2, 1] - m[1, 2]) * k, (m[0, 2] - m[2, 0]) * k, (m[1, 0] - m[0, 1]) * k
        else:
            i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
            j, k2 = (i + 1) % 3, (i + 2) % 3
            sq = np.sqrt(1.0 + m[i, i] - m[j, j] - m[k2, k2]) * 2.0
            q = [0.0, 0.0, 0.0]
            q[i] = 0.25 * sq
            q[j] = (m[j, i] + m[i, j]) / sq
            q[k2] = (m[k2, i] + m[i, k2]) / sq
            w = (m[k2, j] - m[j, k2]) / sq
            x, y, z = q
        out = self.rp.Quat()
        out.w, out.x, out.y, out.z = float(w), float(x), float(y), float(z)
        return out

    def _batch_states(self, starts, nc):
        rp = self.rp
        cars, balls = [], []
        v = lambda x: rp.Vec3(float(x[0]), float(x[1]), float(x[2]))
        for st in starts:
            for c in st["cars"]:
                s = rp.CarState()
                s.pos, s.vel, s.ang_vel = v(c["pos"]), v(c["vel"]), v(c["ang_vel"])
                s.rot = self._quat(c["fwd"], c["right"], c["up"])
                s.boost = float(c.get("boost", 33.3))
                s.is_on_ground = bool(c.get("on_ground", True))
                s.tick_count_since_update = 0
                cars.append(s)
            b = rp.BallState()
            bb = st["ball"]
            b.pos, b.vel, b.ang_vel = v(bb["pos"]), v(bb["vel"]), v(bb["ang_vel"])
            b.tick_count_since_update = 0
            balls.append(b)
        return cars, balls


class _Device:
    """A device buffer for the controls (CUDA runtime through ctypes; RocketGPU reads the controls from the card)."""

    def __init__(self, nbytes):
        for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12", "cudart64_13.dll", "cudart64_12.dll"):
            try:
                self.rt = ctypes.CDLL(name)
                break
            except OSError:
                continue
        else:
            raise RuntimeError("the CUDA runtime (libcudart) was not found")
        p = ctypes.c_void_p()
        if self.rt.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(nbytes)):
            raise RuntimeError("cudaMalloc failed")
        self.ptr, self.nbytes = p.value, nbytes

    def write(self, arr):
        self.rt.cudaMemcpy(ctypes.c_void_p(self.ptr), arr.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(arr.nbytes), 1)

    def sync(self):
        self.rt.cudaDeviceSynchronize()
