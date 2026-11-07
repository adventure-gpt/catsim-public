"""Any simbench simulator as a worker for the full-state suite's own RocketSim v3 harness (its `score_v3.py`).

    python -m simbench.fs_worker catsim [--data catsim_data] [--meshes collision_meshes]

The suite (github.com/thecosmicskye/RocketGPU-fullstate-suite) scores RocketSim v3 through a worker program that it
starts and talks to over stdin and stdout. This module speaks that worker interface, as the suite's README and its
v3 worker's header describe it, and hands each segment to a simbench adapter, so a simulator runs inside the suite's
own harness the same way v3 does. Nothing of the suite is imported or copied; you run its harness, this is the worker:

    python score_v3.py data bench/fs_worker/catsim-fs-worker --json catsim.json

The interface, per segment, little-endian float64 in: the car count (a negative number ends), the tick count; per car
its preset index (RocketSim v3's order: Octane, Dominus, Plank, Breakout, Hybrid, Merc, Psyclops), team and 59 numbers
of v3's CarState (position, the forward, right and up axes, velocity, angular velocity, the 8 previous controls, the
jump and flip flags, flip torque, jump ticks, flip time, the jumping / flipping / jump-hold flags, air times, boost,
handbrake, supersonic flag and timer, boost times and flag, auto-flip flag, timer and scale, bump cooldown, victim and
last extra hit, demolished flag and timer, on-ground flag, the four wheels' contacts); the ball's position, axes and
velocities (18 numbers) and whether it sleeps; the pad count and per pad x, y and cooldown; then per tick, per car,
8 controls, whether the car may drive, and the boost a pad gave it. Out, float32: per tick, per car then the ball,
position (uu), velocity (uu/s), angular velocity, forward, right, up. This is the suite's level B start (the public
state); simbench's own scorer (`python -m simbench suite`) also runs levels A and the packet variant, and frame timing.

Note: the harness divides positions and velocities by 50 to compare them with its Bullet-unit truth. For a simulator
that holds Bullet units (catSim does), each value goes out as the uu float that divides back to exactly its bits.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np

from . import sims
from .state import TICK_CAR

PRESETS = ("Octane", "Dominus", "Plank", "Breakout", "Hybrid", "Merc", "Psyclops")
CAR_F = 59


class In:
    def __init__(self, f):
        self.f = f

    def take(self, n: int) -> np.ndarray:
        b = self.f.read(8 * n)
        if len(b) < 8 * n:
            raise EOFError
        return np.frombuffer(b, "<f8")


def car_state(r) -> dict:
    v = lambda i: [float(r[i]), float(r[i + 1]), float(r[i + 2])]
    return {"pos": v(0), "fwd": v(3), "right": v(6), "up": v(9), "vel": v(12), "ang_vel": v(15),
            "prev_controls": [float(x) for x in r[18:26]], "has_jumped": bool(r[26]), "has_double_jumped": bool(r[27]),
            "has_flipped": bool(r[28]), "flip_rel_torque": v(29), "jump_ticks": int(r[32]), "flip_time": float(r[33]),
            "is_flipping": bool(r[34]), "is_jumping": bool(r[35]), "jump_hold_broken": bool(r[36]), "air_time": float(r[37]),
            "air_time_since_jump": float(r[38]), "boost": float(r[39]), "handbrake_val": float(r[40]), "is_supersonic": bool(r[41]),
            "supersonic_grace_timer": float(r[42]), "time_since_boosted": float(r[43]), "boosting_time": float(r[44]),
            "is_boosting": bool(r[45]), "is_auto_flipping": bool(r[46]), "auto_flip_timer": float(r[47]),
            "auto_flip_torque_scale": float(r[48]), "bump_cooldown_timer": float(r[49]), "bump_victim": int(r[50]),
            "last_extra_hit_tick": int(r[51]), "is_demoed": bool(r[52]), "demo_respawn_timer": float(r[53]),
            "on_ground": bool(r[54]), "wheels_contact": [bool(x) for x in r[55:59]]}


def to_uu(bt: np.ndarray) -> np.ndarray:
    """Bullet units -> uu that the harness's / 50 turns back into the same Bullet-unit bits: x 50, then, where that
    float doesn't survive the division, its nearest neighbour that does."""
    bt = np.asarray(bt, np.float32)
    uu = (bt * np.float32(50.0)).astype(np.float32)
    bad = (uu / np.float32(50.0)).astype(np.float32) != bt
    for step in (1, -1, 2, -2, 3, -3):
        if not bad.any():
            break
        cand = uu.copy()
        for _ in range(abs(step)):
            cand = np.where(bad, np.nextafter(cand, np.float32(np.inf if step > 0 else -np.inf)), cand).astype(np.float32)
        ok = bad & ((cand / np.float32(50.0)).astype(np.float32) == bt)
        uu = np.where(ok, cand, uu)
        bad &= ~ok
    return uu


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sim", help="a simbench simulator name (catsim, rocketsim-v3, ...) or name=path/to/adapter.py")
    ap.add_argument("--data", default=os.environ.get("CATSIM_DATA"))
    ap.add_argument("--meshes", default=os.environ.get("ROCKETGPU_COLLISION_MESH_DIR"))
    a = ap.parse_args(argv)
    env = {"catsim_data": str(Path(a.data).resolve()) if a.data else None,
           "collision_meshes": str(Path(a.meshes).resolve()) if a.meshes else None}
    w = sims.start(a.sim, env)
    units = w.hello.get("units", "uu")
    inp, out = In(sys.stdin.buffer), sys.stdout.buffer
    try:
        while True:
            head = inp.take(1)
            if head[0] < 0:
                break
            nc, ticks = int(head[0]), int(inp.take(1)[0])
            cars, states = [], []
            for _ in range(nc):
                p, t = inp.take(2)
                cars.append({"preset": PRESETS[int(p)], "team": int(t)})
                states.append(car_state(inp.take(CAR_F)))
            b = inp.take(19)
            v = lambda i: [float(b[i]), float(b[i + 1]), float(b[i + 2])]
            ball = {"pos": v(0), "fwd": v(3), "right": v(6), "up": v(9), "vel": v(12), "ang_vel": v(15), "asleep": bool(b[18])}
            n_pads = int(inp.take(1)[0])
            pads = inp.take(3 * n_pads).reshape(n_pads, 3).tolist() if n_pads else []
            rows = inp.take(ticks * nc * TICK_CAR).astype(np.float32).reshape(ticks, nc, TICK_CAR)
            start = {"cars": states, "ball": ball, "pads": pads, "pads_from_recording": True,
                     "next_controls": rows[0, :, :8].tolist() if ticks else None}
            _, data = w.request({"op": "segment", "cars": cars, "ticks": ticks, "frames": False, "start": start}, rows.tobytes())
            res = np.frombuffer(data, np.float32).reshape(ticks, nc + 1, 18).copy()
            if units == "bt":
                res[..., 0:6] = to_uu(res[..., 0:6])
            out.write(res.astype("<f4").tobytes())
            out.flush()
    except EOFError:
        pass
    finally:
        w.close()


if __name__ == "__main__":
    main()
