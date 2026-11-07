"""TEMPLATE: copy this file, fill in the parts marked FILL IN, and your simulator runs every benchmark it can.

    cp bench/adapters/TEMPLATE.py my_sim.py
    python -m simbench check --sims mysim=my_sim.py            # a quick test: starts it, runs a segment, checks the shapes
    python -m simbench all   --sims mysim=my_sim.py,catsim     # every benchmark your capabilities allow

How it fits together. Each simulator runs in its own process (a "worker"); the harness talks to it over stdin/stdout
(bench/simbench/protocol.py). This file is a Python worker: `python -m simbench.kit my_sim.py` serves it, and the kit
turns each request into calls of the methods below. A simulator with a Rust API can use the Rust kit instead
(bench/workers/kit, see bench/workers/rocketsim_v3 for a complete one); any other language can speak the protocol
directly. Your simulator is never imported into the harness, so any licence works.

The words used below:
  state    a plain dict (bench/simbench/state.py): per car `pos`, `fwd`, `right`, `up`, `vel`, `ang_vel` (uu, uu/s,
           rad/s), `boost` (0..100) and, depending on the level, the hidden fields (RocketSim v3's CarState names:
           `is_jumping`, `jump_ticks`, `flip_time`, `wheels_contact`, ...); the ball the same way, plus `asleep`.
           Set every field your simulator can take: that is its best shot, and the results list what you set.
  inputs   per tick, per car: 8 controls (throttle, steer, pitch, yaw, roll, jump, boost, handbrake), then whether the car
           may drive (the game holds cars during countdowns), then the boost the game gave it from a pad (0..1).
  bodies   per car then the ball, 18 floats: pos, vel (in your `units`), ang_vel, forward, right, up axes.

Capabilities say which benchmarks run; anything you leave out is reported as "not supported", never guessed. Keep the
`choices` list honest: it is printed next to your numbers.
"""
from __future__ import annotations

import numpy as np

from simbench.kit import Adapter


class MySim(Adapter):
    name = "mysim"                      # FILL IN: the name printed in every table
    version = "?"                       # FILL IN (in setup is fine): exact version or commit
    about = "my simulator, its Python API"
    device = "cpu"                      # "cpu" or "gpu": CPU and GPU speed tables are never mixed
    units = "uu"                        # "uu", or "bt" (uu / 50) if that is what your simulator holds exactly
    caps = {
        "levels": ["packet", "v3"],     # start states you can take: "packet", "v3" (RocketSim v3's CarState), "rlpr",
                                        # "full" (the full-state suite's own field layout; override `segment` for it)
        "frame_timing": False,          # True if you take each frame's real duration and render alpha (`step_frame`)
        "game_flow": False,             # True if you run the game's own countdowns, goals and restarts (levels C, D)
        "rng_stream": False,            # True if you predict kickoff / respawn spots from the game's random stream (D)
        "spot_choices": False,          # True if you take kickoff / respawn spots handed to you (C)
        "threads": False,               # True if your `speed` uses the thread count it is given
    }
    choices = [                         # FILL IN: what you set, what you can't, which settings you use
        "sets position, rotation, velocities and boost; jump and flip state",
        "can't set: wheel contacts, contact caches",
        "release build, default settings",
    ]

    def setup(self, env):
        """env: {"catsim_data": folder, "collision_meshes": RocketSim's mesh folder (made from catSim's dump), "threads": n}"""
        import mysim                    # FILL IN: your module
        self.sim = mysim
        mysim.init(env["collision_meshes"])
        self.version = mysim.__version__

    def new_arena(self, cars, map="Stadium_P"):
        """cars: [{"preset": "Octane" | "Dominus" | "Plank" | "Breakout" | "Hybrid" | "Merc" | "Psyclops", "team": 0 or 1,
        "config": the full car configuration when the benchmark has one}]"""
        a = self.sim.Arena()                                            # FILL IN
        return {"arena": a, "cars": [a.add_car(c["team"], c["preset"]) for c in cars]}

    def set_state(self, arena, state):
        for car, c in zip(arena["cars"], state["cars"]):
            car.set(pos=c["pos"], forward=c["fwd"], up=c["up"], vel=c["vel"], ang_vel=c["ang_vel"], boost=c.get("boost", 33.3))
            # FILL IN: every hidden field you can take, e.g. if "is_jumping" in c: car.jumping = c["is_jumping"]
        b = state["ball"]
        arena["arena"].ball.set(pos=b["pos"], vel=b["vel"], ang_vel=b["ang_vel"])

    def step(self, arena, inputs, ticks=1, frame=None):
        for car, r in zip(arena["cars"], inputs):
            car.controls(*r[:8]) if r[8] >= 0.5 else car.controls(0, r[1], 0, 0, 0, 0, 0, 0)
        arena["arena"].step(ticks)
        for car, r in zip(arena["cars"], inputs):
            if r[9] > 0:
                car.add_boost(100 * r[9])

    def bodies(self, arena):
        out = np.zeros((len(arena["cars"]) + 1, 18), np.float32)
        for k, car in enumerate(arena["cars"]):
            s = car.state()
            out[k] = [*s.pos, *s.vel, *s.ang_vel, *s.forward, *s.right, *s.up]
        b = arena["arena"].ball.state()
        out[-1] = [*b.pos, *b.vel, *b.ang_vel, *b.forward, *b.right, *b.up]
        return out

    # Optional, and worth it: your fastest path for the speed benchmark (batched, threaded or on the GPU). The default in
    # the kit steps one arena after another on one thread. See bench/adapters/rocketgpu.py and rocketsim_v2.py.
    # def speed(self, job, starts, controls): ...
