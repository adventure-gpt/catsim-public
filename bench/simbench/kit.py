"""Worker kit for adapters written in Python: subclass `Adapter`, then run `python -m simbench.kit your_adapter.py`.

The kit speaks the wire protocol (simbench.protocol) and turns each request into calls of your adapter's methods, so an
adapter is only the simulator-specific part: make an arena, set a state, set controls and step, read the bodies back.
Batched or threaded simulators can also override `speed` (and `segments`) to use their own fastest path.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import protocol
from .state import BODY, TICK_CAR, TICK_FRAME


class Adapter:
    """One simulator. Fill in the class attributes and the five methods marked REQUIRED; the rest are optional.

    Capabilities (`caps`) tell the harness which benchmarks the adapter can run; anything left out is reported as
    "not supported" for this simulator, never guessed:
      levels        accuracy levels it can start from: "packet" (B1), "v3" (B2), "rlpr", "full" (A)
      frame_timing  it takes each frame's real duration and render alpha (`step(..., frame=...)`)
      game_flow     it runs the game's own countdowns, goals, kickoffs and respawns over a whole recording (level C)
      rng_stream    it predicts kickoff and respawn spots from the game's random-number state (level D)
      spot_choices  it takes kickoff and respawn spots handed to it (level C's assist)
      threads       its `speed` uses the thread count it is given (else only 1-thread speed rows are run)
    and the class attribute `device`: "cpu" or "gpu" (speed tables never mix the two).
    """
    name = "my-sim"
    version = "?"          # the simulator's exact version or commit, printed with every result
    about = ""             # one line: what is run and how (build, settings), printed with every result
    device = "cpu"
    units = "uu"           # the bodies' position and velocity units: "uu" (Unreal) or "bt" (Bullet: uu / 50), as the
                           # simulator holds them, so no conversion rounds them (the harness converts, as each truth needs)
    caps: dict = {"levels": ["packet"], "frame_timing": False, "game_flow": False, "rng_stream": False, "spot_choices": False,
                  "threads": False}
    choices: list = []     # plain-words list of what this adapter sets, what it can't, which settings it uses

    def setup(self, env: dict) -> None:
        """Called once. `env`: {"catsim_data": folder or None, "collision_meshes": folder or None, "threads": n}."""

    def new_arena(self, cars: list, map: str):  # REQUIRED
        """A soccar arena with these cars ({"preset": RocketSim preset name, "body": game body product id or None,
        "team": 0 blue / 1 orange}), in order. Returns anything; it is passed back to the other methods."""
        raise NotImplementedError

    def set_state(self, arena, state: dict) -> None:  # REQUIRED
        """Put the arena in this state (simbench.state). Set every field your simulator can take."""
        raise NotImplementedError

    def step(self, arena, inputs: np.ndarray, ticks: int = 1, frame: np.ndarray | None = None) -> None:  # REQUIRED
        """`inputs`: float32 [cars, 10] (8 controls, may-drive flag, pad boost gain 0..1 given after the tick).
        Step `ticks` physics ticks (1/120 s each). `frame`: the tick's frame row (simbench.state.TICK_FRAME) when the
        benchmark has one and the adapter steps tick by tick; ignore it if you don't use it."""
        raise NotImplementedError

    def step_frame(self, arena, ticks: np.ndarray, frame_time: float, alpha: float) -> np.ndarray:
        """One game frame of len(ticks) physics ticks with its real duration and render alpha; only called when
        `caps["frame_timing"]`. `ticks`: float32 [ticks, cars, 10], the inputs recorded on each tick of the frame (the
        game reads controls once, at the frame's start; pad credits come after the frame). Returns the bodies after each
        tick: float32 [ticks, cars + 1, 18]. The default steps the ticks one by one."""
        out = []
        for t in range(len(ticks)):
            self.step(arena, ticks[t], 1)
            out.append(self.bodies(arena))
        return np.stack(out)

    def bodies(self, arena) -> np.ndarray:  # REQUIRED
        """float32 [cars + 1, 18]: per car then the ball, pos, vel (in `units`), ang_vel, forward, right, up.
        A car without a body (demolished) is all zeros."""
        raise NotImplementedError

    def close_arena(self, arena) -> None:
        pass

    # ------------------------------------------------------------ batched paths (override for speed)

    def speed(self, job: dict, starts: list, controls: np.ndarray) -> dict:
        """The speed benchmark. `controls`: float32 [arenas, decisions, cars, 8]. Each repeat: every arena to its start
        state (not timed), then for each decision: set each car's controls, step `decision_ticks`, read every body.
        Returns {"seconds": [timed seconds per repeat], "threads": threads used}. This default runs one arena after
        another on one thread; override it with your simulator's own batched or threaded path."""
        cars = job["cars"]
        arenas = [self.new_arena(cars, job.get("map", "Stadium_P")) for _ in range(job["arenas"])]
        n_dec, k = controls.shape[1], job["decision_ticks"]
        inp = np.zeros((len(cars), TICK_CAR), np.float32)
        inp[:, 8] = 1.0
        out = []
        for rep in range(job["warmup"] + job["repeats"]):
            for a, st in zip(arenas, starts):
                self.set_state(a, st)
            t0 = time.perf_counter()
            for d in range(n_dec):
                for i, a in enumerate(arenas):
                    inp[:, :8] = controls[i, d]
                    self.step(a, inp, k)
                    self.bodies(a)
            if rep >= job["warmup"]:
                out.append(time.perf_counter() - t0)
        for a in arenas:
            self.close_arena(a)
        return {"seconds": out, "threads": 1}

    def segment(self, req: dict, inputs: np.ndarray, frames: np.ndarray | None, full: bytes) -> np.ndarray:
        """One accuracy segment: a fresh arena, the start state, then `ticks` ticks of inputs. Returns float32
        [ticks, cars + 1, 18]. Override to batch segments or to take level A's full state (`req["start"]["full"]`
        names byte ranges of `full`)."""
        a = self.new_arena(req["cars"], req.get("map", "Stadium_P"))
        self.set_state(a, req["start"])
        n = int(req["ticks"])
        out = np.zeros((n, len(req["cars"]) + 1, BODY), np.float32)
        timed = frames is not None and self.caps.get("frame_timing")
        t = 0
        while t < n:
            k = max(1, min(int(frames[t, 2]), n - t)) if timed else 1
            if timed:
                out[t:t + k] = self.step_frame(a, inputs[t:t + k], float(frames[t, 1]), float(frames[t, 0]))
            else:
                self.step(a, inputs[t], 1, None if frames is None else frames[t])
                out[t] = self.bodies(a)
            t += k
        self.close_arena(a)
        return out


def _load(path: str) -> Adapter:
    p = Path(path).resolve()
    sys.path.insert(0, str(p.parent))
    spec = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    found = [v for v in vars(mod).values() if inspect.isclass(v) and issubclass(v, Adapter) and v is not Adapter]
    if len(found) != 1:
        raise SystemExit(f"{path}: expected one Adapter subclass, found {len(found)}")
    return found[0]()


def serve(adapter: Adapter) -> None:
    inp, out = protocol.stdio()
    arenas: dict[int, object] = {}
    next_id = 0
    while True:
        try:
            req, payload = protocol.recv(inp)
        except EOFError:
            return
        op = req.get("op")
        try:
            if op == "hello":
                protocol.send(out, {"name": adapter.name, "version": adapter.version, "about": adapter.about,
                                    "device": adapter.device, "units": adapter.units, "caps": adapter.caps, "choices": adapter.choices,
                                    "protocol": protocol.PROTOCOL})
            elif op == "setup":
                adapter.setup(req.get("env", {}))
                protocol.send(out, {})
            elif op == "speed":
                job = req["job"]
                n = job["arenas"] * job["decisions"] * len(job["cars"])
                ctl = np.frombuffer(payload[:n * 32], np.float32).reshape(job["arenas"], job["decisions"], len(job["cars"]), 8)
                protocol.send(out, adapter.speed(job, req["starts"], ctl))
            elif op == "segment":
                n, c = int(req["ticks"]), len(req["cars"])
                need = n * c * TICK_CAR * 4
                inputs = np.frombuffer(payload[:need], np.float32).reshape(n, c, TICK_CAR)
                frames = None
                if req.get("frames"):
                    frames = np.frombuffer(payload[need:need + n * TICK_FRAME * 4], np.float32).reshape(n, TICK_FRAME)
                    need += n * TICK_FRAME * 4
                res = adapter.segment(req, inputs, frames, payload[need:])
                protocol.send(out, {}, np.ascontiguousarray(res, np.float32).tobytes())
            elif op == "new":
                arenas[next_id] = adapter.new_arena(req["cars"], req.get("map", "Stadium_P"))
                protocol.send(out, {"arena": next_id})
                next_id += 1
            elif op == "set":
                adapter.set_state(arenas[req["arena"]], req["state"])
                protocol.send(out, {})
            elif op == "step":
                a = arenas[req["arena"]]
                c = int(req["cars"])
                inputs = np.frombuffer(payload, np.float32).reshape(c, TICK_CAR)
                adapter.step(a, inputs, int(req.get("ticks", 1)))
                protocol.send(out, {}, np.ascontiguousarray(adapter.bodies(a), np.float32).tobytes())
            elif op == "bodies":
                protocol.send(out, {}, np.ascontiguousarray(adapter.bodies(arenas[req["arena"]]), np.float32).tobytes())
            elif op == "close":
                adapter.close_arena(arenas.pop(req["arena"]))
                protocol.send(out, {})
            elif op == "game":                    # a whole recorded game (simbench.game): an optional `Adapter.game`
                if not hasattr(adapter, "game"):
                    protocol.send(out, {"error": "this adapter has no whole-game replay (Adapter.game)"})
                else:
                    from .game import unpack
                    res = adapter.game(req, unpack(req, payload))
                    protocol.send(out, {}, np.ascontiguousarray(res, np.float32).tobytes())
            elif op == "quit":
                protocol.send(out, {})
                return
            else:
                protocol.send(out, {"error": f"unknown request {op!r}"})
        except Exception as e:  # noqa: BLE001  (the worker reports and keeps serving)
            protocol.send(out, {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]})


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m simbench.kit path/to/adapter.py")
    from simbench import kit as _kit          # the adapter subclasses simbench.kit.Adapter, not this __main__ copy
    _kit.serve(_kit._load(sys.argv[1]))
