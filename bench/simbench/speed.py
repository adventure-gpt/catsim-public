"""Speed: every simulator does the same work, and the harness times it the same way.

The work, identical for every simulator: arenas of Octanes (1v1, 2v2 or 3v3), half starting from a kickoff and half
from a random mid-game state (cars on the ground anywhere on the field, moving; the ball in the air, moving), all states
made by the harness from a fixed seed. Each car gets a random action from RLGym's 90-action table every 8 ticks
(kept with probability 0.7), the controls also made by the harness. One decision = set every car's controls, step 8
ticks, read every car's and the ball's state. Timed: from the first decision to the last; the start states are set
before the clock starts. Every simulator runs its own pad pickups, demolitions and so on. Goals don't reset anything;
the 8-second episodes rarely have one.

Reported: arena-ticks per second (arenas x ticks / seconds), the median of the rounds and their spread, and CPU-time per
arena-tick (threads x seconds / arena-ticks: the cost of one arena-tick on one thread). Rounds take the simulators in
turn, the order turning round each round, so slow drift in the machine's load falls on all of them alike.
"""
from __future__ import annotations

import json
import math
import statistics
import time

import numpy as np

from . import sims
from .client import WorkerError

KICKOFF = [(-2048.0, -2560.0, 0.25 * math.pi), (2048.0, -2560.0, 0.75 * math.pi), (-256.0, -3840.0, 0.5 * math.pi),
           (256.0, -3840.0, 0.5 * math.pi), (0.0, -4608.0, 0.5 * math.pi)]


def action_table() -> np.ndarray:
    """RLGym's 90-action lookup table: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]."""
    out = []
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    out.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == roll == jump == 0:
                            continue
                        handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                        out.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
    return np.asarray(out, np.float32)


def _car(x, y, yaw, speed, boost, team):
    if team == 1:
        x, y, yaw = -x, -y, yaw + math.pi
    c, s = math.cos(yaw), math.sin(yaw)
    return {"pos": [x, y, 17.0], "fwd": [c, s, 0.0], "right": [-s, c, 0.0], "up": [0.0, 0.0, 1.0],
            "vel": [c * speed, s * speed, 0.0], "ang_vel": [0.0, 0.0, 0.0], "boost": boost, "on_ground": True,
            "has_jumped": False, "has_double_jumped": False, "has_flipped": False, "is_supersonic": False, "is_demoed": False}


def starts(n_arenas: int, team_size: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_arenas):
        cars = []
        if k % 2 == 0:
            spots = rng.permutation(5)[:team_size]
            for team in (0, 1):
                for s in spots:
                    x, y, yaw = KICKOFF[s]
                    cars.append(_car(x, y, yaw, 0.0, 33.3, team))
            ball = {"pos": [0.0, 0.0, 93.15], "vel": [0.0, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0], "asleep": False}
        else:
            for team in (0, 1):
                for _ in range(team_size):
                    cars.append(_car(float(rng.uniform(-3500, 3500)), float(rng.uniform(-4500, 4500)),
                                     float(rng.uniform(-math.pi, math.pi)), float(rng.uniform(0, 1400)),
                                     float(rng.integers(0, 101)), 0))
                    cars[-1]["team"] = team
            ball = {"pos": [float(rng.uniform(-3000, 3000)), float(rng.uniform(-4000, 4000)), float(rng.uniform(150, 1500))],
                    "vel": [float(v) for v in rng.uniform(-1500, 1500, 3)], "ang_vel": [float(v) for v in rng.uniform(-3, 3, 3)],
                    "asleep": False}
        for c in cars:
            c.pop("team", None)
        out.append({"cars": cars, "ball": ball})
    return out


def controls(n_arenas: int, n_cars: int, decisions: int, seed: int):
    """(float32 [arenas, decisions, cars, 8] controls, uint8 [decisions, arenas, cars] the same as table rows)."""
    rng = np.random.default_rng(seed + 1)
    table = action_table()
    pick = rng.integers(0, len(table), (n_arenas, n_cars))
    out = np.empty((n_arenas, decisions, n_cars, 8), np.float32)
    rows = np.empty((decisions, n_arenas, n_cars), np.uint8)
    for d in range(decisions):
        change = rng.random((n_arenas, n_cars)) < 0.3
        pick = np.where(change, rng.integers(0, len(table), (n_arenas, n_cars)), pick)
        out[:, d] = table[pick]
        rows[d] = pick
    return out, rows


MODES = {"1v1": 1, "2v2": 2, "3v3": 3}


def configs(plan: str, max_threads: int, device: str = "cpu") -> list:
    """(mode, arenas, threads, decisions) to run. `plan`: "quick" (a few minutes) or "full". GPU simulators run the
    batch sizes GPUs are built for (no thread counts)."""
    full = plan == "full"
    out = []
    if device == "gpu":
        for mode in MODES:
            for arenas in ((1024, 4096, 16384) if full else (1024, 4096)):
                out.append((mode, arenas, 1, 120 if full else 30))
        return out
    threads = sorted({1, 4, max_threads} if full else {1, max_threads})
    threads = [t for t in threads if t <= max_threads]
    for mode in MODES:
        out.append((mode, 1, 1, 1500 if full else 300))            # one arena at a time
        for t in threads:
            out.append((mode, 96, t, 120 if full else 40))         # batched, by thread count
        out.append((mode, 1024, max_threads, 120 if full else 30))  # the training-style load point
    return out


def run(names: list, setup_env: dict, plan: str, rounds: int, max_threads: int, seed: int, log_dir, emit, only_device=None):
    """Run the plan, CPU simulators and GPU simulators apart (never in one table); `emit(record)` gets one record per
    simulator and configuration. A simulator whose thread count is fixed when it starts (caps "threads_at_setup") is
    started again for each thread count."""
    hello = {}
    for n in names:
        w, why = sims.try_start(n, setup_env, log_dir)
        if w is None:
            emit({"kind": "speed", "sim": sims.label(n), "status": "not run", "why": why})
            print(f"  {sims.label(n):14s}: not run ({why})", flush=True)
            continue
        hello[n] = w.hello
        w.close()
    for device in ("cpu", "gpu"):
        if only_device and device != only_device:
            continue
        group = [n for n in hello if hello[n].get("device", "cpu") == device]
        if group:
            print(f"  -- {device.upper()} simulators: {', '.join(sims.label(n) for n in group)}", flush=True)
            _run_group(group, hello, setup_env, plan, rounds, max_threads, seed, log_dir, emit, device)


def _run_group(names, hello, setup_env, plan, rounds, max_threads, seed, log_dir, emit, device):
    workers, started_with = {}, {}

    def worker(n, threads):
        fixed = hello[n].get("caps", {}).get("threads_at_setup")
        want = threads if fixed else None
        if n in workers and started_with[n] != want:
            workers.pop(n).close()
        if n not in workers:
            env = dict(setup_env, threads=threads) if fixed else setup_env
            workers[n] = sims.start(n, env, log_dir)
            started_with[n] = want
        return workers[n]

    for mode, arenas, threads, decisions in configs(plan, max_threads, device):
        ts = MODES[mode]
        st = starts(arenas, ts, seed)
        ctl, rows = controls(arenas, 2 * ts, decisions, seed)
        payload = np.ascontiguousarray(ctl).tobytes() + np.ascontiguousarray(rows).tobytes()
        cars = [{"preset": "Octane", "team": t} for t in (0, 1) for _ in range(ts)]
        job = {"mode": mode, "cars": cars, "arenas": arenas, "threads": threads, "decision_ticks": 8, "decisions": decisions,
               "warmup": 1, "repeats": 1, "seed": seed, "map": "Stadium_P"}
        samples = {n: [] for n in names}
        skip = {}
        for r in range(rounds):
            for n in (names if r % 2 == 0 else names[::-1]):
                if n in skip:
                    continue
                if threads > 1 and not hello[n].get("caps", {}).get("threads"):
                    skip[n] = "steps one arena at a time on one thread (no threaded path)"
                    continue
                try:
                    rep, _ = worker(n, threads).request({"op": "speed", "job": job, "starts": st}, payload)
                except WorkerError as e:
                    skip[n] = str(e).splitlines()[0]
                    continue
                samples[n].append((rep["seconds"][0], rep.get("threads", threads)))
        for n in names:
            h, name = hello[n], sims.label(n)
            if n in skip or not samples[n]:
                emit({"kind": "speed", "sim": name, "version": h.get("version"), "device": device, "mode": mode,
                      "arenas": arenas, "threads": threads, "status": "not supported" if "no threaded" in skip.get(n, "") else "failed",
                      "why": skip.get(n, "no samples")})
                continue
            ticks = arenas * decisions * 8
            tps = [ticks / s for s, _ in samples[n]]
            used = samples[n][0][1]
            emit({"kind": "speed", "sim": name, "version": h.get("version"), "about": h.get("about"), "device": device,
                  "choices": h.get("choices", []), "mode": mode, "arenas": arenas, "threads": threads, "threads_used": used,
                  "decisions": decisions, "rounds": len(tps), "arena_ticks_per_s": statistics.median(tps),
                  "min": min(tps), "max": max(tps), "cpu_us_per_arena_tick": used * 1e6 / statistics.median(tps),
                  "status": "ok"})
    for w in workers.values():
        w.close()
