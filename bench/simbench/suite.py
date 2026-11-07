"""Levels A and B on the full-state suite's recordings (github.com/thecosmicskye/RocketGPU-fullstate-suite).

You download the recordings yourself (its README); this file only reads them. It is our own scorer, written from the
suite's README (the protocol and the metric definitions), and imports nothing of the suite's.

Each segment: a fresh arena with the recording's cars, the segment's start state as the level gives it, then the game's
recorded inputs for up to 120 ticks (controls, whether each car may drive, the boost pad pickups the game credited,
each frame's render alpha and duration). Every tick after a car spawns and outside held (all cars frozen) ticks, every
car and the ball are compared with the game's own state:

  exact body-ticks   position, velocity, angular velocity and the rotation matrix bit-equal to the game's (Bullet units)
  exact segments     every scored body-tick of the segment exact
  segment pass       car-ticks whose normalized error is under 1: sqrt of the sum of squares of the car's and the ball's
                     (position / 2 uu, velocity / 3 uu/s, angular velocity / 0.05 rad/s, forward / 0.05, up / 0.05)
  mean error k       that error, capped at 10, averaged over the car-ticks k ticks into their segments (k = 1, 30, 120)

Levels: A gives the full start state (the suite's `s.*` arrays) to adapters that can take it; B gives the public state,
RocketSim v3's whole CarState (the suite's "public" rows); its variant Bp gives only what an RLBot packet lists.
"""
from __future__ import annotations

import concurrent.futures
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np

from . import sims
from .client import WorkerError
from .state import packet_of

F32 = np.float32
KS = (1, 30, 120)
CAR_F = 61          # per car in a public row: preset, team, then RocketSim v3's 59 CarState numbers
PRESETS = {"OCTANE": "Octane", "DOMINUS": "Dominus", "PLANK": "Plank", "BREAKOUT": "Breakout", "HYBRID": "Hybrid",
           "MERC": "Merc", "PSYCLOPS": "Psyclops"}
LEVEL_NAME = {"A": "full", "B": "v3", "Bp": "packet"}


def _v(r, i):
    return [float(r[i]), float(r[i + 1]), float(r[i + 2])]


def public_state(row, nc: int, tick: int, packet_only: bool) -> dict:
    """A suite public row -> simbench state (RocketSim v3's CarState fields, or only the packet's)."""
    cars = []
    for k in range(nc):
        r = row[k * CAR_F + 2:(k + 1) * CAR_F]
        c = {"pos": _v(r, 0), "fwd": _v(r, 3), "right": _v(r, 6), "up": _v(r, 9), "vel": _v(r, 12), "ang_vel": _v(r, 15),
             "prev_controls": [float(x) for x in r[18:26]], "has_jumped": bool(r[26]), "has_double_jumped": bool(r[27]),
             "has_flipped": bool(r[28]), "flip_rel_torque": _v(r, 29), "jump_ticks": int(r[32]), "flip_time": float(r[33]),
             "is_flipping": bool(r[34]), "is_jumping": bool(r[35]), "jump_hold_broken": bool(r[36]), "air_time": float(r[37]),
             "air_time_since_jump": float(r[38]), "boost": float(r[39]), "handbrake_val": float(r[40]),
             "is_supersonic": bool(r[41]), "supersonic_grace_timer": float(r[42]), "time_since_boosted": float(r[43]),
             "boosting_time": float(r[44]), "is_boosting": bool(r[45]), "is_auto_flipping": bool(r[46]),
             "auto_flip_timer": float(r[47]), "auto_flip_torque_scale": float(r[48]), "bump_cooldown_timer": float(r[49]),
             "bump_victim": int(r[50]), "last_extra_hit_tick": int(r[51]), "is_demoed": bool(r[52]),
             "demo_respawn_timer": float(r[53]), "on_ground": bool(r[54]), "wheels_contact": [bool(x) for x in r[55:59]]}
        if packet_only:
            c = packet_of(c)
        cars.append(c)
    b = row[nc * CAR_F:nc * CAR_F + 19]
    ball = {"pos": _v(b, 0), "fwd": _v(b, 3), "right": _v(b, 6), "up": _v(b, 9), "vel": _v(b, 12), "ang_vel": _v(b, 15)}
    if not packet_only:
        ball["asleep"] = bool(b[18])
    at = nc * CAR_F + 19
    n_pads = int(row[at])
    pads = [[float(row[at + 1 + 3 * i]), float(row[at + 2 + 3 * i]), float(row[at + 3 + 3 * i])] for i in range(n_pads)]
    return {"cars": cars, "ball": ball, "pads": pads, "tick": tick, "pads_from_recording": True}


def full_payload(z, keys, n: int):
    """The segment's full state: {field: [offset, length]} and the bytes."""
    index, parts, at = {}, [], 0
    for k in keys:
        b = np.ascontiguousarray(z[k][n]).tobytes()
        index[k[2:]] = [at, len(b)]
        parts.append(b)
        at += len(b)
    return index, b"".join(parts)


def to_truth(out: np.ndarray, units: str) -> np.ndarray:
    """[ticks, bodies, 18] in the protocol's order -> the suite's truth layout (Bullet units; rotation by rows)."""
    t = np.empty_like(out)
    k = F32(50.0) if units == "uu" else None
    t[..., 0:6] = out[..., 0:6] / k if k is not None else out[..., 0:6]
    t[..., 6:9] = out[..., 6:9]
    fwd, right, up = out[..., 9:12], out[..., 12:15], out[..., 15:18]
    for row in range(3):
        t[..., 9 + 3 * row] = fwd[..., row]
        t[..., 10 + 3 * row] = right[..., row]
        t[..., 11 + 3 * row] = up[..., row]
    return t


def errors(sim: np.ndarray, tru: np.ndarray, nc: int) -> np.ndarray:
    """Normalized error per car-tick [ticks, cars] (car's terms and the ball's, sqrt of the sum of squares)."""
    s, t = sim.astype(np.float64), tru.astype(np.float64)

    def terms(a, b):
        sb, tb = a[..., 9:18].reshape(*a.shape[:-1], 3, 3), b[..., 9:18].reshape(*b.shape[:-1], 3, 3)
        return np.stack([np.linalg.norm(a[..., 0:3] - b[..., 0:3], axis=-1) * 50.0 / 2.0,
                         np.linalg.norm(a[..., 3:6] - b[..., 3:6], axis=-1) * 50.0 / 3.0,
                         np.linalg.norm(a[..., 6:9] - b[..., 6:9], axis=-1) / 0.05,
                         np.linalg.norm(sb[..., :, 0] - tb[..., :, 0], axis=-1) / 0.05,
                         np.linalg.norm(sb[..., :, 2] - tb[..., :, 2], axis=-1) / 0.05], -1)
    cars = terms(s[:, :nc], t[:, :nc])
    absent_s = ~np.any(sim[:, :nc, :9] != 0, axis=-1)
    absent_t = ~np.any(tru[:, :nc, :9] != 0, axis=-1)
    cars = np.where((absent_s & absent_t)[..., None], 0.0, cars)
    cars = np.where((absent_s ^ absent_t)[..., None], 1e3, cars)
    ball = terms(s[:, nc], t[:, nc])
    return np.sqrt((cars ** 2).sum(-1) + (ball ** 2).sum(-1)[:, None])


def frame_rows(frac, start_dt, i0: int, ticks: int) -> np.ndarray:
    """Per tick [render alpha, frame duration, physics ticks in the frame (on its first tick; 0 after)]. A tick whose
    recorded `frac` is -1 shares its rendered frame with the next tick; the frame's duration is its recorded `start_dt`
    when there is one, else 1/120 s a tick; the render alpha is the frame's last `frac` (else the one before)."""
    out = np.zeros((ticks, 5), F32)
    out[:, 3] = frac[i0:i0 + ticks]
    out[:, 4] = start_dt[i0:i0 + ticks]
    alpha = float(frac[i0 - 1]) if i0 > 0 and frac[i0 - 1] != -1.0 and np.isfinite(frac[i0 - 1]) else 0.5
    t = 0
    while t < ticks:
        j = t
        while j < ticks - 1 and frac[i0 + j] == -1.0:
            j += 1
        n = j - t + 1
        last = float(frac[i0 + j])
        if last != -1.0 and math.isfinite(last):
            alpha = last
        dt = float(start_dt[i0 + t]) if start_dt[i0 + t] > 0 else n / 120.0
        out[t:j + 1, 0] = alpha
        out[t:j + 1, 1] = dt
        out[t, 2] = n
        t = j + 1
    return out


def respawn_spots(truth, i0: int, ticks: int, nc: int, start: dict) -> list:
    """A demolished car's respawn spot is a choice the game made (the suite hands it to its simulators): read here from
    where the car's body first shows in the recording, to the spawn table's 128 uu grid, facing the body's way."""
    out = []
    for k in range(nc):
        spot = None
        if start["cars"][k].get("is_demoed"):
            for j in range(i0, min(i0 + ticks, len(truth))):
                row = truth[j][k]
                if np.any(row[:3]):
                    x, y = float(row[0]) * 50.0, float(row[1]) * 50.0
                    yaw = int(round(math.atan2(float(row[12]), float(row[9])) / math.pi * 32768.0 / 8192.0)) * 8192
                    spot = [round(x / 128.0) * 128.0, round(y / 128.0) * 128.0, float(yaw)]
                    break
        out.append(spot)
    return out


def score_recording(holder, path: Path, level: str, units: str) -> dict:
    """`holder`: [worker]; a worker that dies is replaced in it."""
    worker = holder[0]
    z = np.load(path)
    meta = json.loads(z["meta"].tobytes())
    nc = meta["ncars"]
    at = {int(f): i for i, f in enumerate(z["frame"])}
    credits = {}
    for j, s_, g in zip(z["credit_tick"], z["credit_slot"], z["credit_gain"]):
        credits.setdefault(int(j), []).append((int(s_), float(g)))
    ctl, drv, held, truth = z["ctl"], z["drv"], z["held"], z["truth"]
    frac, start_dt = z["frac"].astype(F32), z["start_dt"].astype(F32)
    pub = z["public"]
    full_keys = sorted(k for k in z.keys() if k.startswith("s.")) if level == "A" else []
    fulls = {k: z[k] for k in full_keys}           # (an npz decompresses an array on every access: read each once)
    cars_spec = [{"preset": PRESETS.get(str(p).upper(), "Octane"), "team": 0, "body": None} for p in meta["public_cars"]]
    r = {"name": meta["name"], "passed": 0, "total": 0, "at_k": {str(k): [0.0, 0] for k in KS}, "exact_bt": 0,
         "body_ticks": 0, "exact_segs": 0, "segs": 0, "failed_segs": 0, "first_error": None}
    for n, seg in enumerate(meta["segments"]):
        log = meta["cars"][seg["cars"]]
        cars = []
        for k, (_, args, _kw) in enumerate(log):
            c = dict(cars_spec[k] if k < len(cars_spec) else cars_spec[-1])
            c["team"] = int(args[0]["value"])
            c["config"] = {key: v for key, v in args[1].items() if not key.startswith("__")}
            cars.append(c)
        ticks = int(seg["ticks"])
        i0 = at[seg["start"]]
        idx = np.arange(i0, i0 + ticks)
        start = public_state(pub[n], nc, int(seg["start"]), level == "Bp")
        # every level gives the inputs; the first tick's, here too, so that an adapter can guess what a car was holding
        start["next_controls"] = [[float(x) for x in ctl[i0, k]] for k in range(nc)]
        live = [k for k in range(nc) if not start["cars"][k].get("is_demoed")]
        inp = np.zeros((ticks, nc, 10), F32)
        inp[:, :, :8] = ctl[idx]
        inp[:, :, 8] = drv[idx]
        for t, i in enumerate(idx):
            for slot, g in credits.get(int(i), ()):
                if len(live) < nc and slot < len(live):
                    slot = live[slot]
                inp[t, slot, 9] = g
        frames = frame_rows(frac, start_dt, i0, ticks)
        req = {"op": "segment", "cars": cars, "map": "Stadium_P", "ticks": ticks, "frames": True, "start": start,
               "spawn": bool(seg["spawn"]), "timer_hold": bool(seg["timer_hold"]), "level": LEVEL_NAME[level]}
        payload = inp.tobytes() + frames.tobytes()
        if level == "A":
            index, blob = full_payload(fulls, full_keys, n)
            req["start"] = dict(start, full=index, timer_hold=bool(seg["timer_hold"]),
                                respawn_spots=respawn_spots(truth, i0, ticks, nc, start))
            req["dt0"] = float(start_dt[i0]) if start_dt[i0] > 0 else 1.0 / 120.0
            req["calls"] = seg.get("calls", [])
            req["mutator"] = seg.get("mutator")
            req["ids"] = seg.get("ids", [])
            payload += blob
        try:
            _, data = worker.request(req, payload)
        except WorkerError as e:
            r["failed_segs"] += 1
            if r["first_error"] is None:
                r["first_error"] = f"segment {n}: " + str(e).splitlines()[0][:300]
            if worker.p.poll() is not None:          # the worker died: a fresh one for the rest
                worker = holder[0] = worker.restart()
            continue
        out = np.frombuffer(data, F32).reshape(ticks, nc + 1, 18)
        sim = to_truth(out, units)
        tru = truth[idx].astype(F32)
        scored = ~held[idx]
        if seg["spawn"]:
            scored[0] = False
        if not scored.any():
            continue
        ex = (sim.view(np.uint32) == tru.view(np.uint32)).all(-1)            # [ticks, bodies]
        if os.environ.get("SIMBENCH_DEBUG") and not ex[scored].all() and r.get("_dbg", 0) < 6:
            r["_dbg"] = r.get("_dbg", 0) + 1
            t, b = np.argwhere(~ex & scored[:, None])[0]
            d = sim[t, b].astype(np.float64) - tru[t, b]
            print(f"    {meta['name']} seg {n} start {seg['start']} spawn {seg['spawn']} hold {seg['timer_hold']}: tick {t} body {b}: "
                  + " ".join(f"{k}:{d[k]:.3g}" for k in range(18) if d[k] != 0), flush=True)
        ex = ex[scored]
        r["body_ticks"] += ex.size
        r["exact_bt"] += int(ex.sum())
        r["segs"] += 1
        r["exact_segs"] += int(ex.all())
        e = errors(sim, tru, nc)[scored]
        r["total"] += e.size
        r["passed"] += int((e < 1.0).sum())
        ks = np.nonzero(scored)[0] + 1
        for k in KS:
            hit = ks == k
            if hit.any():
                v = np.minimum(e[hit], 10.0)
                r["at_k"][str(k)][0] += float(v.sum())
                r["at_k"][str(k)][1] += int(v.size)
    return r


def summarize(rs: list) -> dict:
    P, N = sum(x["passed"] for x in rs), sum(x["total"] for x in rs)
    EB, BT = sum(x["exact_bt"] for x in rs), sum(x["body_ticks"] for x in rs)
    ES, SG = sum(x["exact_segs"] for x in rs), sum(x["segs"] for x in rs)
    mk = {}
    for k in KS:
        s = sum(x["at_k"][str(k)][0] for x in rs)
        c = sum(x["at_k"][str(k)][1] for x in rs)
        mk[str(k)] = s / c if c else None
    return {"recordings": len(rs), "body_ticks": BT, "exact_body_ticks": 100.0 * EB / max(BT, 1), "segments": SG,
            "exact_segments": 100.0 * ES / max(SG, 1), "car_ticks": N, "segment_pass": 100.0 * P / max(N, 1),
            "mean_error": mk, "failed_segments": sum(x["failed_segs"] for x in rs)}


def run(names, setup_env, data: Path, levels, splits, limit, jobs, log_dir, emit):
    data = Path(data)
    folder = data / "data" if (data / "data").is_dir() else data
    man_path = next((p for p in (data / "manifest.json", data.parent / "manifest.json", folder / "manifest.json") if p.is_file()), None)
    if man_path is None:
        raise SystemExit(f"no manifest.json next to {data}: point --suite-data at the suite's checkout (with data/ inside)")
    man = json.loads(man_path.read_text())["recordings"]
    split_of = {x["name"]: x["split"] for x in man}
    want = None if splits == "all" else set(splits.split(","))
    recs = [x["name"] for x in man if (want is None or x["split"] in want) and (folder / (x["name"] + ".npz")).is_file()]
    if limit:
        recs = recs[:limit]
    levels = [x.strip() for x in (levels or "A,B,Bp").split(",") if x.strip() in LEVEL_NAME]
    for name in names:
        probe, why = sims.try_start(name, setup_env, Path(log_dir))
        if probe is None:
            for lv in levels:
                emit({"kind": "accuracy", "level": lv, "suite": "full-state suite", "sim": sims.label(name), "status": "not run", "why": why})
            continue
        hello = probe.hello
        probe.close()
        for lv in levels:
            if LEVEL_NAME[lv] not in hello.get("caps", {}).get("levels", []):
                emit({"kind": "accuracy", "level": lv, "suite": "full-state suite", "sim": sims.label(name), "status": "not supported",
                      "why": f"its adapter can't take the {LEVEL_NAME[lv]} start state", "version": hello.get("version")})
                print(f"  {sims.label(name):14s} level {lv}: not supported", flush=True)
                continue
            t0 = time.time()
            res = []
            lock = threading.Lock()
            pool = []

            def work(rec_name):
                with lock:
                    w = pool.pop() if pool else None
                if w is None:
                    w = sims.start(name, setup_env, Path(log_dir))
                holder = [w]
                try:
                    return score_recording(holder, folder / (rec_name + ".npz"), lv, hello.get("units", "uu"))
                finally:
                    with lock:
                        pool.append(holder[0])

            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as ex:
                for r in ex.map(work, recs):
                    res.append(r)
            for w in pool:
                w.close()
            per = {x["name"]: [round(100.0 * x["exact_bt"] / max(x["body_ticks"], 1), 4), round(100.0 * x["passed"] / max(x["total"], 1), 4),
                               x["failed_segs"]] for x in res}
            by_split = {}
            for x in res:
                by_split.setdefault(split_of.get(x["name"], "?"), []).append(x)
            summary = {"all": summarize(res)} | {s: summarize(v) for s, v in by_split.items()}
            s = summary["all"]
            emit({"kind": "accuracy", "level": lv, "suite": "full-state suite", "sim": sims.label(name), "status": "ok",
                  "version": hello.get("version"), "about": hello.get("about"), "choices": hello.get("choices", []),
                  "summary": summary, "seconds": round(time.time() - t0), "recordings": [x["name"] for x in res],
                  "per_recording": per,
                  "first_error": next((x["first_error"] for x in res if x["first_error"]), None)})
            print(f"  {sims.label(name):14s} level {lv}: exact body-ticks {s['exact_body_ticks']:.4f}%  exact segments "
                  f"{s['exact_segments']:.3f}%  pass {s['segment_pass']:.3f}%  k1/k30/k120 "
                  f"{s['mean_error']['1'] or 0:.4f}/{s['mean_error']['30'] or 0:.4f}/{s['mean_error']['120'] or 0:.4f}  "
                  f"({s['recordings']} recordings, {s['body_ticks']:,} body-ticks, {s['failed_segments']} failed segments, "
                  f"{time.time() - t0:.0f} s)", flush=True)
