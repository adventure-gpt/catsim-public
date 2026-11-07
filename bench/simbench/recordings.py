"""Levels C and D: whole recorded games, open loop from the first kickoff to the end.

Our own recordings (bench/recordings/, and the larger set from this repository's release, unpacked into the folder
SIMBENCH_RECORDINGS names): real RLBot matches recorded from the game, each with its first kickoff's start, every frame's
timing and inputs, and the packets to compare. RECORDINGS.json says what each one is.

  Level C   Given: the start at the first kickoff, every frame's inputs and timing, and the game's random CHOICES
            (each later kickoff's spots and each respawn's spot, from the recording).
            Scored: every compared packet row exact to the end, and the first row that isn't.
  Level D   Given: the same start, inputs and timing, and the random stream's starting state instead of the choices:
            the simulator predicts every kickoff and respawn spot itself.
            Scored: the same.

A simulator whose adapter has no whole-game replay (no `game` request) runs level C through simbench.flow: the
harness runs the match flow (kickoffs, countdowns, restarts, respawns) and the simulator the physics, every assist
named. Level D tests predicting the game's own random draws from its random stream, which only a simulator that runs
the game's flow can do; handing the draws over is level C.

The RocketGPU full-state suite's recordings could run whole too (level C only: they hold no random-stream state); that needs the
suite's full start state in the "game" request and isn't built yet.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from . import flow, game, sims
from .state import SHORT

HERE = Path(__file__).resolve().parents[1] / "recordings"


def recording_files(extra: str | None = None) -> list[Path]:
    out = sorted(HERE.glob("*.npz"))
    folder = extra or os.environ.get("SIMBENCH_RECORDINGS")
    if folder and Path(folder).is_dir():
        have = {p.name for p in out}
        out += sorted(p for p in Path(folder).glob("*.npz") if p.name not in have)
    return out


def run(names, setup_env, levels, limit, jobs, out_dir, emit):
    levels = [x.strip() for x in (levels or "C,D,Cf,Df").split(",") if x.strip() in ("C", "D", "Cf", "Df")]
    files = recording_files()
    have = {f.name for f in files}
    files += [f for f in recording_files(os.environ.get("SIMBENCH_FLOAT_RECORDINGS")) if f.name not in have]
    by_level = {lv: [f for f in files if lv in game.levels(game.load_meta(f))] for lv in levels}
    for lv in levels:
        if limit:
            by_level[lv] = by_level[lv][:limit]
    for name in names:
        label = sims.label(name)
        probe, why = sims.try_start(name, setup_env, Path(out_dir))
        if probe is None:
            for lv in levels:
                emit({"kind": "accuracy", "level": lv, "suite": "our recordings", "sim": label, "status": "not run", "why": why})
            continue
        hello = probe.hello
        probe.close()
        caps = hello.get("caps", {})
        units = hello.get("units", "uu")
        own_flow = bool(caps.get("game_flow") and caps.get("spot_choices"))
        for lv in levels:
            if lv in ("D", "Df") and not (caps.get("game_flow") and caps.get("rng_stream")):
                emit({"kind": "accuracy", "level": lv, "suite": "our recordings", "sim": label, "status": "not supported",
                      "version": hello.get("version"), "needs": "rng_stream",
                      "why": "these games test predicting the game's own random draws from its random stream, which this "
                             "simulator doesn't model; handing it the draws instead is the whole-games run with the spots given"})
                continue
            harness_flow = lv in ("C", "Cf") and not own_flow
            floats = lv in ("Cf", "Df")
            if not by_level[lv]:
                emit({"kind": "accuracy", "level": lv, "suite": "our recordings", "sim": label, "status": "not run",
                      "why": "no recordings for this run found (bench/recordings, SIMBENCH_RECORDINGS, SIMBENCH_FLOAT_RECORDINGS)"})
                continue
            t0 = time.time()
            lock = threading.Lock()
            pool = []

            def work(path):
                with lock:
                    w = pool.pop() if pool else None
                if w is None:
                    w = sims.start(name, setup_env, Path(out_dir))
                try:
                    rec = game.load(path)
                    try:
                        J = len(rec["truth_phase"])
                        fview = np.zeros((len(rec["float_frame"]), int(rec["meta"]["cars"]) + 1, 19), np.float32) if floats else None
                        if harness_flow:
                            view = flow.run(w, rec, units, fview, births=flow.birth(rec, setup_env.get("catsim_data")))
                            r = flow.score(rec, view)
                        else:
                            req, payload = game.pack(rec, lv, float_rows=floats)
                            _, data = w.request(req, payload)
                            flat = np.frombuffer(data, np.float32)
                            nview = J * game.width(int(rec["meta"]["cars"]))
                            view = flat[:nview].astype(np.float64).reshape(J, -1)
                            r = game.score(rec, view.astype(np.float32))
                            if floats:
                                fview = flat[nview:].reshape(fview.shape)
                        if floats:
                            r = dict(game.score_float(rec, fview, units if harness_flow else "bt"), packet=r, bodies_only=harness_flow)
                    except Exception as e:  # noqa: BLE001
                        if w.p.poll() is not None:
                            w = sims.start(name, setup_env, Path(out_dir))
                        return {"name": rec["meta"]["name"], "error": str(e).splitlines()[0][:300]}
                    r["errors_after_kickoff"] = flow.kickoff_errors(rec, view)
                    r.update(name=rec["meta"]["name"], mode=rec["meta"]["mode"], map=rec["meta"]["map"],
                             seconds=rec["meta"]["seconds"], ticks=rec["meta"]["physics_ticks"])
                    return r
                finally:
                    with lock:
                        pool.append(w)

            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(jobs, len(by_level[lv])))) as ex:
                res = list(ex.map(work, by_level[lv]))
            for w in pool:
                w.close()
            ok = [r for r in res if "error" not in r]
            summary = {"recordings": len(res), "failed": len(res) - len(ok),
                       "exact_to_the_end": sum(r["exact_to_the_end"] for r in ok),
                       "compared_rows": sum(r["compared_rows"] for r in ok), "exact_rows": sum(r["exact_rows"] for r in ok),
                       "ticks": sum(r["ticks"] for r in ok)}
            summary["exact_pct"] = 100.0 * summary["exact_rows"] / max(summary["compared_rows"], 1)
            # the time to the first drift (a recording exact to the end counts its whole length), and the errors after kickoffs
            drift = [(r["first_drift"]["seconds"] if r["first_drift"] and r["first_drift"]["seconds"] is not None else r["seconds"]) for r in ok]
            if drift:
                q = np.percentile(drift, [0, 25, 50, 75, 100])
                summary["first_drift_s"] = {"min": float(q[0]), "q1": float(q[1]), "median": float(q[2]), "q3": float(q[3]), "max": float(q[4])}
            err = {}
            for h in flow.HORIZONS:
                s_ = sum(r["errors_after_kickoff"][str(h)][0] for r in ok)
                c_ = sum(r["errors_after_kickoff"][str(h)][1] for r in ok)
                err[str(h)] = s_ / c_ if c_ else None
            summary["error_after_kickoff"] = err
            summary["bodies_only"] = harness_flow
            base = ASSISTS[lv.rstrip("f")]
            assists = (base if not harness_flow else base[:1] + flow.ASSISTS)
            emit({"kind": "accuracy", "level": lv, "suite": "our recordings", "sim": label, "status": "ok",
                  "version": hello.get("version"), "about": hello.get("about"), "choices": hello.get("choices", []),
                  "flow": "the harness's (simbench.flow)" if harness_flow else "the simulator's own",
                  "assists": assists, "summary": summary, "recordings": res, "seconds": round(time.time() - t0)})
            e = summary["error_after_kickoff"]
            fmt = lambda x: "-" if x is None else f"{x:.3f}"
            print(f"  {label:14s} {SHORT[lv]}{' (harness flow, bodies only)' if harness_flow else ''}: "
                  f"{summary['exact_to_the_end']}/{len(res)} recordings exact to the end, "
                  f"{summary['exact_rows']:,}/{summary['compared_rows']:,} compared rows exact ({summary['exact_pct']:.3f}%), "
                  f"first drift median {summary.get('first_drift_s', {}).get('median', 0):.1f} s, "
                  f"error 1/5/30 s after kickoffs {fmt(e['1.0'])}/{fmt(e['5.0'])}/{fmt(e['30.0'])}, {time.time() - t0:.0f} s", flush=True)
            for r in res:
                if "error" in r:
                    print(f"      {r['name']}: failed ({r['error']})", flush=True)
                else:
                    f = r["first_drift"]
                    print(f"      {r['name']:28s} {r['mode']} {r['map']:10s} {r['seconds']:6.1f} s: "
                          + ("exact to the end" if f is None else
                             f"exact for {f.get('compared_row', f.get('row', 0)):,} of {f['of']:,} compared rows, first drift at tick {f['tick'] or 0:,} "
                             f"({f['seconds']} s, {f['phase']}: {', '.join(f['bodies']) or ('pads' if f.get('pads') else 'boost')})")
                          + (f"; ball rotation {r['ball_rotation_exact']}/{r['ball_rotation_compared']}" if 'ball_rotation_exact' in r else ''), flush=True)


# Every assist each level gives, named (what a simulator is handed beyond the start, the inputs and the frame timing).
ASSISTS = {
    "C": ["the first kickoff's start: each car's spot and how many ticks before the first packet its body was born, the "
          "broadphase handles, the countdown's timer as it stood",
          "each later kickoff's spots and each demolition respawn's spot, from the recording (the game's random choices)",
          "the end-of-match rule: a match that ends in a frame publishes the state before it"],
    "D": ["the same start", "the game's random-number state and clock at the start, the kickoff spawn spots it chooses "
          "from, and what draws from the stream (the cars' items' animated parts, from the data folder's items.txt): "
          "the simulator predicts every kickoff and respawn spot itself",
          "the end-of-match rule as with the spots given"],
}
