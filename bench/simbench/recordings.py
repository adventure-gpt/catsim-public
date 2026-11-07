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

The full-state suite's recordings could run whole too (level C only: they hold no random-stream state); that needs the
suite's full start state in the "game" request and isn't built yet.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import threading
import time
from pathlib import Path

from . import game, sims

HERE = Path(__file__).resolve().parents[1] / "recordings"


def recording_files(extra: str | None = None) -> list[Path]:
    out = sorted(HERE.glob("*.npz"))
    folder = extra or os.environ.get("SIMBENCH_RECORDINGS")
    if folder and Path(folder).is_dir():
        have = {p.name for p in out}
        out += sorted(p for p in Path(folder).glob("*.npz") if p.name not in have)
    return out


def run(names, setup_env, levels, limit, jobs, out_dir, emit):
    levels = [x.strip() for x in (levels or "C,D").split(",") if x.strip() in ("C", "D")]
    files = recording_files()
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
        for lv in levels:
            need = "game_flow" if lv == "C" else "rng_stream"
            if not caps.get("game_flow") or not caps.get("spot_choices" if lv == "C" else "rng_stream"):
                why = ("it doesn't run the game's own round flow over a whole game" if not caps.get("game_flow")
                       else "it doesn't take the game's random-number stream" if lv == "D" else "it doesn't take spot choices")
                emit({"kind": "accuracy", "level": lv, "suite": "our recordings", "sim": label, "status": "not supported",
                      "version": hello.get("version"), "why": f"level {lv}: {why}", "needs": need})
                continue
            if not by_level[lv]:
                emit({"kind": "accuracy", "level": lv, "suite": "our recordings", "sim": label, "status": "not run",
                      "why": "no recordings of this level found (bench/recordings, SIMBENCH_RECORDINGS)"})
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
                    req, payload = game.pack(rec, lv)
                    try:
                        _, data = w.request(req, payload)
                    except Exception as e:  # noqa: BLE001
                        return {"name": rec["meta"]["name"], "error": str(e).splitlines()[0][:300]}
                    import numpy as np
                    r = game.score(rec, np.frombuffer(data, np.float32))
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
            emit({"kind": "accuracy", "level": lv, "suite": "our recordings", "sim": label, "status": "ok",
                  "version": hello.get("version"), "about": hello.get("about"), "choices": hello.get("choices", []),
                  "assists": ASSISTS[lv], "summary": summary, "recordings": res, "seconds": round(time.time() - t0)})
            print(f"  {label:14s} level {lv}: {summary['exact_to_the_end']}/{len(res)} recordings exact to the end, "
                  f"{summary['exact_rows']:,}/{summary['compared_rows']:,} compared rows exact ({summary['exact_pct']:.3f}%), "
                  f"{summary['ticks']:,} ticks, {time.time() - t0:.0f} s", flush=True)
            for r in res:
                if "error" in r:
                    print(f"      {r['name']}: failed ({r['error']})", flush=True)
                else:
                    f = r["first_drift"]
                    print(f"      {r['name']:28s} {r['mode']} {r['map']:10s} {r['seconds']:6.1f} s: "
                          + ("exact to the end" if f is None else
                             f"exact for {f['compared_row']:,} of {f['of']:,} compared rows, first drift at tick {f['tick']:,} "
                             f"({f['seconds']} s, {f['phase']}: {', '.join(f['bodies']) or ('pads' if f['pads'] else 'boost')})")
                          + f"; ball rotation {r['ball_rotation_exact']}/{r['ball_rotation_compared']}", flush=True)


# Every assist each level gives, named (what a simulator is handed beyond the start, the inputs and the frame timing).
ASSISTS = {
    "C": ["the first kickoff's start: each car's spot and how many ticks before the first packet its body was born, the "
          "broadphase handles, the countdown's timer as it stood",
          "each later kickoff's spots and each demolition respawn's spot, from the recording (the game's random choices)",
          "the end-of-match rule: a match that ends in a frame publishes the state before it"],
    "D": ["the same start", "the game's random-number state and clock at the start, the kickoff spawn spots it chooses "
          "from, and what draws from the stream (the cars' items' animated parts, from the data folder's items.txt): "
          "the simulator predicts every kickoff and respawn spot itself",
          "the end-of-match rule as in C"],
}
