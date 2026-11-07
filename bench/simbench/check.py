"""`python -m simbench check --sims mine=path/to/adapter.py`: a quick test of an adapter before the real benchmarks.

Starts the worker, prints what it says about itself, runs one short segment from a kickoff-like state (60 ticks of
driving and jumping), checks the bodies' shape and that they move and stay finite, runs the same segment again and says
whether the two runs are identical (determinism), and times a tiny speed job.
"""
from __future__ import annotations

import json

import numpy as np

from . import sims, speed
from .client import WorkerError


def main(names, env, out_dir) -> None:
    for name in names:
        print(f"== {name}")
        w, why = sims.try_start(name, env, out_dir)
        if w is None:
            print(f"  could not start: {why}")
            continue
        h = w.hello
        print(f"  {h.get('name')} {h.get('version')}  device {h.get('device')}  units {h.get('units')}")
        print(f"  caps {json.dumps(h.get('caps'))}")
        for c in h.get("choices", []):
            print(f"  - {c}")
        st = speed.starts(2, 1, 7)[1]
        cars = [{"preset": "Octane", "team": 0}, {"preset": "Octane", "team": 1}]
        ticks = 60
        inp = np.zeros((ticks, 2, 10), np.float32)
        inp[:, :, 0] = 1.0
        inp[:, :, 8] = 1.0
        inp[20:24, :, 5] = 1.0
        req = {"op": "segment", "cars": cars, "ticks": ticks, "frames": False, "start": st}
        try:
            runs = [np.frombuffer(w.request(req, inp.tobytes())[1], np.float32).reshape(ticks, 3, 18) for _ in range(2)]
        except WorkerError as e:
            print(f"  segment FAILED: {e}")
            w.close()
            continue
        a = runs[0]
        ok = np.isfinite(a).all()
        moved = float(np.abs(a[-1, 0, 0:3] - a[0, 0, 0:3]).max())
        print(f"  segment: {a.shape} bodies, finite {ok}, car 0 moved {moved:.3g} {h.get('units')} in {ticks} ticks, "
              f"two runs identical: {bool((runs[0] == runs[1]).all())}")
        job = {"mode": "1v1", "cars": cars, "arenas": 4, "threads": 1, "decision_ticks": 8, "decisions": 30, "warmup": 0,
               "repeats": 1, "seed": 1, "map": "Stadium_P"}
        ctl, rows = speed.controls(4, 2, 30, 1)
        try:
            rep, _ = w.request({"op": "speed", "job": job, "starts": speed.starts(4, 1, 1)}, ctl.tobytes() + rows.tobytes())
            print(f"  speed: {4 * 30 * 8 / rep['seconds'][0] / 1e3:.0f}k arena-ticks/s on a tiny job (not a benchmark)")
        except WorkerError as e:
            print(f"  speed FAILED: {e}")
        w.close()
