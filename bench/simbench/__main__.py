"""simbench: one harness, any simulator. docs/BENCHMARKS.md explains every benchmark and how to read the results.

    python -m simbench speed       --sims catsim,rocketsim-v3          [--plan quick|full] [--rounds 5]
    python -m simbench suite       --sims catsim,rocketsim-v3 --suite-data DIR   [--levels full,public,packet] [--splits dev,test]
    python -m simbench rlpr        --sims catsim                       (RocketSim's own rlpr_metric, our bridge)
    python -m simbench recordings  --sims catsim                       [--levels spots,stream]
    python -m simbench all         --sims ...                          (every benchmark the simulators support)
    python -m simbench report      [--doc ../docs/BENCHMARKS.md]       (rebuild the report from bench_results/)
    python -m simbench check       --sims mine=path/to/adapter.py      (a quick test of a new adapter)
    python -m simbench build       catsim                              (build a native worker; done on first use)

Every benchmark ends by writing bench_results/report.md, report.html (open it in a browser) and the charts, and printing
the summary table. Common options: --data DIR (catSim's dumper output; default $CATSIM_DATA or ./catsim_data), --meshes DIR (RocketSim's
collision_meshes; default: made from --data), --out DIR (default ./bench_results), --threads N (default: all).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

from . import machine, sims


def setup_env(a) -> dict:
    data = a.data or os.environ.get("CATSIM_DATA") or ("catsim_data" if Path("catsim_data").is_dir() else None)
    meshes = a.meshes
    if not meshes and data:
        from .meshes import convert
        meshes = str(Path(a.out) / "collision_meshes")
        if not (Path(meshes) / "soccar" / "mesh_15.cmf").is_file():
            bad = convert(Path(data), Path(meshes))
            if bad:
                print(f"note: RocketSim's meshes made from {data} differ from RocketSim's dumper's in {', '.join(bad)}")
    return {"catsim_data": str(Path(data).resolve()) if data else None,
            "collision_meshes": str(Path(meshes).resolve()) if meshes else None,
            "threads": a.threads}


class Results:
    def __init__(self, out: Path, bench: str):
        out.mkdir(parents=True, exist_ok=True)
        self.path = out / "results.jsonl"
        self.machine = machine.describe()
        self.stamp = datetime.datetime.now().isoformat(timespec="seconds")
        self.bench = bench

    def emit(self, rec: dict) -> None:
        rec = dict(rec, bench=self.bench, when=self.stamp, machine=self.machine)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        if rec.get("status") == "ok" and rec["kind"] == "speed":
            print(f"  {rec['sim']:14s} {rec['mode']} {rec['arenas']:5d} arenas {rec['threads_used']:2d} thr: "
                  f"{rec['arena_ticks_per_s'] / 1e3:9.1f}k arena-ticks/s  ({rec['min'] / 1e3:.1f}k-{rec['max'] / 1e3:.1f}k, "
                  f"{rec['rounds']} rounds)  {rec['cpu_us_per_arena_tick']:.2f} us/arena-tick", flush=True)
        elif rec.get("status") not in (None, "ok"):
            where = f" {rec['mode']} {rec.get('arenas')} arenas {rec.get('threads')} thr" if "mode" in rec else ""
            print(f"  {rec['sim']:14s}{where}: {rec['status']} ({rec.get('why', '')})", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m simbench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench", choices=["speed", "suite", "rlpr", "recordings", "all", "report", "check", "build"])
    ap.add_argument("names", nargs="*", help="(build) the simulators to build")
    ap.add_argument("--sims", default="catsim,rocketsim-v3")
    ap.add_argument("--data")
    ap.add_argument("--meshes")
    ap.add_argument("--out", default="bench_results")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--plan", default="full", choices=["quick", "full"])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1106)
    ap.add_argument("--device", choices=["cpu", "gpu"], help="(speed) only simulators on this device")
    ap.add_argument("--only", help="(speed) only these rows, as mode:arenas:threads[,...] (e.g. 3v3:1024:16)")
    ap.add_argument("--suite-data", help="(suite) the full-state suite's data folder (its README says how to get it)")
    ap.add_argument("--levels", help="which of the benchmark's starts to run: full, public, packet (suite); spots, stream "
                    "(recordings); or the ids A, B, Bp, C, D (default: all the benchmark has)")
    ap.add_argument("--splits", default="all", help="(suite) which of its splits, or recording names")
    ap.add_argument("--limit", type=int, default=0, help="(suite, recordings) at most this many recordings (a quick look)")
    ap.add_argument("--jobs", type=int, default=0, help="(suite, recordings) worker processes per simulator (default: threads / 2)")
    ap.add_argument("--doc", help="(report) also fill this docs/BENCHMARKS.md's generated sections")
    a = ap.parse_args(argv)
    if a.levels:
        from .state import WORDS
        a.levels = ",".join(WORDS.get(x.strip(), x.strip()) for x in a.levels.split(","))
    names = [s.strip() for s in a.sims.split(",") if s.strip()]
    out = Path(a.out)
    if a.bench == "build":
        for n in a.names or names:
            print(sims.build(n))
        return
    if a.bench == "report":
        from . import report
        report.main(out, Path(a.doc) if a.doc else None)
        return
    env = setup_env(a)
    m = machine.describe()
    print(f"machine: {machine.line(m, gpu=bool(m['gpus']))}")
    if a.bench == "check":
        from . import check
        check.main(names, env, out)
        return
    todo = ["speed", "suite", "rlpr", "recordings"] if a.bench == "all" else [a.bench]
    for b in todo:
        res = Results(out, b)
        print(f"== {b}", flush=True)
        if b == "speed":
            from . import speed
            if a.only:
                speed.ONLY = {(m, int(n), int(t)) for m, n, t in (x.split(":") for x in a.only.split(","))}
            speed.run(names, env, "quick" if a.plan == "quick" else "full", a.rounds, a.threads, a.seed, out, res.emit, a.device)
        elif b == "suite":
            from . import suite
            if not a.suite_data:
                print("  skipped: give --suite-data (the full-state suite's recordings; docs/BENCHMARKS.md says where)")
                continue
            suite.run(names, env, Path(a.suite_data), a.levels, a.splits, a.limit, a.jobs or max(1, a.threads // 2), out, res.emit)
        elif b == "rlpr":
            from . import rlpr
            rlpr.run(names, env, out, res.emit)
        elif b == "recordings":
            from . import recordings
            recordings.run(names, env, a.levels, a.limit, a.jobs or max(1, a.threads // 2), out, res.emit)
    print(f"results: {out / 'results.jsonl'}")
    from . import report
    report.main(out, quiet=True)


if __name__ == "__main__":
    main()
