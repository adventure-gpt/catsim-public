"""RocketSim's own replay metric (rocketsim_test's `rlpr_metric`), unchanged, with any simbench simulator behind it.

RocketSim (MIT) is fetched at a pinned release into bench_results/rlpr/RocketSim; its five bundled RLPR recordings come
with it. Our bridge backend (bench/rlpr_bridge/simbench_bridge.rs, part of this repository) is copied in, and two lines
of rlpr_metric.rs are edited so the metric constructs the bridge when SIMBENCH_WORKER is set (RocketSim v3's own backend
otherwise) and prints the simulator's name. The scorer itself is RocketSim's, untouched.

Two runs per simulator, as RocketSim's README describes them: segments of 120 ticks from each recorded state (pass % and
mean normalized error, by contact category), and one tick from every recorded state (`--reset-each-tick`).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from . import sims

REPO = "https://github.com/ZealanL/RocketSim.git"
TAG = "v0.3.5"
COMMIT = "88bd345ec1eee863d09c834259215cf025596c08"
BRIDGE = Path(__file__).resolve().parents[1] / "rlpr_bridge" / "simbench_bridge.rs"
EDITS = [
    ('#[path = "rlpr_metric/v3.rs"]\nmod v3;',
     '#[path = "rlpr_metric/v3.rs"]\nmod v3;\n#[path = "rlpr_metric/simbench_bridge.rs"]\nmod simbench_bridge;'),
    ("let mut v3_backend = v3::V3Backend::with_dodge_deadzone(args.dodge_deadzone);",
     "let mut v3_backend = simbench_bridge::Either::new(args.dodge_deadzone);"),
    ('print_report("v3", ', 'print_report(simbench_bridge::label(), '),
]


def checkout(root: Path) -> Path:
    rs = root / "RocketSim"
    if not (rs / ".git").is_dir():
        root.mkdir(parents=True, exist_ok=True)
        print(f"  fetching RocketSim {TAG} into {rs} ...", flush=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", TAG, REPO, str(rs)], check=True)
    head = subprocess.run(["git", "-C", str(rs), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        raise SystemExit(f"{rs} is at {head[:10]}, not RocketSim {TAG} ({COMMIT[:10]}): remove it and run again")
    return rs


def prepare(rs: Path, meshes: Path) -> Path:
    dst = rs / "collision_meshes" / "soccar"
    dst.mkdir(parents=True, exist_ok=True)
    for f in (meshes / "soccar").glob("*.cmf"):
        shutil.copy2(f, dst / f.name)
    ex = rs / "rocketsim_test" / "examples"
    shutil.copy2(BRIDGE, ex / "rlpr_metric" / "simbench_bridge.rs")
    src = ex / "rlpr_metric.rs"
    s = src.read_text(encoding="utf-8")
    if "simbench_bridge" not in s:
        for old, new in EDITS:
            n = s.count(old)
            if n < 1:
                raise SystemExit(f"rlpr_metric.rs changed upstream: can't find {old!r}")
            s = s.replace(old, new)
        src.write_text(s, encoding="utf-8")
    print("  building rlpr_metric (release) ...", flush=True)
    subprocess.run(["cargo", "build", "--release", "-q", "-p", "rocketsim_test", "--example", "rlpr_metric"], cwd=rs, check=True)
    exe = rs / "target" / "release" / "examples" / sims.exe("rlpr_metric")
    if not exe.is_file():
        raise SystemExit(f"rlpr_metric was not built at {exe}")
    return exe


ROW = re.compile(r"^(\S+)\s+(car_ball|car_car|ball_world|chassis_world|wheel_world|no_contact|total)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)")


def parse(text: str) -> dict:
    """The metric's combined table: category -> support, passed, pass %, mean norm, max norm."""
    out = {}
    combined = text.split("Combined recordings:")[-1] if "Combined recordings:" in text else text
    for line in combined.splitlines():
        m = ROW.match(line.strip())
        if m:
            _, cat, sup, passed, pct, mean, mx = m.groups()
            num = lambda x: None if x == "-" else float(x)
            out[cat] = {"support": int(sup), "passed": int(passed), "pass_pct": num(pct), "mean_norm": num(mean), "max_norm": num(mx)}
    return out


def run(names, setup_env, out_dir, emit):
    out_dir = Path(out_dir)
    if not setup_env.get("collision_meshes"):
        raise SystemExit("rlpr needs RocketSim's collision meshes: give --data (catSim's dump) or --meshes")
    rs = checkout(out_dir / "rlpr")
    exe = prepare(rs, Path(setup_env["collision_meshes"]))
    for name in names:
        label = sims.label(name)
        env = dict(os.environ)
        hello = None
        if label != "rocketsim-v3":
            w, why = sims.try_start(name, setup_env, out_dir)
            if w is None:
                emit({"kind": "rlpr", "sim": label, "status": "not run", "why": why})
                continue
            hello = w.hello
            w.close()
            if "rlpr" not in hello.get("caps", {}).get("levels", []):
                emit({"kind": "rlpr", "sim": label, "status": "not supported", "version": hello.get("version"),
                      "why": "its adapter can't take an RLPR tick record"})
                print(f"  {label:14s}: not supported (its adapter can't take an RLPR tick record)", flush=True)
                continue
            cmd, extra = sims.command(name)
            env.update(extra)
            env["SIMBENCH_WORKER"] = json.dumps([c.replace("\\", "/") for c in cmd])
            env["SIMBENCH_SETUP"] = json.dumps(setup_env)
            env["SIMBENCH_LABEL"] = label
        else:
            env.pop("SIMBENCH_WORKER", None)
        res = {}
        for mode, flags in (("segments_120", []), ("one_tick", ["--reset-each-tick"])):
            r = subprocess.run([str(exe), *flags], cwd=rs, env=env, capture_output=True, text=True)
            log = out_dir / f"rlpr_{label}_{mode}.txt"
            log.write_text(r.stdout + r.stderr, encoding="utf-8")
            if r.returncode:
                res[mode] = {"error": (r.stderr or r.stdout)[-500:]}
                continue
            res[mode] = parse(r.stdout)
        ok = all("total" in res.get(m, {}) for m in ("segments_120", "one_tick"))
        emit({"kind": "rlpr", "sim": label, "status": "ok" if ok else "failed", "version": (hello or {}).get("version") or f"rocketsim {TAG} (its own backend)",
              "about": (hello or {}).get("about") or "RocketSim v3's own rlpr_metric backend", "choices": (hello or {}).get("choices", []),
              "rocketsim": f"{TAG} {COMMIT[:10]}", "results": res})
        if ok:
            a, b = res["segments_120"]["total"], res["one_tick"]["total"]
            print(f"  {label:14s} 120-tick segments: pass {a['pass_pct']:.4f}%  mean norm {a['mean_norm']:.4f} ({a['support']:,} car-ticks)"
                  f"   one tick: pass {b['pass_pct']:.4f}%  mean norm {b['mean_norm']:.4f}", flush=True)
        else:
            print(f"  {label:14s}: failed ({json.dumps(res)[:300]})", flush=True)
