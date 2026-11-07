"""The simulators the harness knows how to start, and how to add your own.

Every simulator runs in its own worker process (simbench.protocol). Built in:

  catsim         catSim's fast engine, native Rust worker (bench/workers/catsim)
  catsim-py      catSim through its Python package (bench/adapters/catsim_py.py)
  rocketsim-v3   RocketSim v3, the `rocketsim` crate, native Rust worker (bench/workers/rocketsim_v3)
  rocketsim-v2   RocketSim v2 (C++) through its Python package `rocketsim` on PyPI (bench/adapters/rocketsim_v2.py)
  rocketgpu      RocketGPU through its own Python module, its CUDA build (bench/adapters/rocketgpu.py)
  rocketgpu-cpu  the same adapter on RocketGPU's CPU build
  rlmax          RLMAX, native Rust worker over its crate (bench/workers/rlmax), on the GPU

Your own: `--sims mine=path/to/adapter.py` (a Python adapter, see bench/adapters/TEMPLATE.py) or
`--sims mine=cmd:path/to/worker.exe` (any program that speaks the protocol).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .client import Worker, WorkerError, python_adapter

BENCH = Path(__file__).resolve().parents[1]

BUILTIN = {
    "catsim": {"rust": "catsim", "bin": "simbench-catsim"},
    "catsim-py": {"py": "catsim_py.py"},
    "rocketsim-v3": {"rust": "rocketsim_v3", "bin": "simbench-rocketsim-v3"},
    "rocketsim-v2": {"py": "rocketsim_v2.py"},
    "rocketgpu": {"py": "rocketgpu.py", "env": {"SIMBENCH_ROCKETGPU_BUILD": "cuda"}},
    "rocketgpu-cpu": {"py": "rocketgpu.py", "env": {"SIMBENCH_ROCKETGPU_BUILD": "cpu"}},
    "rlmax": {"rust": "rlmax", "bin": "simbench-rlmax"},
}


def exe(name: str) -> str:
    return name + (".exe" if sys.platform == "win32" else "")


def rust_binary(crate: str, binary: str) -> Path:
    target = Path(os.environ.get("CARGO_TARGET_DIR") or BENCH / "workers" / crate / "target")
    return target / "release" / exe(binary)


def build(name: str, quiet: bool = False) -> Path:
    """Build a built-in Rust worker (cargo build --release in its folder)."""
    spec = BUILTIN[name]
    if "rust" not in spec:
        raise SystemExit(f"{name} is a Python adapter: nothing to build")
    if not shutil.which("cargo"):
        raise SystemExit("cargo (Rust) is needed to build the native workers: https://rustup.rs")
    folder = BENCH / "workers" / spec["rust"]
    print(f"building {name} in {folder} (release) ...", flush=True)
    r = subprocess.run(["cargo", "build", "--release"], cwd=folder, capture_output=quiet, text=True)
    if r.returncode:
        raise SystemExit(f"building {name} failed" + (":\n" + r.stderr[-3000:] if quiet else ""))
    return rust_binary(spec["rust"], spec["bin"])


def command(name: str) -> tuple[list[str], dict]:
    """The worker command line and extra environment for a simulator name (built-in or `name=path`)."""
    env = {"PYTHONPATH": str(BENCH) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    if "=" in name:
        _, _, target = name.partition("=")
        if target.startswith("cmd:"):
            return target[4:].split(), env
        return python_adapter(Path(target)), env
    spec = BUILTIN.get(name)
    if spec is None:
        raise SystemExit(f"unknown simulator {name!r}: built in are {', '.join(BUILTIN)}; or give name=path/to/adapter.py")
    env.update(spec.get("env", {}))
    if "py" in spec:
        return python_adapter(BENCH / "adapters" / spec["py"]), env
    path = rust_binary(spec["rust"], spec["bin"])
    if not path.is_file() or _stale(path, spec["rust"]):
        path = build(name)
    return [str(path)], env


def _stale(binary: Path, crate: str) -> bool:
    """A source of the worker (its crate, the kit, catSim's engine for catSim's worker) is newer than its binary."""
    built = binary.stat().st_mtime
    roots = [BENCH / "workers" / crate, BENCH / "workers" / "kit"]
    if crate == "catsim":
        roots.append(BENCH.parent / "engine" / "core")
    for r in roots:
        for f in list(r.glob("src/**/*.rs")) + list(r.glob("Cargo.toml")):
            if f.stat().st_mtime > built:
                return True
    return False


def label(name: str) -> str:
    return name.partition("=")[0]


def start(name: str, setup_env: dict, log_dir: Path | None = None) -> Worker:
    cmd, env = command(name)
    log = (log_dir / f"{label(name)}.log") if log_dir else None
    w = Worker(cmd, env=env, log=log)
    w.request({"op": "setup", "env": setup_env})
    w.hello = w.request({"op": "hello"})[0]          # again: an adapter may learn its version during setup
    w.restart = lambda: start(name, setup_env, log_dir)
    w.label = label(name)
    return w


def try_start(name: str, setup_env: dict, log_dir: Path | None = None):
    """(worker, None), or (None, why it could not start) in plain words."""
    try:
        return start(name, setup_env, log_dir), None
    except (WorkerError, SystemExit, OSError) as e:
        msg = str(e).strip().splitlines()
        return None, (msg[0] if msg else type(e).__name__)
