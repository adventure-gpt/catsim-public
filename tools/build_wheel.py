"""Build catSim's wheel for this machine: compile the engine, then pack it with the Python package.

    python tools/build_wheel.py [--jobs 4] [--out dist]

No network: everything comes from this repo and the local Rust and Python installs. The wheel holds no game
data; users make that with `python -m catsim.dump` (or `catsim-dump`).
"""
import argparse
import base64
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME, VERSION = "catsim", "0.1.1"        # NAME: the import package
DIST = "catsim-rl"                          # the name on PyPI ("catsim" there is an unrelated project)
DIST_FILE = DIST.replace("-", "_")
SKIP = {"__pycache__"}


def tag() -> str:
    if sys.platform == "win32":
        return "cp310-abi3-win_amd64"
    if sys.platform.startswith("linux") and platform.machine() in ("x86_64", "AMD64"):
        return "cp310-abi3-linux_x86_64"
    raise SystemExit(f"no wheel recipe for {sys.platform} {platform.machine()}")


def build(jobs: int, online: bool = False) -> Path:
    env = dict(os.environ, PYO3_PYTHON=sys.executable, CARGO_BUILD_JOBS=str(jobs))
    # Keep the builder's own paths (home folder, toolchain, checkout) out of the compiled module. RUSTFLAGS replaces
    # engine/.cargo/config.toml's flags, so the CPU level is repeated here.
    home = Path.home()
    cargo_home = Path(os.environ.get("CARGO_HOME", home / ".cargo"))
    rustup_home = Path(os.environ.get("RUSTUP_HOME", home / ".rustup"))
    flags = ["-C", "target-cpu=x86-64-v2"]
    for src, dst in ((cargo_home, "/cargo"), (rustup_home, "/rustup"), (ROOT, "/catsim"), (home, "/home")):
        flags.append(f"--remap-path-prefix={src}={dst}")
    if sys.platform == "win32":
        flags += ["-C", "link-arg=/PDBALTPATH:%_PDB%"]
    env["RUSTFLAGS"] = " ".join(flags)
    subprocess.run(["cargo", "build", "--release", *([] if online else ["--offline"]), "-p", "catsim_python", "-j", str(jobs)],
                   cwd=ROOT / "engine", env=env, check=True)
    tgt = Path(env.get("CARGO_TARGET_DIR", ROOT / "engine" / "target")) / "release"
    lib = tgt / ("_native.dll" if sys.platform == "win32" else "lib_native.so")
    if not lib.is_file():
        raise SystemExit(f"built library not found: {lib}")
    return lib


def record_line(arc: str, data: bytes) -> str:
    h = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{arc},sha256={h},{len(data)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "dist")
    ap.add_argument("--lib", type=Path, help="use this already-built library instead of building")
    ap.add_argument("--tag", help="wheel tag instead of this machine's (e.g. cp310-abi3-manylinux_2_28_x86_64 in CI)")
    ap.add_argument("--online", action="store_true", help="let cargo fetch crates (CI); the default builds offline")
    a = ap.parse_args()
    lib = a.lib or build(a.jobs, a.online)
    t = a.tag or tag()
    files: dict[str, bytes] = {}
    pkg = ROOT / "python" / NAME
    for p in sorted(pkg.rglob("*")):
        if p.is_file() and not (set(p.parts) & SKIP) and p.suffix not in (".pyd", ".so", ".dll", ".pyc"):
            files[f"{NAME}/{p.relative_to(pkg).as_posix()}"] = p.read_bytes()
    files[f"{NAME}/_native.pyd" if sys.platform == "win32" else f"{NAME}/_native.abi3.so"] = lib.read_bytes()
    readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
    dist = f"{DIST_FILE}-{VERSION}.dist-info"
    files[f"{dist}/METADATA"] = (
        "Metadata-Version: 2.1\n"
        f"Name: {DIST}\nVersion: {VERSION}\n"
        "Summary: A bit-exact Rocket League physics simulator with a RocketSim-style Python API\n"
        "License: MIT\n"
        "Author: adventure-gpt\n"
        "Project-URL: Source, https://github.com/adventure-gpt/catsim-public\n"
        "Requires-Python: >=3.10\n"
        "Provides-Extra: dump\nRequires-Dist: cryptography; extra == \"dump\"\n"
        "Description-Content-Type: text/markdown\n\n" + readme).encode()
    if (ROOT / "LICENSE").exists():
        files[f"{dist}/LICENSE"] = (ROOT / "LICENSE").read_bytes()
    files[f"{dist}/WHEEL"] = f"Wheel-Version: 1.0\nGenerator: catsim build_wheel.py\nRoot-Is-Purelib: false\nTag: {t}\n".encode()
    files[f"{dist}/entry_points.txt"] = b"[console_scripts]\ncatsim-dump = catsim.dump.__main__:main\n"
    rec = [record_line(k, v) for k, v in files.items()] + [f"{dist}/RECORD,,"]
    files[f"{dist}/RECORD"] = ("\n".join(rec) + "\n").encode()
    a.out.mkdir(parents=True, exist_ok=True)
    whl = a.out / f"{DIST_FILE}-{VERSION}-{t}.whl"
    with zipfile.ZipFile(whl, "w", zipfile.ZIP_DEFLATED) as z:
        for k, v in files.items():
            z.writestr(k, v)
    print(whl)


if __name__ == "__main__":
    main()
