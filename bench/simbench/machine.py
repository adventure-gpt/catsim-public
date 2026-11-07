"""The machine a result was measured on, printed with every table."""
from __future__ import annotations

import os
import time
import platform
import shutil
import subprocess
import sys


def _run(cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def cpu_name() -> str:
    if sys.platform == "win32":
        out = _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"])
        if out:
            return out.splitlines()[0].strip()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def ram_gb() -> float:
    if sys.platform == "win32":
        out = _run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"])
        try:
            return round(int(out) / 2 ** 30, 1)
        except ValueError:
            return 0.0
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                return round(int(line.split()[1]) / 2 ** 20, 1)
    except OSError:
        pass
    return 0.0


def gpus() -> list[str]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    return [x.strip() for x in out.splitlines() if x.strip()]


def describe() -> dict:
    return {
        "cpu": cpu_name(),
        "threads": os.cpu_count(),
        "ram_gb": ram_gb(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "gpus": gpus(),
    }


def line(m: dict, gpu: bool = False) -> str:
    s = f"{m['cpu']}, {m['threads']} threads, {m['ram_gb']:g} GB RAM, {m['os'].split(' (')[0]}"
    if gpu:
        s = (", ".join(m["gpus"]) or "no NVIDIA GPU found") + f"; host {s}"
    return s


# ---- what else the machine was doing during a timed round ----

def _proc_times(pids) -> float:
    """CPU seconds (user + system) the given processes and their children have used, from /proc (Linux)."""
    hz = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    tot = 0
    for pid in pids:
        try:
            f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
            tot += int(f[11]) + int(f[12]) + int(f[13]) + int(f[14])
        except (OSError, IndexError, ValueError):
            pass
    return tot / hz


def _tree(pid: int) -> list:
    """A process and all its descendants (Linux)."""
    kids = {}
    try:
        for d in os.listdir("/proc"):
            if d.isdigit():
                try:
                    ppid = int(open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()[1])
                    kids.setdefault(ppid, []).append(int(d))
                except (OSError, IndexError, ValueError):
                    pass
    except OSError:
        return [pid]
    out, todo = [], [pid]
    while todo:
        x = todo.pop()
        out.append(x)
        todo += kids.get(x, [])
    return out


def _cpu_busy() -> float:
    """Busy CPU seconds of the whole machine since boot (all cores, /proc/stat)."""
    try:
        f = open("/proc/stat").readline().split()[1:]
        v = [int(x) for x in f]
        idle = v[3] + (v[4] if len(v) > 4 else 0)
        return (sum(v[:8]) - idle) / (os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100)
    except (OSError, ValueError):
        return float("nan")


def _gpu_others(mine) -> list:
    """Other processes holding a GPU context (nvidia-smi), as "pid name"."""
    out = _run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"])
    rows = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if parts and parts[0].isdigit() and int(parts[0]) not in mine:
            rows.append(" ".join(parts))
    return rows


class Busy:
    """Measures what else ran during a timed round: the load average, the CPU seconds of every process outside the
    harness and its workers, and the other processes on the GPU. Linux only for the CPU part (elsewhere: not measured)."""

    def __init__(self, gpu: bool = False):
        self.gpu = gpu

    def __enter__(self):
        self.mine = _tree(os.getpid())
        self.t0 = time.perf_counter()
        self.busy0 = _cpu_busy()
        self.mine0 = _proc_times(self.mine)
        self.gpu0 = _gpu_others(set(self.mine)) if self.gpu else []
        return self

    def __exit__(self, *exc):
        mine = set(self.mine) | set(_tree(os.getpid()))
        wall = time.perf_counter() - self.t0
        other = (_cpu_busy() - self.busy0) - (_proc_times(mine) - self.mine0)
        self.result = {"wall_s": round(wall, 3), "load1": round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None,
                       "other_cpu_s": None if other != other else round(max(other, 0.0), 2),
                       "other_cores": None if other != other else round(max(other, 0.0) / max(wall, 1e-9), 2)}
        if self.gpu:
            g = sorted(set(self.gpu0) | set(_gpu_others(mine)))
            self.result["other_gpu_processes"] = g
        return False
