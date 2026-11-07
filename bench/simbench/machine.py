"""The machine a result was measured on, printed with every table."""
from __future__ import annotations

import os
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
