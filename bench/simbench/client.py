"""The harness's end of the wire: start an adapter's worker process and send it requests."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from . import protocol


class WorkerError(RuntimeError):
    pass


class Worker:
    def __init__(self, cmd: list[str], env: dict | None = None, cwd: str | None = None, log: Path | None = None):
        self.cmd = cmd
        self._log = open(log, "ab") if log else None
        e = dict(os.environ)
        e.update(env or {})
        try:
            self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log or None,
                                      env=e, cwd=cwd, bufsize=0)
        except OSError as err:
            raise WorkerError(f"could not start {cmd[0]}: {err}") from err
        self.hello = self.request({"op": "hello"})[0]

    def request(self, header: dict, payload: bytes = b"") -> tuple[dict, bytes]:
        try:
            protocol.send(self.p.stdin, header, payload)
            rep, data = protocol.recv(self.p.stdout)
        except (EOFError, BrokenPipeError, OSError) as err:
            code = self.p.poll()
            raise WorkerError(f"the worker ({' '.join(self.cmd[:3])}) stopped (exit code {code})") from err
        if "error" in rep:
            raise WorkerError(rep["error"] + ("\n" + rep["trace"] if rep.get("trace") else ""))
        return rep, data

    def close(self) -> None:
        if self.p.poll() is None:
            try:
                self.request({"op": "quit"})
            except WorkerError:
                pass
            try:
                self.p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.p.kill()
        if self._log:
            self._log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def python_adapter(path: Path) -> list[str]:
    """The command that serves a Python adapter file through the kit."""
    return [sys.executable, "-m", "simbench.kit", str(path)]
