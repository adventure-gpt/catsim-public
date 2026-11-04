"""The engine's C runtime tables are math, not copied data: recompute every one and compare."""
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mpmath")
ROOT = Path(__file__).resolve().parents[1]


def test_crt_tables_are_their_definitions():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "gen_crt_tables.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.count("ok ") == 8
