"""The generated report names every benchmark by its label: no "level A".."level D" and no bare Cf / Df anywhere a
reader sees, even from results stored by older versions."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))

from simbench import report  # noqa: E402

MACHINE = {"cpu": "Test CPU", "threads": 4, "ram_gb": 8, "os": "Linux 6 (test)", "python": "3.12", "gpus": []}


def rec(**kw):
    return dict({"machine": MACHINE, "when": "2026-11-07T00:00:00", "version": "test 1"}, **kw)


def results():
    whole = {"recordings": 2, "failed": 0, "exact_to_the_end": 1, "compared_rows": 10, "exact_rows": 9, "ticks": 100,
             "exact_pct": 90.0, "first_drift_s": {"min": 1, "q1": 1, "median": 1.5, "q3": 2, "max": 2},
             "error_after_kickoff": {"1.0": 0.1, "5.0": 0.2, "30.0": None}, "bodies_only": True}
    seg = {"all": {"recordings": 1, "body_ticks": 100, "exact_body_ticks": 99.5, "segments": 10, "exact_segments": 90.0,
                   "car_ticks": 50, "segment_pass": 99.0, "mean_error": {"1": 0.0, "30": 0.1, "120": 0.2}, "failed_segments": 0}}
    out = []
    for lv in ("C", "Cf"):
        out.append(rec(kind="accuracy", level=lv, suite="our recordings", sim="rocketsim-v3", status="ok", summary=whole,
                       assists=["the end-of-match rule as in C"], recordings=[{"name": "g1", "first_drift": None, "seconds": 10}]))
    for lv in ("D", "Df"):
        out.append(rec(kind="accuracy", level=lv, suite="our recordings", sim="rocketsim-v3", status="not supported",
                       why="level D: it doesn't take the game's random-number stream"))
    out.append(rec(kind="accuracy", level="A", suite="full-state suite", sim="rocketgpu-cpu", status="ok", summary=seg,
                   choices=["level A: set_full_state with every suite field", "level B: set_state_v3"]))
    out.append(rec(kind="accuracy", level="A", suite="full-state suite", sim="rocketsim-v3", status="not supported",
                   why="its adapter can't take the full start state"))
    out.append(rec(kind="speed", sim="rocketsim-v3", device="cpu", mode="1v1", arenas=1024, threads=4, threads_used=4,
                   status="ok", arena_ticks_per_s=1e6, min=9e5, max=1.1e6, rounds=5, cpu_us_per_arena_tick=4.0,
                   load=[{"wall_s": 1.0, "load1": 1.0, "other_cpu_s": 0.0, "other_cores": 0.0}]))
    return out


def test_no_benchmark_letters(tmp_path):
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results()) + "\n", encoding="utf-8")
    report.main(tmp_path, quiet=True)
    for name in ("report.md", "report.html"):
        text = (tmp_path / name).read_text(encoding="utf-8")
        assert not report.check_words(text, name), report.check_words(text, name)
    page = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "<script" not in page
