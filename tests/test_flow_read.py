"""The harness flow's read of a simulator's bodies (simbench.flow.packet_rows) against catSim's own packet view, and its plan of
a recording's restarts and respawns. Needs CATSIM_DATA."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

BENCH = Path(__file__).resolve().parents[1] / "bench"
sys.path.insert(0, str(BENCH))

flow = pytest.importorskip("simbench.flow")
game = pytest.importorskip("simbench.game")
REC = BENCH / "recordings" / "1v1_stadium_08.npz"
DATA = os.environ.get("CATSIM_DATA")
pytestmark = pytest.mark.skipif(not DATA or not REC.is_file(), reason="needs CATSIM_DATA and the recording in git")


def _arena(rec):
    catsim = pytest.importorskip("catsim")
    catsim.init(DATA, silent=True)
    ops = rec.get("ops_C") or rec["ops"]
    cfg = next(o[1] for o in ops if o[0] == "arena")
    return catsim._lib().NativeArena([(str(c["body"]), int(c["team"])) for c in cfg["cars"]], DATA, True, cfg["map"], False, 2)


def test_read_matches_catsims_packet_view():
    rec = game.load(REC)
    nc = int(rec["meta"]["cars"])
    a = _arena(rec)
    a.start_at_drop(None)
    a.step([[0.0] * 8] * nc, 240)
    ctl = np.asarray(rec["frame_controls"], np.float32)
    start = int(np.nonzero(rec["truth_phase"][rec["frame_judge"].clip(0)] == 2)[0][0]) + 200
    rows, bodies = [], []
    for f in range(start, start + 1500):
        a.step(ctl[f].tolist(), 1)
        rows.append(np.asarray(a.packet_view(), np.float64))
        b = np.zeros((nc + 1, 18), np.float32)
        for k in range(nc + 1):
            x = a.car_body(k) if k < nc else a.ball_body()
            if x is not None:
                w = np.asarray(x[:18], np.float32)
                basis = w[3:12].reshape(3, 3)
                b[k] = np.concatenate([w[0:3], w[12:15], w[15:18], basis[:, 0], basis[:, 1], basis[:, 2]])
        bodies.append(b)
    view = np.full((len(rows), nc * flow.CAR_W + flow.BALL_W + flow.PADS), np.nan)
    bases = np.full((len(rows), nc + 1, 9), np.nan, np.float32)
    for r, b in enumerate(bodies):
        view[r], bases[r] = flow._view_of(b, "bt", nc)
    flow.packet_rows(view, bases, nc)
    truth = np.stack(rows)
    for k in range(nc):
        at = k * flow.CAR_W
        moved = np.abs(game.q_vel(view[:, at + 4:at + 7]) - truth[:, at + 4:at + 7])
        assert moved.max() < 0.002
        spun = np.abs(game.q_vel(view[:, at + 7:at + 10], 1e4) - truth[:, at + 7:at + 10])
        assert spun.max() < 2e-5
        turned = np.abs(view[:, at + 10:at + 13] - truth[:, at + 10:at + 13])
        assert np.minimum(turned, np.abs(turned - 65536)).max() <= 1


def test_plan_finds_every_restart_and_respawn():
    rec = game.load(REC)
    births = flow.birth(rec, DATA)
    assert births is not None
    kick_at, back_at, kick_rows, back_rows = flow.spawn_plan(rec, births)
    ph = rec["truth_phase"]
    later = [r for r in range(1, len(ph)) if ph[r] == 1 and ph[r - 1] != 1]
    assert kick_rows == set(later)
    for f, plan in kick_at.items():
        assert len(plan["cars"]) == int(rec["meta"]["cars"])
        assert plan["after"] <= int(rec["frame_steps"][f])
