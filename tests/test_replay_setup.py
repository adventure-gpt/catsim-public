"""The replay setup calls (whole-game benchmarks): a match's first kickoff built the game's way. Needs CATSIM_DATA."""
import os

import pytest

catsim = pytest.importorskip("catsim")
DATA = os.environ.get("CATSIM_DATA")
pytestmark = pytest.mark.skipif(not DATA, reason="set CATSIM_DATA to a folder made by `python -m catsim.dump`")

SPOTS = [([0.0, -4608.0, 83.0], [0, 16384, 0]), ([0.0, 4608.0, 83.0], [0, -16384, 0])]


@pytest.fixture(scope="module")
def lib():
    catsim.init(DATA, silent=True)
    return catsim._lib()


def first_kickoff(lib):
    a = lib.NativeArena([("23", 0), ("23", 1)], DATA, True, "Stadium_P", False, 2)
    a.start_at_drop(SPOTS)
    a.set_scripted(False)
    a.set_pad_rule(True)
    a.set_next_handles([23, 24, 25])
    a.restart(SPOTS, 1)
    a.physics_frame = 18
    a.set_ball_at_rest(False)
    a.set_countdown(3, 3, 1 << 20, 0.0, -1.0, False, 0)
    return a


def test_first_kickoff_drop(lib):
    a = first_kickoff(lib)
    for _ in range(5):
        a.step([[0.0] * 8, [0.0] * 8], 1)
    v = a.packet_view()
    assert len(v) == 2 * 14 + 13 + 34
    assert v[0] == 1.0 and abs(v[1]) < 1e-3 and abs(v[2] + 4608.0) < 1e-3     # car 0 on its spot, falling
    assert v[6] < 0.0
    assert a.clock()[3] == 0                                                   # still in the countdown


def test_countdown_runs_out_and_the_round_starts(lib):
    a = first_kickoff(lib)
    a.set_countdown(3, 3, 3, 0.0, -1.0, False, 0)          # 3 shown, its update timer just started
    for _ in range(3 * 120 + 30):
        a.step_frame([[1.0] + [0.0] * 7] * 2, 1, 1.0 / 120.0, 0.5)
    assert a.clock()[3] == 1


def test_same_setup_same_result(lib):
    out = []
    for _ in range(2):
        a = first_kickoff(lib)
        a.queue_kickoff(SPOTS)
        for t in range(600):
            a.step_frame([[1.0, 0.2, 0.0, 0.0, 0.0, float(t % 50 < 5), 1.0, 0.0]] * 2, 1, 1.0 / 120.0, 0.5)
        out.append(a.packet_view())
    assert out[0] == out[1]
