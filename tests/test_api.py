"""The RocketSim-style API, on real game data. Needs a data folder from the dumper: set CATSIM_DATA to it."""
import math
import os

import pytest

catsim = pytest.importorskip("catsim")
DATA = os.environ.get("CATSIM_DATA")
pytestmark = pytest.mark.skipif(not DATA, reason="set CATSIM_DATA to a folder made by `python -m catsim.dump`")


@pytest.fixture(scope="module")
def rs():
    catsim.init(DATA, silent=True)
    return catsim


def two_cars(rs):
    a = rs.Arena(rs.GameMode.SOCCAR)
    blue = a.add_car(rs.Team.BLUE, rs.CarConfig(rs.CarConfig.OCTANE))
    orange = a.add_car(rs.Team.ORANGE, rs.CarConfig(rs.CarConfig.DOMINUS))
    return a, blue, orange


def test_kickoff_layout(rs):
    a, blue, orange = two_cars(rs)
    b = a.ball.get_state()
    assert abs(b.pos.x) < 1e-3 and abs(b.pos.y) < 1e-3 and 90 < b.pos.z < 110
    assert blue.get_state().pos.y < 0 < orange.get_state().pos.y
    assert len(a.get_boost_pads()) == 34 and sum(p.is_big for p in a.get_boost_pads()) == 6


def test_drive_and_boost(rs):
    a, blue, _ = two_cars(rs)
    start = blue.get_state()
    blue.set_controls(rs.CarControls(throttle=1.0, boost=True))
    a.step(120)
    s = blue.get_state()
    assert (s.pos - start.pos).length() > 500
    assert s.boost < start.boost
    assert a.tick_count == 120


def test_rotation_axes_are_orthonormal(rs):
    a, blue, _ = two_cars(rs)
    m = blue.get_state().rot_mat
    for v in (m.forward, m.right, m.up):
        assert math.isclose(v.length(), 1.0, abs_tol=1e-5)
    r = m.up.cross(m.forward)
    assert (r - m.right).length() < 1e-4


def test_state_set_round_trip(rs):
    a, blue, _ = two_cars(rs)
    s = blue.get_state()
    s.pos, s.vel, s.boost = rs.Vec(100, -1000, 17), rs.Vec(0, 500, 0), 50
    blue.set_state(s)
    t = blue.get_state()
    assert t.pos == rs.Vec(100, -1000, 17) and t.boost == 50
    a.ball.set_state(rs.BallState(rs.Vec(0, 0, 500), rs.Vec(0, 0, 0), rs.Vec(0, 0, 0)))
    a.step(30)
    assert a.ball.get_state().pos.z < 500


def test_same_inputs_same_result(rs):
    out = []
    for _ in range(2):
        a, blue, orange = two_cars(rs)
        blue.set_controls(rs.CarControls(throttle=1, steer=0.3, boost=True))
        orange.set_controls(rs.CarControls(throttle=1, jump=True))
        a.step(400)
        out.append((blue.get_state().pos.as_tuple(), orange.get_state().vel.as_tuple(), a.ball.get_state().pos.as_tuple()))
    assert out[0] == out[1]


def test_ball_prediction_matches_stepping(rs):
    a = rs.Arena()
    a.ball.set_state(rs.BallState(rs.Vec(0, 0, 300), rs.Vec(800, 1200, 400), rs.Vec(1, 2, 3)))
    pred = a.ball_prediction(240, 1)
    a.step(240)
    assert pred[-1].pos == a.ball.get_state().pos


def test_goal_callback(rs):
    a, blue, orange = two_cars(rs)
    goals = []
    a.set_goal_score_callback(lambda arena, team, data: goals.append((team, data)), "x")
    a.ball.set_state(rs.BallState(rs.Vec(0, 4500, 300), rs.Vec(0, 3000, 0), rs.Vec()))
    a.step(240)
    assert goals and goals[0] == (rs.Team.BLUE, "x")


def test_cars_drive_after_a_goal_until_the_countdown(rs):
    """With round_flow the 3 s after a goal take input, as in the game; the kickoff countdown after them doesn't."""
    a = rs.Arena(round_flow=True)
    blue = a.add_car(rs.Team.BLUE, rs.CarConfig(rs.CarConfig.OCTANE))
    a.add_car(rs.Team.ORANGE, rs.CarConfig(rs.CarConfig.DOMINUS))
    a.use_game_countdown()                       # a countdown long enough to watch (a 1 s start, then 3, 2, 1)
    a.ball.set_state(rs.BallState(rs.Vec(0, 5300, 300), rs.Vec(0, 500, 0)))   # just inside orange's net
    a.step(1)
    r = a.get_round()
    assert a.score == (1, 0) and r.phase == 2 and abs(r.phase_left - 3.0) < 1e-6
    start = blue.get_state()
    blue.set_controls(rs.CarControls(throttle=1.0, jump=True, boost=True))
    a.step(36)
    s = blue.get_state()
    assert a.get_round().phase == 2 and blue.get_hidden().driving
    assert s.pos.z > start.pos.z + 50 and s.boost < start.boost          # it jumped and boosted
    ticks = 36
    while a.get_round().phase == 2:
        a.step(1)
        ticks += 1
    assert ticks == 360                          # the window is 3 s
    assert a.get_round().phase == 0 and not any(c.get_hidden().driving for c in a.get_cars())
    twin = a.copy()                              # the same arena with no input (the cars drop onto their spots)
    for c in twin.get_cars():
        c.set_controls(rs.CarControls())
    a.step(120)                                  # jump, boost and throttle still held in `a`
    twin.step(120)
    assert a.get_round().phase == 0
    mine, other = blue.get_state(), twin.get_cars()[0].get_state()
    # not bit for bit: the held throttle still reaches the tire friction model, a few 1e-7 uu/s
    assert (mine.pos - other.pos).length() < 0.01 and (mine.vel - other.vel).length() < 0.01
    assert mine.boost == other.boost and not blue.get_hidden().jump.active


def test_car_configs(rs):
    sizes = {n: rs.CarConfig(n).hitbox_size for n in ("Octane", "MuscleCar", "Darkcar", "LongCar", "Hybrid", "Merc")}
    assert sizes["Octane"].x == pytest.approx(120.507, abs=1e-3)
    assert len({s.as_tuple() for s in sizes.values()}) == 6


def test_second_map_pads(rs):
    a = rs.Arena(map="UF_Night_P")
    zs = {p.is_big: p.get_pos().z for p in a.get_boost_pads()}
    assert zs == {False: 70.0, True: 74.0}
    assert {p.get_pos().z for p in rs.Arena().get_boost_pads() if p.is_big} == {72.0}


def test_random_draw_model_parts_order(rs):
    items = os.path.join(DATA, "items.txt")
    if not os.path.isfile(items):
        pytest.skip("the data folder has no items.txt (dumped with --skip-items)")
    lo = dict(car_id=23, decal_id=5337, wheels_id=4842, boost_id=4994, antenna_id=12019, hat_id=12697, trails_id=8969,
              goal_explosion_id=12660)
    a = rs.Arena(map="UF_Night_P")
    for team in (0, 1, 0, 1):
        a.add_car(rs.Team(team), loadout=lo)
    s = a.start_game_rng(12345, 30.0)
    assert s is not None and a.draw_model_note == "covered"
    parts = [0, 1, 1, 1, 1, 2]                       # body, four skeletal wheels, the skeletal antenna
    want = [(-1, 4)] * 5 + [(i, k) for i in range(4) for k in parts]
    assert a.draw_parts() == want
    before = a.crt_state
    a.step_frame(1, 1 / 120, 0.5)
    assert a.crt_state != before
