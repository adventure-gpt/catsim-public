"""Hidden state: exact copies, and moving a car's and the ball's whole state into a fresh arena. Needs CATSIM_DATA."""
import os

import pytest

catsim = pytest.importorskip("catsim")
DATA = os.environ.get("CATSIM_DATA")
pytestmark = pytest.mark.skipif(not DATA, reason="set CATSIM_DATA to a folder made by `python -m catsim.dump`")


@pytest.fixture(scope="module")
def rs():
    catsim.init(DATA, silent=True)
    return catsim


def arena(rs):
    a = rs.Arena()
    for team, body in ((0, rs.CarConfig.OCTANE), (1, rs.CarConfig.DOMINUS), (0, rs.CarConfig.BREAKOUT), (1, rs.CarConfig.MERC)):
        a.add_car(rs.Team(team), rs.CarConfig(body))
    return a


def controls(rs, t, k):
    """Busy, fixed inputs: driving, boosting, jumping and dodging at different times per car."""
    phase = (t // 40 + k) % 6
    return rs.CarControls(throttle=1.0, steer=0.4 if k % 2 else -0.3, pitch=-1.0 if phase == 2 else 0.0, yaw=0.5 if phase == 4 else 0.0,
                          jump=phase in (1, 3) and t % 40 < 8 + (t % 40 > 20) * 6, boost=phase != 5, handbrake=phase == 5)


def run(rs, a, start, ticks):
    for t in range(start, start + ticks):
        for k, c in enumerate(a.get_cars()):
            c.set_controls(controls(rs, t, k))
        a.step(1)


def snapshot(a):
    out = [tuple(x for x in c.get_body().to_row()) if c.get_body() else None for c in a.get_cars()]
    out += [tuple(c.get_hidden().to_row()) if c.get_hidden() else None for c in a.get_cars()]
    b = a.ball.get_body()
    out.append(None if b is None else tuple(b.to_row()))
    return out


def test_field_lists_match(rs):
    assert len(rs._lib().car_hidden_fields()) == len(rs.CarHidden().to_row()) == 61
    assert len(rs._lib().body_fields()) == len(rs.BodyState().to_row()) == 20


def test_copy_is_exact(rs):
    a = arena(rs)
    run(rs, a, 0, 300)
    b = a.copy()
    assert snapshot(a) == snapshot(b)
    run(rs, a, 300, 400)
    run(rs, b, 300, 400)
    assert snapshot(a) == snapshot(b)


def test_hidden_round_trip(rs):
    a = arena(rs)
    run(rs, a, 0, 237)
    for c in a.get_cars():
        h = c.get_hidden()
        assert rs.CarHidden.from_row(h.to_row()) == h
        c.set_hidden(h)
        assert c.get_hidden() == h
        body = c.get_body()
        c.set_body(body)
        assert c.get_body() == body


def test_move_whole_state_to_a_fresh_arena(rs):
    """Bodies, hidden state, the round and the pads into a new arena: the next ticks are identical while no contact
    point is cached (cars on their wheels or in the air, the ball in the air). Contact manifolds are not settable."""
    a = arena(rs)
    run(rs, a, 0, 200)
    a.ball.set_body(rs.BodyState(rs.Vec(0, 20, 30), (1, 0, 0, 0, 1, 0, 0, 0, 1), rs.Vec(1, -2, 30), rs.Vec(0.5, 0.25, 0)))
    run(rs, a, 200, 1)
    b = arena(rs)
    b.set_round(a.get_round())
    for i, p in enumerate(a.get_boost_pads()):
        b.get_boost_pads()[i].set_timer(*p.get_timer())
    for ca, cb in zip(a.get_cars(), b.get_cars()):
        cb.set_body(ca.get_body())
        cb.set_hidden(ca.get_hidden())
    b.ball.set_body(a.ball.get_body())
    assert snapshot(a) == snapshot(b)
    for t in range(201, 261):
        run(rs, a, t, 1)
        run(rs, b, t, 1)
        assert snapshot(a) == snapshot(b), f"tick {t}"


def test_demolish_and_respawn_spot(rs):
    a = rs.Arena(round_flow=True)
    car = a.add_car(rs.Team.BLUE)
    a.add_car(rs.Team.ORANGE)
    car.demolish_now(0.5)
    assert car.get_state().is_demoed and car.get_body() is None
    car.queue_respawn(rs.Vec(-2304, -4608, 17), (0, 16384, 0))
    a.step(80)
    s = car.get_state()
    assert not s.is_demoed and abs(s.pos.x + 2304) < 1 and abs(s.pos.y + 4608) < 1


def test_pads_and_boost(rs):
    a = rs.Arena()
    car = a.add_car(rs.Team.BLUE)
    pad = a.get_boost_pads()[3]
    pad.set_state(rs.BoostPadState(False, 2.5))
    st = pad.get_state()
    assert not st.is_active and abs(st.cooldown - 2.5) < 1e-6
    before = car.get_state().boost
    a.set_pad_rule(False)
    car.add_boost(12)
    assert abs(car.get_state().boost - (before + 12)) < 1e-3
