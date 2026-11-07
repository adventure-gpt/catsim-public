"""Hidden state: exact copies, and moving a car's and the ball's whole state, contact memory included, into a fresh
arena. Needs CATSIM_DATA."""
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
    assert len(rs._lib().car_hidden_fields()) == len(rs.CarHidden().to_row()) == 65
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


def move(rs, a, contacts=True):
    """Bodies, hidden state, the round, the pads and (when `contacts`) the contact memory into a new arena."""
    b = arena(rs)
    b.set_round(a.get_round())
    for i, p in enumerate(a.get_boost_pads()):
        b.get_boost_pads()[i].set_timer(*p.get_timer())
    for ca, cb in zip(a.get_cars(), b.get_cars()):
        cb.set_body(ca.get_body())
        cb.set_hidden(ca.get_hidden())
    b.ball.set_body(a.ball.get_body())
    if contacts:
        b.set_contacts(a.get_contacts())
    return b


def test_move_whole_state_to_a_fresh_arena(rs):
    """With nothing touching (cars on their wheels or in the air, the ball in the air) the bodies, hidden state, round
    and pads are the whole state."""
    a = arena(rs)
    run(rs, a, 0, 200)
    a.ball.set_body(rs.BodyState(rs.Vec(0, 20, 30), (1, 0, 0, 0, 1, 0, 0, 0, 1), rs.Vec(1, -2, 30), rs.Vec(0.5, 0.25, 0)))
    run(rs, a, 200, 1)
    b = move(rs, a, contacts=False)
    assert snapshot(a) == snapshot(b)
    for t in range(201, 261):
        run(rs, a, t, 1)
        run(rs, b, t, 1)
        assert snapshot(a) == snapshot(b), f"tick {t}"


def touching(rs):
    """A car on its roof (box contacts with several cached points), one against the side wall, two meeting head on,
    the ball dropped onto the first car's wheels."""
    a = arena(rs)
    cars = a.get_cars()
    up, flip = (1, 0, 0, 0, 1, 0, 0, 0, 1), (1, 0, 0, 0, -1, 0, 0, 0, -1)
    cars[0].set_body(rs.BodyState(rs.Vec(-20, -30, 1.2), flip, rs.Vec(1, 0, 0), rs.Vec(0, 0, 1)))
    cars[1].set_body(rs.BodyState(rs.Vec(70, 0, 0.4), up, rs.Vec(20, 0, 0), rs.Vec()))
    cars[2].set_body(rs.BodyState(rs.Vec(0, 30, 0.4), (0, -1, 0, 1, 0, 0, 0, 0, 1), rs.Vec(0, -25, 0), rs.Vec()))
    cars[3].set_body(rs.BodyState(rs.Vec(0, 10, 0.4), (0, 1, 0, -1, 0, 0, 0, 0, 1), rs.Vec(0, 25, 0), rs.Vec()))
    a.ball.set_body(rs.BodyState(rs.Vec(-20, -30, 6), up, rs.Vec(0, 0, -5), rs.Vec()))
    return a


def push(rs, a, t):
    for k, c in enumerate(a.get_cars()):
        c.set_controls(rs.CarControls(throttle=1.0 if k else 0.0, steer=0.2 * ((t // 30) % 3 - 1), boost=k in (1, 2) and t % 50 < 30))
    a.step(1)


def test_contacts_round_trip(rs):
    a = touching(rs)
    for t in range(60):
        push(rs, a, t)
    m = a.get_contacts()
    assert any(len(x.points) == 4 for x in m.manifolds)
    b = move(rs, a)
    m2 = b.get_contacts()
    assert m2.boxes == m.boxes and m2.pairs == m.pairs
    assert [(x.owner, x.index, x.child, x.body0, x.body1, x.points) for x in m2.manifolds] ==         [(x.owner, x.index, x.child, x.body0, x.body1, x.points) for x in m.manifolds]


def test_move_with_contacts_is_exact(rs):
    """Cars and the ball touching the arena and each other: with the contact memory moved too, the new arena's next
    ticks are identical; without it they drift (catSim makes its contacts again from nothing)."""
    drift = 0
    for start in range(10, 160, 12):
        a = touching(rs)
        for t in range(start):
            push(rs, a, t)
        for contacts in (True, False):
            a2, b = a.copy(), move(rs, a, contacts)
            same = True
            for t in range(start, start + 90):
                push(rs, a2, t)
                push(rs, b, t)
                same = snapshot(a2) == snapshot(b)
                if not same:
                    break
            if contacts:
                assert same, f"from tick {start}, diverged at {t}"
            else:
                drift += not same
    assert drift > 0


def test_contact_point_is_the_games(rs):
    """A point made from what a recording keeps (both local points, the normal, the distance) equals the cached one in
    every field the next step reads before making it again (world positions are refreshed first)."""
    a = touching(rs)
    for t in range(60):
        push(rs, a, t)
    for x in a.get_contacts().manifolds:
        for p in x.points:
            q = a.contact_point(x.body0, x.body1, p.local_a, p.local_b, p.normal, p.distance)
            names = rs.ContactPoint.fields()
            for k in ("local_a", "local_b", "normal_b", "distance", "combined_friction", "combined_rolling",
                      "combined_restitution", "sphere_on_static"):
                for i, n in enumerate(names):
                    if n.split(".")[0] == k:
                        assert q.row[i] == p.row[i], (n, x.body0, x.body1)


def test_body_order(rs):
    """The moving bodies' object order: set and read back, kept by a copy, and its history: a demolished car leaves (the
    last body takes its place) and its respawn goes on the end."""
    a = arena(rs)
    assert a.body_order() == ["ball", "car0", "car1", "car2", "car3"]
    a.set_body_order(["ball", "car2", "car0", "car3", "car1"])
    assert a.body_order() == ["ball", "car2", "car0", "car3", "car1"]
    assert a.copy().body_order() == a.body_order()
    with pytest.raises(ValueError):
        a.set_body_order(["ball", "car0", "car1"])
    with pytest.raises(ValueError):
        a.set_body_order(["ball", "car0", "car0", "car2", "car3"])
    assert a.body_order() == ["ball", "car2", "car0", "car3", "car1"]
    b = rs.Arena(round_flow=True)
    cars = [b.add_car(rs.Team(k % 2)) for k in range(3)]
    for _ in range(600):
        b.step(1)
    cars[0].demolish_now(0.5)
    assert b.body_order() == ["ball", "car2", "car1"]
    for _ in range(120):
        b.step(1)
    assert b.body_order() == ["ball", "car2", "car1", "car0"]


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
