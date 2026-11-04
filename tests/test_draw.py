"""The random-draw model's settings from the dumped item list (no game data needed: a made-up items.txt)."""
import pytest

from catsim.draw import Items, Loadout, NotCovered, settings

ITEMS = """catsim-items 1
[items]
23 body Body_Octane 1 ok
31 body Body_Venom 1 ok
4842 wheels wheel_yinyangcats 4 ok
376 wheels WHEEL_Star 0 ok
12019 antenna antenna_cattoy 1 ok
1 antenna Antenna_8Ball 1 ok
12697 topper hat_cosmickitty 0 ok
12660 goal_explosion explosion_s23catportal 1 ok
4994 boost boost_fish 0 ok
5337 decal Skin_octane_FeralCat 0 ok
8969 trail ss_stringyball 0 ok
9999 topper hat_strange 0 unmodeled:attachment_ParticleSystem
[maps]
Stadium_P 0
UF_Night_P 5
"""

CATS = Loadout(car_id=23, decal_id=5337, wheels_id=4842, boost_id=4994, antenna_id=12019, hat_id=12697,
                   trails_id=8969, goal_explosion_id=12660)


@pytest.fixture
def items(tmp_path):
    p = tmp_path / "items.txt"
    p.write_text(ITEMS, encoding="utf-8")
    return Items(p)


def test_cat_loadout_on_uf_night(items):
    s = settings(items, "UF_Night_P", [CATS] * 4)
    assert s == {"wheels": [4], "extras": [1], "map": [5], "goal_mesh": [1], "goal": [3, 7], "demolish": [4, 6],
                 "reset": [2, 7], "respawn": [2, 3]}


def test_default_loadout(items):
    s = settings(items, "Stadium_P", [Loadout(car_id=23), Loadout(car_id=31)])
    assert (s["wheels"], s["extras"], s["map"], s["goal_mesh"]) == ([0], [0], [0], [0])


def test_antenna_stick_counts_as_a_part(items):
    s = settings(items, "Stadium_P", [Loadout(car_id=23, wheels_id=376, antenna_id=1)])
    assert (s["wheels"], s["extras"]) == ([0], [1])


@pytest.mark.parametrize("cars, why", [
    ([Loadout(car_id=23, hat_id=9999)], "ParticleSystem"),
    ([Loadout(car_id=23, hat_id=4242)], "not in the dumped product list"),
    ([Loadout(car_id=23, hat_id=4842)], "is a wheels, not a topper"),
    ([CATS, Loadout(car_id=23)], "differ"),
])
def test_not_covered(items, cars, why):
    with pytest.raises(NotCovered, match=why):
        settings(items, "UF_Night_P", cars)


def test_unknown_map(items):
    with pytest.raises(NotCovered, match="map"):
        settings(items, "Beckwith_P", [CATS])


def test_loadout_from_rlbot_dict():
    lo = Loadout.from_dict({"car_id": 23, "wheels_id": "4842", "team_color_id": 35, "custom_color_id": 0})
    assert (lo.car_id, lo.wheels_id, lo.hat_id) == (23, 4842, 0)
