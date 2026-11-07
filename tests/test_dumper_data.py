"""The dumper's versions and the data it writes: the known builds, what it says on an unknown one, the checked-data
check (on synthetic folders), and, with CATSIM_DATA set to a folder made by `python -m catsim.dump`, that folder."""
import json
import os
import re
from pathlib import Path

import pytest

from catsim.dump import check, versions
from catsim.dump.f32 import C002, bits, frombits, mul
from catsim.dump.tables import PRESET_FOR_PRESETLESS, _hundredths

DATA = os.environ.get("CATSIM_DATA")
needs_data = pytest.mark.skipif(not DATA, reason="set CATSIM_DATA to a folder made by `python -m catsim.dump`")


def test_known_versions_list_every_package():
    assert len(set(versions.PACKAGES)) == len(versions.PACKAGES)
    seen = []
    for label, want in versions.KNOWN.items():
        assert set(want) == set(versions.PACKAGES), label
        assert all(re.fullmatch(f"[0-9a-f]{{{versions.PREFIX}}}", h) for h in want.values()), label
        assert want not in seen, f"{label} has the same hashes as another version"
        seen.append(want)


def test_checked_data_names_every_physics_file():
    from catsim.dump import DATA_FORMAT
    assert versions.DATA_FILES[0] == "tables.txt" and len(versions.DATA_FILES) == 18
    for label, want in versions.CHECKED_DATA.items():
        assert set(want) == set(versions.DATA_FILES), label
        assert all(re.fullmatch(f"[0-9a-f]{{{versions.PREFIX}}}", h) for h in want.values()), label
    # a new data format makes new files: the checked data has to be made again for it
    assert any(f"data format {DATA_FORMAT})" in label for label in versions.CHECKED_DATA)


def test_unknown_version_says_what_changed_and_what_to_do():
    label, want = next(iter(versions.KNOWN.items()))
    packages = {n: h + "0" * 48 for n, h in want.items()}
    packages["TAGame.upk"] = "f" * 64
    packages["Startup.upk"] = "e" * 64
    fp = versions.Fingerprint(packages, "1.0.99999.0", versions.match(packages))
    assert fp.label is None
    versions.require_known(fp, allow_unknown=True)             # allowed: no error
    with pytest.raises(versions.UnknownVersion) as e:
        versions.require_known(fp, allow_unknown=False)
    msg = str(e.value)
    assert "1.0.99999.0" in msg and label in msg
    assert "2 of the 23 packages" in msg and "Startup.upk: eeee" in msg and "TAGame.upk: ffff" in msg
    assert "Stadium_P.upk" not in msg                           # only the packages that changed are listed
    assert "pip install -U" in msg and "--allow-unknown-version" in msg and versions.ISSUES in msg
    # the same hashes again are that version
    assert versions.match({n: h + "0" * 48 for n, h in want.items()}) == label


def fake_data(folder: Path) -> dict:
    """A data folder with every physics file (made-up contents) and CHECKED_DATA entries that name it."""
    (folder / "soccar").mkdir(parents=True)
    for i, n in enumerate(versions.DATA_FILES):
        (folder / n).write_bytes(f"file {i}\n".encode())
    want = {n: versions.sha256(folder / n)[:versions.PREFIX] for n in versions.DATA_FILES}
    files = {n: versions.sha256(folder / n) for n in versions.DATA_FILES}
    (folder / "catsim_data.json").write_text(json.dumps({"format": 1, "dumper": "catsim test", "game": {}, "files": files}))
    return {"the test data": want}


def test_check_data_tells_the_checked_data_from_anything_else(tmp_path, monkeypatch):
    monkeypatch.setattr(versions, "CHECKED_DATA", fake_data(tmp_path))
    c = versions.check_data(tmp_path)
    assert c.label == "the test data" and "the same files as the test data" in c.describe()
    # One changed byte in the tables: not the checked data, and it says which file.
    t = tmp_path / "tables.txt"
    t.write_bytes(t.read_bytes()[:-1] + b"!")
    c = versions.check_data(tmp_path)
    assert c.label is None and c.differ == ["tables.txt"] and c.missing == []
    assert c.describe().startswith("NOT the checked data") and "tables.txt" in c.describe()
    (tmp_path / "soccar" / "mesh_15.cmf").unlink()
    c = versions.check_data(tmp_path)
    assert c.differ == ["tables.txt"] and c.missing == ["soccar/mesh_15.cmf"]


def test_check_reports_the_maker_and_edits(tmp_path, monkeypatch):
    monkeypatch.setattr(versions, "CHECKED_DATA", fake_data(tmp_path))
    out = []
    assert check(tmp_path, log=out.append) is True
    assert any("made by: catsim test" in x for x in out)
    m = json.loads((tmp_path / "catsim_data.json").read_text())
    del m["dumper"]                                             # what 0.1.0 and 0.1.1 wrote
    (tmp_path / "catsim_data.json").write_text(json.dumps(m))
    (tmp_path / "soccar" / "arena.txt").write_bytes(b"edited")
    out = []
    assert check(tmp_path, log=out.append) is False
    text = "\n".join(out)
    assert "older than catsim 0.1.2" in text
    assert "changed or missing since the dump: soccar/arena.txt" in text and "NOT the checked data" in text


# ---------------------------------------------------------------- a real data folder

def tables(folder: Path) -> dict[str, list[list[str]]]:
    t, cur = {}, None
    for line in (folder / "tables.txt").read_text(encoding="ascii").splitlines():
        if line.startswith("[") and line.endswith("]"):
            cur = t.setdefault(line[1:-1], [])
        elif cur is not None and line and not line.startswith("#"):
            cur.append(line.split())
    return t


@needs_data
def test_data_folder_is_the_checked_data():
    """A dump from any checked build is the checked data, file for file. A folder from an older or patched
    dumper, a hand-edited table or a game version nobody has checked fails here."""
    c = versions.check_data(Path(DATA))
    assert c.label, c.describe()


@needs_data
def test_presetless_body_runs_on_its_base_preset_with_rounded_rests():
    """The Darkcar row: Plank's box, offsets, hit distance, wheel radii and suspension, with the wheel rest
    positions rounded to 0.01 uu, and the wheels' connection points moved with them."""
    t = tables(Path(DATA))
    names = [r[0][2:] for r in t["presets"]]
    rows = {r[0][2:]: [int(w, 16) for w in r[1:]] for r in t["presets"]}
    for body, base in PRESET_FOR_PRESETLESS.items():
        d, p = rows[body], rows[base]
        assert d[:6] == p[:6]                                    # box half extents, offsets, hit distance
        for axle, ax in (("front", 6), ("back", 13)):            # 7 words per axle
            cx, cy, cz, rest, radius, rx, ry = range(ax, ax + 7)
            assert (d[cz], d[rest], d[radius]) == (p[cz], p[rest], p[radius])
            for w in (rx, ry):
                assert d[w] == bits(_hundredths(frombits(p[w]))), f"{body}'s {axle} rest is not {base}'s rounded"
            for c, r in ((cx, rx), (cy, ry)):
                if p[c] == bits(mul(frombits(p[r]), C002)):      # where the base's wheel isn't clamped to the box
                    assert d[c] == bits(mul(frombits(d[r]), C002))
        if (d[11], d[12], d[18]) != (p[11], p[12], p[18]):       # its own rests fold its own steer curves
            for curve in ("steer_folded", "handbrake_folded"):
                assert t[curve][names.index(body)] != t[curve][names.index(base)], curve
