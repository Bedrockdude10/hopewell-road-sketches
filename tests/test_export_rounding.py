"""The exported geometry JSON is rounded, so a changed line means a changed shape.

Without this the export carries repr(float) - 17 significant digits, the last three of which
are float64 noise on a ~500,000 ft state-plane coordinate. Any change to the order of
operations upstream perturbs every vertex below the micron and rewrites the whole file: one
30-line edit to src/geometry/model/ produced 53,394 lines of geometry diff, none of it a real
movement. A diff in which every line changed cannot answer "did I move geometry I did not mean
to move", which is the only question worth asking of a rendered artifact under version control.

So what is asserted here is not "numbers are shorter". It is the two properties that make the
diff mean something - noise below EXPORT_DECIMALS is DISCARDED, and movement above it SURVIVES -
plus the structural things a naive recursive rounder gets wrong.
"""
import json

from src.render.coords import EXPORT_DECIMALS, round_for_export


def test_noise_below_the_floor_is_discarded():
    """The property the rounder exists for: two runs that differ only in float64 noise produce
    BYTE-IDENTICAL json. This is the thing that was false before."""
    one = {"coords": [[12.340000000000001, -5.6789012345678]]}
    other = {"coords": [[12.339999999999998, -5.678901234567802]]}
    assert one != other
    assert json.dumps(round_for_export(one)) == json.dumps(round_for_export(other))


def test_real_movement_survives():
    """The other half, and the one that makes the first half safe. A rounder that collapsed a
    genuine difference would hide exactly the bug the diff is being read for - so a shift one
    decade ABOVE the floor has to still show up."""
    before = round_for_export({"x": 12.3400000})
    after = round_for_export({"x": 12.3400100})
    assert before != after


def test_the_floor_is_finer_than_anything_visible():
    """EXPORT_DECIMALS is a claim about metres, so it is worth stating what it buys rather than
    leaving 6 as a bare number: a micron, which is 1000x finer than the millimetre at which two
    renders could differ visibly, and - because crosswalk_axis is a UNIT VECTOR, where the same
    absolute rounding reads as an angle - 0.1 mm of drift over a 100 m leg."""
    assert 10 ** -EXPORT_DECIMALS <= 1e-6
    assert (10 ** -EXPORT_DECIMALS) * 100 <= 1e-4      # metres of error at 100 m, via the axis


def test_ints_stay_ints():
    """`faces` are vertex indices into `vertices_m`. A `.0` on one is a malformed mesh.

    Weaker than the two above, and worth saying so: `round(int, n)` returns an int in Python, so
    the obvious recursive rounder passes this too. What it does catch is the OTHER obvious
    implementation - formatting every number through `%.6f` or a json float encoder, which is
    how you would shrink the file without a traversal, and which writes `0.000000` here."""
    rounded = round_for_export({"faces": [[0, 1, 2]], "vertices_m": [[1.5, 2.5, 0.0]]})
    assert json.dumps(rounded["faces"]) == "[[0, 1, 2]]"
    assert all(isinstance(i, int) for face in rounded["faces"] for i in face)


def test_bools_and_none_and_strings_pass_through():
    """`confirmed`, `surveyed` and `mesh` are bools; `height_source` and `kerb` are strings; a
    leg with no matched crossing exports None rather than a number. isinstance(True, int) is
    true, which is how a bool ends up as 1 in a document like this."""
    payload = {"confirmed": True, "mesh": False, "kerb": "raised",
               "crosswalk_reach_left_m": None, "notes": ["LaneNarrowing(...)"]}
    assert round_for_export(payload) == payload
    assert round_for_export(payload)["confirmed"] is True


def test_it_reaches_all_the_way_down():
    """The export is dicts of lists of lists of pairs, and `theme` is a nested dict of its own.
    Rounding only the top level would silently miss every coordinate in the file."""
    nested = {"legs": [{"centerline_paint_m": [[[1.23456789, 2.3456789]]]}]}
    out = round_for_export(nested)
    assert out["legs"][0]["centerline_paint_m"][0][0] == [1.234568, 2.345679]


def test_tuples_become_lists():
    """pt_to_local_m returns a list but several callers build tuples, and json.dump writes both
    as arrays - so normalising here keeps the file's form independent of which one a caller used."""
    assert round_for_export({"axis": (0.5, 0.5)}) == {"axis": [0.5, 0.5]}
