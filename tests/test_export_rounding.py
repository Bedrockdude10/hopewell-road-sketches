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

The second half of the file is the same argument about LINE BREAKS rather than digits, and it is
the same reasoning: rounding decides whether a changed line is a changed shape, and the line
breaks decide whether a reader can see which shape changed.
"""
import json

from src.render.coords import EXPORT_DECIMALS, dumps_for_export, round_for_export


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


# ---------------------------------------------------------------------------
# The file's SHAPE, which is the other half of the same argument. Rounding decides whether a
# changed line means a changed shape; the line breaks decide whether a reader can see WHICH
# shape. json.dump(indent=2) put every number on its own line, so a vertex spanned four lines,
# x was never beside y, and 62% of the 33 MB of committed exports was indentation.
# ---------------------------------------------------------------------------

def test_a_coordinate_is_one_line():
    """The unit a reader of this diff wants is the VERTEX. Under indent=2 a moved vertex was two
    changed lines with `],` and `[` between them, which is why the whole point of rounding -
    "a changed line is a changed shape" - stopped short of being useful."""
    out = dumps_for_export({"coords": [[-27.680958, -3.780609], [1.5, 2.5]]})
    assert "    [-27.680958, -3.780609],\n" in out
    assert out.count("\n") == 6      # {, "coords": [, two vertices, ], }, trailing


def test_structure_above_a_coordinate_is_still_indented():
    """Only the INNERMOST numeric list goes inline. Collapsing the level above it too would put a
    whole marking channel on one line, and a diff of one 40,000-character line names nothing."""
    out = dumps_for_export({"legs": [{"name": "broad_st_east", "coords": [[1.0, 2.0]]}]})
    assert '\n  "legs": [\n' in out
    assert '\n      "name": "broad_st_east",\n' in out


def test_it_parses_back_to_exactly_what_was_rounded():
    """The property that makes the reformat safe to run over 33 MB of committed exports: the
    writer is a FORMATTER, so the document it emits is the document round_for_export produced -
    every key, every value, every type. Verified against all 65 committed files before the
    reformat landed; kept here as the invariant rather than the one-off check."""
    payload = {"units": "meters", "notes": [], "theme": {}, "crosswalk_reach_left_m": None,
               "legs": [{"confirmed": True, "faces": [[0, 1, 2]],
                         "vertices_m": [[1.5, 2.5, 0.0]], "axis": (0.5, 0.5)}]}
    assert json.loads(dumps_for_export(payload)) == round_for_export(payload)


def test_empty_containers_read_the_same_as_json_dump():
    """`notes`, `tree_points` and `existing_marked_crosswalks` export present-and-empty, and an
    empty list is how a reader tells "this scenario has none" from "this file is too old to have
    the key at all" - so it has to be `[]` and not a blank line pair."""
    assert dumps_for_export({"notes": [], "theme": {}}) == '{\n  "notes": [],\n  "theme": {}\n}\n'


def test_a_row_of_bools_is_not_a_coordinate():
    """isinstance(True, int), so the test for "is this a row of numbers" is the one place a bool
    could be mistaken for one. Nothing in the export writes a bool list today; the check is here
    because the cost of it being wrong is silent."""
    assert dumps_for_export({"flags": [True, False]}) == '{\n  "flags": [\n    true,\n    false\n  ]\n}\n'


def test_the_file_ends_with_a_newline():
    """json.dump does not write one, so every committed export ended mid-line and every diff
    that touched the last building carried `\\ No newline at end of file`."""
    assert dumps_for_export({"units": "meters"}).endswith("}\n")
