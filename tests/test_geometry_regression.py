"""Golden-file regression over the exported geometry, per site and per scenario.

Every other test in this suite asserts a property someone thought to state. This one asserts
nothing about the geometry except that it is the SAME geometry - which is the only guard that
covers the changes nobody predicted. A refactor that "shouldn't change the output" is the
normal way a crosswalk moves 4 ft, and the export JSON is what the 3D render actually
consumes, so drift here is drift in the picture.

WHY A DIGEST AND NOT THE EXPORT ITSELF. The real files are 0.7-1.1 MB each, and eight of them
committed as goldens would be ~6 MB of JSON that regenerates wholesale on any geometry change.
A diff nobody can read is a diff nobody reviews, and "regenerate and eyeball the diff" IS the
workflow this test exists to enable - so what gets committed is a summary small enough to read
in full: every leg's frame, every prop's position, and a count-and-bounding-box per drawing
channel. That catches a marking that moved, a prop that vanished, a channel that emptied, and
a pavement that changed extent, while staying stable against float noise.

The complement is scripts/diff_exports.py, which compares two directories of the full exports
key by key. This says WHETHER something changed, in CI, without being asked; that says WHAT
changed, in detail, once you know to look. Neither replaces the other.

WHEN THIS FAILS, IT IS NOT NECESSARILY WRONG. A changed render is fine; an unexplained one is
not. Read the diff, satisfy yourself that every moved number is a thing you meant to move,
then regenerate:

    ./scripts/test.sh tests/test_geometry_regression.py --force-regen

and commit the updated goldens IN THE SAME COMMIT as the change that moved them, so the diff
is reviewable next to its cause.
"""
import contextlib
import io
import json
import tempfile
from pathlib import Path

import pytest

from src.geometry.treatments import DesignState
from src.geometry.markings import CHANNELS
from src.render.export import export_scenario
from src.site import load_site_scenarios, run_scenario
from tests.conftest import SITES, needs_source_data

# Metres, so this is a millimetre - far below any distance this project draws or argues about,
# and coarse enough that the same arithmetic in a different order lands on the same number.
# scripts/diff_exports.py uses 1e-4 for the same reason; this is rounded rather than compared
# with a tolerance, so it sits a decade coarser to stay off the rounding boundary.
#
# THIS IS NOW THE SECOND ROUNDING, not the first: the export itself quantises to
# coords.EXPORT_DECIMALS (1 um) so its diffs mean something, and this digests that. Rounding
# twice is not the same as rounding once, and one vertex found the seam - broad_st_east's
# centreline paint is 21.396500330568525 m raw, which is 21.397 at 3 places, but 21.3965 at 6
# places, which is an exact tie that round-half-to-even sends to 21.396. A 1 mm move in the
# digest, no move in the geometry.
#
# Left as it is rather than papered over. Quantising first makes the pipeline DETERMINISTIC at
# this precision, so this can only bite once per vertex ever - whereas the raw value drifting
# across a 3-place boundary was possible on any run before. If a golden regenerates with a
# single 1 mm coordinate and nothing else, this is the reason to check for first.
PLACES = 3

# Channels that are lists of polylines/polygons: summarised by shape rather than listed. A
# name here that is missing from an export is itself a finding - see _digest.
# Context geometry - the street itself rather than anything painted on it. Listed, because
# these are not markings and there is no registry to derive them from.
CONTEXT_CHANNELS = (
    "kerbs", "pavement_near", "pavement_far", "sidewalks_near", "sidewalks_far",
    "paved_surfaces", "corner_parcels", "refuge_islands", "raised_crossings", "tree_points",
)

# EVERY PAINT CHANNEL, DERIVED FROM THE REGISTRY rather than listed. A hardcoded list is a second
# record of which markings exist, and it drifted the first time it was tested: the two-way lane's
# contraflow stripe was declared in src/geometry/markings.py, exported with 30 segments, drawn in
# both views - and absent from this tuple, so it had no golden at all. Derived, a new marking
# cannot be added without one, which is the same argument markings.CHANNELS already wins for
# export.PAINT_KIND_LISTS.
POLYLINE_CHANNELS = CONTEXT_CHANNELS + tuple(channel.key for channel in CHANNELS)

# Per-leg fields worth pinning: the frame every marking on that leg is placed in. A crosswalk
# drawn off the wrong axis is the specific 2D/3D disagreement this project has already shipped
# once (see scripts/blender/blender_scene.py:_marking_frame).
LEG_FIELDS = ("width_m", "near_m", "far_m", "crosswalk_centre_m", "crosswalk_axis",
              "crosswalk_offset_m", "crosswalk_style", "stop_bar_centre_m", "stop_bar_axis",
              # A bar's SPAN and LATERAL OFFSET, not just the frame it sits in. Pinning the frame
              # alone said where the bar is and nothing about how far across the road it reaches,
              # so changing where it starts - from the alignment to the painted centreline, which
              # moved it 3.15 ft on broad_st_east - left every golden identical. A marking's
              # extent is as much of the drawn result as its position.
              "stop_bar_span_m", "stop_bar_lateral_offset_m",
              # THE CENTRELINE'S ACTUAL DRAWN GEOMETRY, which had no golden coverage at all until
              # a sign error moved it 2.84 ft on broad_st_west - onto the wrong side of the
              # alignment, 8.16 ft from one kerb-side edge and 13.84 from the other, while every
              # check reported two 11.00 ft lanes. Nothing failed, because every check measured
              # the divider the design INTENDED and none compared it against the line drawn.
              #
              # This is the marking src/render/crosswalks.py documents at
              # `centerline_paint_ft`, where the 3D render drew the leg's chord and was up to
              # 7.58 ft out. Twice now the double yellow has moved feet with the suite green.
              "centerline_paint_m",
              # THE SWEEP FOR EXTENT-WITHOUT-FRAME, after the stop bar and the centreline each
              # turned out to be pinned by position and not by size. A crosswalk had the same
              # shape of hole: its centre and axis were pinned and how far it REACHES was not, so
              # it could stretch, shrink or change bar count silently.
              "crosswalk_reach_left_m", "crosswalk_reach_right_m", "crosswalk_skew_deg",
              "crosswalk_bar_count",
              # Where the centreline paint starts, and in which style. "double_yellow" quietly
              # becoming "none" would erase a marking from both views with nothing to say so.
              "centerline_start_m", "centerline_style",
              # The stop bar's station and the width it is sized against - the two inputs to the
              # span already pinned above.
              "stop_bar_offset_m", "stop_bar_width_m",
              # PROVENANCE, not geometry, and pinned for that reason: these say whether a
              # crossing sits where OSM surveyed it or where this project guessed, and whether a
              # width was field-measured. A render silently downgrading from surveyed to
              # estimated is asserting something weaker about itself than it did yesterday.
              "crosswalk_offset_source", "confirmed")


def _round(value):
    """Round every number anywhere inside `value`, preserving structure."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        # `+ 0.0` normalises -0.0, which round() preserves and YAML then writes as `-0.0`,
        # producing a golden diff for a sign that means nothing.
        return round(value, PLACES) + 0.0
    if isinstance(value, list):
        return [_round(item) for item in value]
    if isinstance(value, dict):
        return {k: _round(v) for k, v in value.items()}
    return value


def _numbers(value) -> list[float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        return [n for item in value for n in _numbers(item)]
    if isinstance(value, dict):
        return [n for _k, v in sorted(value.items()) for n in _numbers(v)]
    return []


def _extent(value) -> dict | None:
    """Bounding box of a channel, from its coordinate pairs.

    A count alone says a channel still has 40 polylines; the extent says they are still in the
    same place. Together they catch both "a marking disappeared" and "every marking shifted",
    which counts alone would miss entirely.
    """
    coords = list(_walk_points(value))
    if not coords:
        return None
    xs = [x for x, _ in coords]
    ys = [y for _, y in coords]
    return {"min": [round(min(xs), PLACES) + 0.0, round(min(ys), PLACES) + 0.0],
            "max": [round(max(xs), PLACES) + 0.0, round(max(ys), PLACES) + 0.0]}


def _walk_points(value):
    """Every [x, y] pair anywhere inside `value`. A point is a list of 2+ numbers whose first
    two entries are numbers - which is how every coordinate in this export is written."""
    if isinstance(value, dict):
        for _k, v in sorted(value.items()):
            yield from _walk_points(v)
    elif isinstance(value, list):
        if (len(value) >= 2 and all(isinstance(n, (int, float)) and not isinstance(n, bool)
                                    for n in value[:2])
                and not any(isinstance(n, (list, dict)) for n in value)):
            yield (float(value[0]), float(value[1]))
        else:
            for item in value:
                yield from _walk_points(item)


def _digest(export: dict) -> dict:
    """A readable summary of one exported scenario.

    Deliberately keyed by NAME rather than by position wherever a name exists - legs and props
    - because "the same 25 props in a different order" and "a prop moved" are different
    findings, and a positional golden reports the first as if it were the second. This is the
    same choice scripts/diff_exports.py makes, for the same reason.
    """
    digest = {
        "name": export.get("name"),
        "units": export.get("units"),
        "scalars": _round({k: export.get(k) for k in
                            ("crosswalk_depth_m", "stop_bar_curb_clearance_m")}),
        "existing_marked_crosswalks": sorted(export.get("existing_marked_crosswalks", [])),
        "frame": _round(export.get("frame")),
    }

    digest["legs"] = {
        leg["name"]: _round({field: leg.get(field) for field in LEG_FIELDS if field in leg})
        for leg in export.get("legs", [])
    }

    # Props carry no name, so identity is (type, position) - and the position IS the finding,
    # so it is listed rather than summarised. 25 of them at a typical site.
    digest["props"] = sorted(
        f"{p.get('type')}@{p['position_ft'][0]:.2f},{p['position_ft'][1]:.2f}"
        if isinstance(p.get("position_ft"), list) and len(p["position_ft"]) >= 2
        else f"{p.get('type')}@?"
        for p in export.get("props", [])
    )

    # Buildings are OSM context, not the subject: a count and a total vertex count is enough to
    # notice the fetch changed, without committing 79 footprints.
    buildings = export.get("buildings", [])
    digest["buildings"] = {"count": len(buildings),
                            "vertices": sum(len(_numbers(b.get("vertices_m", []))) // 2
                                            for b in buildings)}

    channels = {}
    for channel in POLYLINE_CHANNELS:
        # `is None` rather than `.get(channel, [])`: a channel that stops being exported at all
        # is a bigger finding than one that empties, and both must be visible in the golden.
        value = export.get(channel)
        channels[channel] = ("ABSENT FROM EXPORT" if value is None else
                             {"count": len(value),
                              "points": sum(1 for _ in _walk_points(value)),
                              "extent": _extent(value)})
    digest["channels"] = channels
    return digest


def _export_digest(model, state, name: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        # theme={} rather than the default: build_default_theme() downloads textures, and the
        # export only carries their file paths - which are machine-specific and would make the
        # golden unportable even if the network were available.
        path = export_scenario(model, state, name, Path(tmp) / "geometry.json", theme={})
        return _digest(json.loads(Path(path).read_text()))


@pytest.fixture(scope="module")
def digests(site_models):
    """{(site, scenario): digest} for every site, built once.

    Module-scoped and computed in one go because export_scenario is the expensive step here
    and both tests below read the same eight results.
    """
    out = {}
    for site in SITES:
        model = site_models[site]
        scenarios = load_site_scenarios(site)
        with contextlib.redirect_stdout(io.StringIO()):    # phase notes are noise here
            baseline = DesignState.from_model(model)
            out[(site, "existing")] = _export_digest(model, baseline, "Existing Conditions")
            proposed = run_scenario(scenarios.build_demo_scenario,
                                     DesignState.from_model(model), model)
            out[(site, "proposed")] = _export_digest(model, proposed, "Proposed Treatments")
            # AND THE TWO-WAY CORRIDOR, where a site has one. Covered explicitly rather than left
            # to build_demo_scenario because it is the one design here that is ASYMMETRIC about
            # the alignment - it shifts the travel lanes, moves the centreline paint off the
            # datum, and sizes a stall from what it leaves on the far kerb. Every one of those is
            # a number no other scenario exercises, so without a golden of its own the whole
            # asymmetric path had nothing to differ from, which is exactly how 30 flex posts came
            # to be drawn inside the bike lane with a green suite.
            builder = getattr(scenarios, "build_proposal_two_way_bike_lane", None)
            if builder is not None:
                two_way = run_scenario(builder, DesignState.from_model(model), model)
                out[(site, "two_way_bike_lane")] = _export_digest(model, two_way,
                                                                   "Two-Way Bike Lane")
    return out


TWO_WAY_SITES = [site for site in SITES
                 if hasattr(load_site_scenarios(site), "build_proposal_two_way_bike_lane")]


@needs_source_data
@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("scenario", ["existing", "proposed"])
def test_exported_geometry_is_unchanged(digests, data_regression, site, scenario):
    data_regression.check(digests[(site, scenario)], basename=f"{site}__{scenario}")


@needs_source_data
@pytest.mark.parametrize("site", TWO_WAY_SITES)
def test_the_two_way_corridor_geometry_is_unchanged(digests, data_regression, site):
    """A golden for the asymmetric design specifically - see the note in `digests`."""
    data_regression.check(digests[(site, "two_way_bike_lane")],
                           basename=f"{site}__two_way_bike_lane")


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_the_export_is_deterministic(digests, site_models, site):
    """Exporting the same state twice gives the same digest.

    Without this, a golden that fails intermittently reads as a real regression, and a golden
    that passes proves only that the last run matched the last regeneration. Dict iteration
    order, set iteration in the prop builders and the OSM fetch order are all places this could
    quietly not hold.
    """
    model = site_models[site]
    with contextlib.redirect_stdout(io.StringIO()):
        again = _export_digest(model, DesignState.from_model(model), "Existing Conditions")
    assert again == digests[(site, "existing")]
