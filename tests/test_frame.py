"""The two views have to be pointed at the same ground.

The plan view and the 3D render are the same reconstruction drawn twice, so they are read side by
side - and each of them used to decide its own frame: a hardcoded 110 ft square on the junction
node in 2D, the pavement's own clipped extent in 3D. Nothing compared them, and on the four sites
they disagreed by 1.15-1.57x and by 6.5-12.5 ft of centre.
"""
import contextlib
import io
import json
import math
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shapely.geometry import Point

from src.render.coords import FT_TO_M
from src.render.frame import LEG_REACH_TOLERANCE
from tests.conftest import SITES, needs_source_data

TOL_FT = 0.01


def _export(model, state, name, out_path):
    from src.render.export import export_scenario
    from src.sources.osm_context import fetch_crossings

    with contextlib.redirect_stdout(io.StringIO()):
        crossings = fetch_crossings(model.center_wgs84, radius_m=130)
        path = export_scenario(model, state, name, out_path, buildings=[], crossings=crossings,
                               theme={})
    return json.loads(Path(path).read_text())


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_both_views_frame_the_same_ground(site, site_models, tmp_path):
    """The plan view's axes and the render's camera cover one square of ground, to the foot."""
    from src.geometry.treatments import DesignState
    from src.render.plan_view import plot_design_state

    model = site_models[site]
    state = DesignState.from_model(model)
    data = _export(model, state, "existing", tmp_path / f"{site}.json")

    fig, ax = plt.subplots()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            plot_design_state(ax, model, state, "existing", dimension_labels=False)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
    finally:
        plt.close(fig)

    plan_radius_ft = max(xmax - xmin, ymax - ymin) / 2
    # The render's frame is local metres about the junction; the plan view's is state plane feet.
    render_centre_ft = (model.center_ft.x + data["frame"]["center_m"][0] / FT_TO_M,
                        model.center_ft.y + data["frame"]["center_m"][1] / FT_TO_M)
    render_radius_ft = data["frame"]["radius_m"] / FT_TO_M

    assert plan_radius_ft == pytest.approx(render_radius_ft, abs=TOL_FT), (
        f"{site}: the plan view takes in {plan_radius_ft:.1f} ft where the render takes in "
        f"{render_radius_ft:.1f} ft ({plan_radius_ft / render_radius_ft:.2f}x), so the two "
        f"pictures do not show the same street")
    assert (xmin + xmax) / 2 == pytest.approx(render_centre_ft[0], abs=TOL_FT)
    assert (ymin + ymax) / 2 == pytest.approx(render_centre_ft[1], abs=TOL_FT)


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_the_shared_frame_is_the_one_the_render_computed_for_itself(site, site_models, tmp_path):
    """Adopting one frame moved the plan view, not the camera.

    The definition (the modelled pavement's extent, clipped at the legs' reach, plus a margin) is
    the one blender_scene.py had already worked out; src/render/frame.py only moved it to the side
    of the boundary that can be tested. This recomputes it the way blender_scene.py's fallback
    still does, from the exported pavement, and it has to come out the same.
    """
    from src.geometry.treatments import DesignState

    model = site_models[site]
    data = _export(model, DesignState.from_model(model), "existing", tmp_path / f"{site}.json")

    rings = data.get("pavement_near", []) + data.get("pavement_far", [])
    xs = [x for ring in rings for x, _y in ring]
    ys = [y for ring in rings for _x, y in ring]
    reach = max((math.hypot(*leg["far_m"]) for leg in data["legs"]), default=0.0)
    framed = [(x, y) for x, y in zip(xs, ys)
              if not reach or math.hypot(x, y) <= reach * LEG_REACH_TOLERANCE]
    fx = [x for x, _y in framed] or xs
    fy = [y for _x, y in framed] or ys

    assert data["frame"]["center_m"][0] == pytest.approx((min(fx) + max(fx)) / 2, abs=0.01)
    assert data["frame"]["center_m"][1] == pytest.approx((min(fy) + max(fy)) / 2, abs=0.01)
    assert data["frame"]["radius_m"] == pytest.approx(
        max(max(fx) - min(fx), max(fy) - min(fy)) / 2 * 1.2, abs=0.01)


@needs_source_data
def test_a_proposal_frames_the_same_ground_as_its_baseline(site_models):
    """A before/after pair is only comparable if both panels cover the same square.

    The kerb-moving state is built HERE rather than borrowed from a site's scenario list. It
    used to be Broad & Greenwood's bulb-out proposal - the only scenario in the repo that moved
    a kerb - so deleting that proposal would have quietly left this test with nothing to prove:
    every remaining scenario is paint-only, and a frame cannot shift when no kerb does. The
    subject of this test is the frame, not any one proposal, so it makes its own.
    """
    from src.geometry.model import build_pavement_polygon
    from src.geometry.treatments import DesignState, bulb_out_corner_pair
    from src.render.plan_view import plot_design_state

    model = site_models["broad_st_greenwood"]
    baseline = DesignState.from_model(model)
    with contextlib.redirect_stdout(io.StringIO()):
        proposed = baseline
        for leg_name in ("broad_st_east", "broad_st_west"):
            proposed = bulb_out_corner_pair(proposed, leg_name, extension_ft=8.0,
                                             crossing_ft=25.0)
    assert (build_pavement_polygon(proposed.corner_fillets).area
            < build_pavement_polygon(baseline.corner_fillets).area - 500), (
        "this state is supposed to take roadway into the corners; if it no longer does, it "
        "cannot exercise what this test is about")

    frames = []
    for state, label in ((baseline, "baseline"), (proposed, "proposed")):
        fig, ax = plt.subplots()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                plot_design_state(ax, model, state, label, dimension_labels=False)
            frames.append((ax.get_xlim(), ax.get_ylim()))
        finally:
            plt.close(fig)
    assert [v for pair in frames[0] for v in pair] == pytest.approx(
        [v for pair in frames[1] for v in pair], abs=0.01), (
        "the two panels of a before/after pair frame different ground, so the drawing invites "
        "a comparison it does not support")


@needs_source_data
@pytest.mark.parametrize("site", SITES)
def test_the_centerline_paint_follows_the_road_in_both_views(site, site_models, tmp_path):
    """The double yellow is the same stripe in the plan view and in the render.

    The render used to be handed the leg's near and far points and draw a straight stripe between
    them - the CHORD. Ten of this project's twelve legs are straight 2-vertex lines, so it looked
    right nearly everywhere; on the two that are not, it is 3.98 ft out on broad_st_east and 7.58
    ft on louellen_st_west. That is most of a lane's width of asphalt moved from one side of the
    road to the other, and it put the centerline where the stop bar it meets is not.

    Checked as "every exported vertex lies on the leg's own centerline, half the gap away from
    it", which is what following the road means, and separately that the two views produce the
    same lines - the chord passes neither.
    """
    from src.geometry.treatments import DesignState
    from src.render.crosswalks import (DOUBLE_YELLOW_GAP_FT, centerline_paint_ft,
                                       centerline_start_ft)
    from src.render.scene import SceneGeometry
    from src.sources.osm_context import fetch_crossings

    model = site_models[site]
    state = DesignState.from_model(model)
    data = _export(model, state, "existing", tmp_path / f"{site}.json")
    with contextlib.redirect_stdout(io.StringIO()):
        scene = SceneGeometry.resolve(model, state,
                                      fetch_crossings(model.center_wgs84, radius_m=130))

    painted_legs = 0
    for exported in data["legs"]:
        leg = state.legs[exported["name"]]
        lines = exported["centerline_paint_m"]
        style = exported["centerline_style"]
        if style == "none":
            assert not lines, f"{exported['name']} is styled 'none' but paint was exported"
            continue
        if not lines:
            continue        # paint that starts past the end of the leg - see centerline_start_ft
        painted_legs += 1
        if style == "double_yellow":
            assert len(lines) == 2, f"a double yellow is two stripes, got {len(lines)}"

        for line in lines:
            for x_m, y_m in line:
                point = Point(model.center_ft.x + x_m / FT_TO_M,
                              model.center_ft.y + y_m / FT_TO_M)
                off_ft = leg.centerline.distance(point)
                expected_ft = DOUBLE_YELLOW_GAP_FT / 2 if style == "double_yellow" else 0.0
                assert off_ft == pytest.approx(expected_ft, abs=0.35), (
                    f"{exported['name']}: a centerline vertex sits {off_ft:.2f} ft off the "
                    f"leg's centerline where the paint should be {expected_ft:.2f} ft off it - "
                    f"the stripe is not following the road")

        # And the plan view draws these exact lines, from the same call.
        start_ft = centerline_start_ft(scene.crosswalk_offsets[exported["name"]].offset_ft,
                                        scene.stop_bar_offsets.get(exported["name"]),
                                        exported["name"] in scene.marked_crosswalks)
        drawn = centerline_paint_ft(leg, start_ft, style)
        assert len(drawn) == len(lines), (
            f"{exported['name']}: the plan view draws {len(drawn)} stripe(s) where the render "
            f"gets {len(lines)}")

    assert painted_legs, f"{site} exported no centerline paint at all, so this proves nothing"
