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

    Broad & Greenwood's bulb-out proposal moves four kerbs and takes ~1,090 sq ft of roadway into
    the corners, so a frame measured from the resolved pavement would tighten between the two
    panels - which is why the frame is measured from the model instead.
    """
    from src.geometry.model import build_pavement_polygon
    from src.geometry.treatments import DesignState
    from src.render.plan_view import plot_design_state
    from src.site import load_site_scenarios, run_scenario

    model = site_models["broad_st_greenwood"]
    baseline = DesignState.from_model(model)
    with contextlib.redirect_stdout(io.StringIO()):
        proposed = run_scenario(
            load_site_scenarios("broad_st_greenwood").build_proposal_apron_bulbouts,
            baseline, model)
    assert (build_pavement_polygon(proposed.corner_fillets).area
            < build_pavement_polygon(baseline.corner_fillets).area - 500), (
        "this proposal is supposed to take roadway into the corners; if it no longer does, it "
        "cannot show that the frame ignores the treated pavement")

    limits = []
    for state in (baseline, proposed):
        fig, ax = plt.subplots()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                plot_design_state(ax, model, state, "panel", dimension_labels=False)
            limits.append((ax.get_xlim(), ax.get_ylim()))
        finally:
            plt.close(fig)

    (bx, by), (ax_, ay) = limits
    assert bx == pytest.approx(ax_, abs=TOL_FT), "the two panels of a before/after differ in x"
    assert by == pytest.approx(ay, abs=TOL_FT), "the two panels of a before/after differ in y"
