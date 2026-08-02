"""Plan-view rendering: draws an IntersectionModel + DesignState to a matplotlib axis."""
import geopandas as gpd
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import LineString, Polygon

from src.geometry.model import (
    bollard_points_ft, build_pavement_polygon, corner_overlay_polygon, lane_narrowing_edge_lines_ft,
    lane_narrowing_polygons_ft, lane_narrowing_taper_ft, lane_narrowing_taper_polygons_ft, leg_clearance_ft,
    parking_lane_edge_line_ft, parking_stall_count_ft, parking_stall_lines_ft,
)
from src.geometry.intersection import IntersectionModel, kerb_lines_with_tags_ft
from src.geometry.treatments import LEGAL_PARKING_SETBACK_FT, DesignState
from src.provenance import PLOT_STYLE, leg_width_provenance
from src.render.props import build_props, data_gaps, signalization_conflicts
from src.render.coords import FT_TO_M, wgs84_to_state_plane
from src.render.crosswalks import (CROSSWALK_CLEARANCE_FT, CROSSWALK_DEPTH_M, resolve_crosswalk_offsets,
                                   resolve_crosswalk_skews, resolve_stop_bar_offsets,
                                   stop_bar_band_geometry_ft, stop_bar_width_ft)
from src.sources.osm_context import (fetch_crossings, fetch_kerbs, fetch_sidewalks,
                                     fetch_street_furniture, fetch_traffic_control)

TRAFFIC_CONTROL_RADIUS_M = 60  # matches src/render/export.py
BUILDING_CONTEXT_RADIUS_M = 130  # matches src/render/export.py - same real-world radius crossings are searched
                                  # within, so a leg's crosswalk_offset here matches what the 3D export computes


def sidewalk_lines_ft(sidewalks: list[dict] | None) -> list[LineString]:
    """Fetched OSM sidewalk ways -> state-plane LineStrings."""
    lines = []
    for walk in sidewalks or []:
        coords = walk["coords_wgs84"]
        xs, ys = wgs84_to_state_plane.transform([c[0] for c in coords], [c[1] for c in coords])
        lines.append(LineString(zip(xs, ys)))
    return lines


def _crosswalk_band(leg, offset_ft: float, depth_ft: float, skew_deg: float = 0.0,
                     span_ft: float | None = None, lateral_offset_ft: float = 0.0) -> Polygon:
    """The rectangle a painted crosswalk occupies: `depth_ft` along the leg, centered on
    `offset_ft`, spanning the leg's full curb-to-curb width. Built from the leg's own
    centerline and width, the same inputs blender_crosswalks.py uses (near + u*offset,
    then out to +/- width/2 along n), so the band drawn here is the footprint the 3D
    render will fill with stripes.

    `skew_deg` rotates the band off square to the leg, to match the orientation of the
    surveyed crossing it came from (src/render/crosswalks.py:_crossing_skew_deg). The
    span is divided by cos(skew) so a rotated band still reaches both curb lines rather
    than falling short of them - a crossing at an angle has further to go.

    `span_ft` overrides the full curb-to-curb width, and `lateral_offset_ft` shifts the
    band off the road centerline - together these draw a stop bar, which covers only the
    entering half of the roadway (see src/render/crosswalks.py:stop_bar_band_geometry_ft).
    """
    centerline = leg.centerline
    (x0, y0), (x1, y1) = centerline.coords[0], centerline.coords[1]
    length = np.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    cx, cy = x0 + ux * offset_ft, y0 + uy * offset_ft

    skew = np.radians(skew_deg)
    cos_s, sin_s = np.cos(skew), np.sin(skew)
    # Rotate the leg's own axes by the skew: n is the across-road axis the crosswalk
    # spans, u the along-travel axis its depth runs down.
    nx, ny = -uy, ux
    nx, ny = nx * cos_s - ny * sin_s, nx * sin_s + ny * cos_s
    ux, uy = ux * cos_s - uy * sin_s, ux * sin_s + uy * cos_s

    span = leg.curb_to_curb_ft if span_ft is None else span_ft
    half_w, half_d = span / (2 * max(cos_s, 0.2)), depth_ft / 2
    cx += nx * lateral_offset_ft
    cy += ny * lateral_offset_ft
    return Polygon([
        (cx + nx * half_w + ux * half_d, cy + ny * half_w + uy * half_d),
        (cx - nx * half_w + ux * half_d, cy - ny * half_w + uy * half_d),
        (cx - nx * half_w - ux * half_d, cy - ny * half_w - uy * half_d),
        (cx + nx * half_w - ux * half_d, cy + ny * half_w - uy * half_d),
    ])


def _draw_props(ax, model: IntersectionModel, state: DesignState, crosswalk_offsets: dict,
                 traffic_control: list[dict] | None, street_furniture: list[dict] | None,
                 crossings: list[dict] | None, dimension_labels: bool):
    """Draw the street furniture the 3D render will build - signals above all.

    This calls the SAME src/render/props.py:build_props the export does, so the plan view
    shows exactly the hardware Blender will place, not a second guess at it. Without
    this, nothing in the 2D reconstruction revealed whether an intersection was
    signalized, which matters twice over: three of these four junctions have signals and
    one does not, and a proposal that isn't proposing new signals must not quietly render
    them. A traffic signal appearing here that you didn't intend is now a visible error
    rather than a surprise three phases later.

    Bollards are skipped - the treatment layer above already draws them from
    state.bollard_lines, and drawing them twice would just thicken the markers.
    """
    kerb_lines = kerb_lines_with_tags_ft(model.center_wgs84, model.center_ft)
    props = build_props(model, state, crosswalk_offsets, model.center_ft, traffic_control,
                         street_furniture, crossings, fetch_kerbs(model.center_wgs84, radius_m=120))
    for line, _tags in kerb_lines:
        gpd.GeoSeries([line]).plot(ax=ax, color='black', linewidth=2.2, zorder=6)
    signal_count = 0
    for prop in props:
        kind = prop["type"]
        if kind == "bollard":
            continue
        x, y = prop["position_ft"]
        if kind == "traffic_signal_pole":
            signal_count += 1
            ax.scatter([x], [y], color="black", marker="o", s=46, zorder=7)
            ax.scatter([x], [y], color="limegreen", marker="o", s=16, zorder=7)
            # The mast arm is the part that reaches out over the roadway, and its length
            # is derived from a real leg width - worth seeing in plan, since it's the
            # most visually dominant thing in the 3D render.
            arm_deg, arm_ft = prop.get("arm_heading_deg"), prop.get("arm_length_ft")
            if arm_deg is not None and arm_ft:
                ax.plot([x, x + np.cos(np.radians(arm_deg)) * arm_ft],
                        [y, y + np.sin(np.radians(arm_deg)) * arm_ft],
                        color="black", linewidth=1.6, solid_capstyle="round", zorder=7)
        elif kind == "pedestrian_signal_head":
            ax.scatter([x], [y], color="limegreen", marker="s", s=22, edgecolors="black",
                        linewidths=0.6, zorder=7)
        elif kind == "stop_sign":
            ax.scatter([x], [y], color="red", marker="H", s=44, edgecolors="white", linewidths=0.6, zorder=7)
        elif kind == "pedestrian_pushbutton":
            ax.scatter([x], [y], color="gold", marker="P", s=30, edgecolors="black", linewidths=0.5, zorder=8)
        elif kind == "tactile_paving_pad":
            # Drawn at its true size and orientation, not as a marker: whether the pad
            # sits wholly on the footway or spills into the roadway is exactly the kind
            # of thing the plan view exists to make checkable.
            depth, width = prop.get("pad_depth_ft", 3.0), prop.get("pad_width_ft", 5.0)
            ang = np.radians(prop["heading_deg"])
            ux, uy = np.cos(ang), np.sin(ang)          # along the crossing = pad depth
            nx, ny = -uy, ux                            # along the curb = pad width
            pad = Polygon([
                (x + ux * depth / 2 + nx * width / 2, y + uy * depth / 2 + ny * width / 2),
                (x - ux * depth / 2 + nx * width / 2, y - uy * depth / 2 + ny * width / 2),
                (x - ux * depth / 2 - nx * width / 2, y - uy * depth / 2 - ny * width / 2),
                (x + ux * depth / 2 - nx * width / 2, y + uy * depth / 2 - ny * width / 2),
            ])
            gpd.GeoSeries([pad]).plot(ax=ax, color="darkorange", alpha=0.85, zorder=8)
            gpd.GeoSeries([pad]).boundary.plot(ax=ax, color="black", linewidth=0.5, zorder=8)
        elif kind == "rrfb":
            ax.scatter([x], [y], color="gold", marker="D", s=30, edgecolors="black", linewidths=0.6, zorder=8)
        elif kind == "fire_hydrant":
            ax.scatter([x], [y], color="firebrick", marker="P", s=34, zorder=7)
        elif kind == "yield_sign":
            ax.scatter([x], [y], color="white", marker="v", s=40, edgecolors="red", linewidths=1.2, zorder=7)
        elif kind == "no_turn_on_red_sign":
            ax.scatter([x], [y], color="white", marker="s", s=26, edgecolors="red", linewidths=1.2, zorder=7)
        elif kind == "streetlight":
            ax.scatter([x], [y], color="dimgrey", marker="*", s=34, zorder=6)
        else:  # site- or scenario-specific extras (school zone signs, etc.)
            ax.scatter([x], [y], color="darkgoldenrod", marker="^", s=30, zorder=7)

    if dimension_labels:
        control = (f"SIGNALIZED - {signal_count} signal pole(s)" if signal_count
                    else "NOT signalized - stop/yield control")
        ax.annotate(control, xy=(0.5, 0.005), xycoords="axes fraction", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="black" if signal_count else "dimgrey",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9))


def _draw_surveyed_crossings(ax, crossings: list[dict] | None):
    """Draw each OSM crossing way as surveyed - its real endpoints, length and skew.

    We do NOT paint the crosswalk here: a crossing way runs sidewalk-centerline to
    sidewalk-centerline, so it is consistently 8-24 ft longer than the roadway it spans
    and drawing it as the marking would run paint across the sidewalks. It goes on the
    plot as a reference line, behind the curb-to-curb band we actually draw, so the two
    can be compared by eye. Where the band sticks out past the ends of this line, the
    leg's configured width is too big - which is precisely the failure that was
    previously only arguable from the 3D render.
    """
    for line in sidewalk_lines_ft(crossings):
        gpd.GeoSeries([line]).plot(ax=ax, color="darkviolet", linewidth=1.0, linestyle=":", alpha=0.8, zorder=5)
        for end in (line.coords[0], line.coords[-1]):
            ax.scatter([end[0]], [end[1]], color="darkviolet", s=8, marker="o", zorder=5)


def _draw_crosswalks(ax, model: IntersectionModel, state: DesignState, crosswalk_offsets: dict,
                      crosswalk_skews: dict, dimension_labels: bool):
    """Draw each leg's crosswalk and stop bar exactly where the 3D export puts them.

    Gating mirrors scripts/blender/blender_scene.py precisely - a crosswalk is painted
    only on a leg listed in the config's `intersection.existing_marked_crosswalks` and
    not already carrying a raised crossing (which is drawn separately). Legs failing
    that gate still get their resolved offset drawn as a thin outline, because "this
    leg has no marked crossing" is itself a finding worth seeing in the reconstruction,
    and because a wrong offset on an unmarked leg is exactly the kind of latent error
    that only surfaces later when a proposal marks it.

    Surveyed (OSM) and estimated positions are drawn differently, following the same
    convention this view already uses for confirmed vs. estimated curb widths.
    """
    depth_ft = CROSSWALK_DEPTH_M / FT_TO_M
    marked = set(model.config["intersection"].get("existing_marked_crosswalks", []))
    raised = set(state.raised_crossings)

    for leg_name, leg in state.legs.items():
        offset_ft, source = crosswalk_offsets[leg_name]
        surveyed = source.startswith("osm_survey")
        painted = leg_name in marked and leg_name not in raised
        band = _crosswalk_band(leg, offset_ft, depth_ft, crosswalk_skews.get(leg_name, 0.0))
        edge = "darkviolet" if surveyed else "crimson"

        if painted:
            gpd.GeoSeries([band]).plot(ax=ax, color="white", alpha=0.95, zorder=4)
            gpd.GeoSeries([band]).boundary.plot(
                ax=ax, color=edge, linewidth=1.4, linestyle="-" if surveyed else "--", zorder=4)
        else:
            gpd.GeoSeries([band]).boundary.plot(ax=ax, color="grey", linewidth=0.7, linestyle=":", zorder=4)

        if dimension_labels:
            centroid = band.centroid
            note = "" if painted else "\n(unmarked)"
            ax.annotate(f"{offset_ft:.0f} ft\n{'OSM' if surveyed else 'est.'}{note}",
                        (centroid.x, centroid.y), fontsize=5.5,
                        color=edge if painted else "grey", ha="center", va="center", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))

    # Stop bars, on the same terms the export uses: signalized sites only.
    if model.config.get("signals"):
        for leg_name, stop_offset_ft in resolve_stop_bar_offsets(state, crosswalk_offsets).items():
            # A stop bar covers only the ENTERING half of the roadway - a driver stops
            # in their own lanes, never across the opposing ones. Sized and positioned
            # from the same shared rule blender_crosswalks.add_stop_bar uses, so the 2D
            # bar is the bar that gets rendered. It also inherits the crosswalk's
            # surveyed skew, being painted parallel to it.
            leg = state.legs[leg_name]
            span_ft, lateral_ft = stop_bar_band_geometry_ft(stop_bar_width_ft(state, leg_name))
            bar = _crosswalk_band(leg, stop_offset_ft, 1.5, crosswalk_skews.get(leg_name, 0.0),
                                   span_ft=span_ft, lateral_offset_ft=lateral_ft)
            gpd.GeoSeries([bar]).plot(ax=ax, color="dimgrey", alpha=0.9, zorder=4)


def plot_design_state(ax, model: IntersectionModel, state: DesignState, title: str, dimension_labels: bool = True,
                       crossings: list[dict] | None = None, sidewalks: list[dict] | None = None,
                       traffic_control: list[dict] | None = None, street_furniture: list[dict] | None = None):
    if sidewalks is None:
        try:
            sidewalks = fetch_sidewalks(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
        except RuntimeError as e:
            print(f"  WARNING: could not fetch OSM sidewalks ({e}) - drawn without them.")
            sidewalks = []

    if traffic_control is None:
        try:
            traffic_control = fetch_traffic_control(model.center_wgs84, radius_m=TRAFFIC_CONTROL_RADIUS_M)
        except RuntimeError as e:
            print(f"  WARNING: could not fetch OSM traffic control ({e}) - falling back to guesses.")
            traffic_control = []
    if street_furniture is None:
        try:
            street_furniture = fetch_street_furniture(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
        except RuntimeError:
            street_furniture = []
    for note in signalization_conflicts(model, traffic_control):
        print(f"  NOTE: {note}")

    model.parcels.boundary.plot(ax=ax, color="tan", linewidth=0.6, zorder=1)
    model.corner_parcels.boundary.plot(ax=ax, color="saddlebrown", linewidth=1.5, zorder=1)

    try:
        pavement = build_pavement_polygon(state.corner_fillets)
        gpd.GeoSeries([pavement]).plot(ax=ax, color="#d9d9d9", zorder=2)
    except ValueError:
        pass

    # Real OSM sidewalk centerlines, drawn behind everything else. These are what the
    # crossing ways actually connect to, and they bound where the curb can possibly be
    # (src/geometry/model.py:sidewalk_span_ft) - so having them on the plot is what makes
    # an over-wide leg visible instead of merely arguable.
    for walk in sidewalk_lines_ft(sidewalks):
        gpd.GeoSeries([walk]).plot(ax=ax, color="steelblue", linewidth=1.0, linestyle=(0, (4, 2)),
                                    alpha=0.65, zorder=2)

    for name, leg in state.legs.items():
        tier = leg_width_provenance(model.config["legs"][name])
        style_kw = PLOT_STYLE[tier]
        color = style_kw["color"]
        for curb in (leg.left_curb, leg.right_curb):
            gpd.GeoSeries([curb]).plot(ax=ax, linewidth=2, zorder=3, **style_kw)
        if dimension_labels:
            mid = leg.centerline.interpolate(min(leg.centerline.length * 0.85, leg.centerline.length - 5))
            ax.annotate(f"{leg.curb_to_curb_ft:.1f} ft", (mid.x, mid.y), fontsize=7, color=color,
                        ha="center", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    for corner, pieces in state.corner_fillets.items():
        if "error" in pieces:
            continue
        gpd.GeoSeries([pieces["arc"]]).plot(ax=ax, color="darkorange", linewidth=2.5, zorder=4)
        if dimension_labels and "radius_ft" in pieces:
            mid = pieces["arc"].interpolate(0.5, normalized=True)
            ax.annotate(f"R={pieces['radius_ft']:.0f} ft", (mid.x, mid.y), fontsize=7, color="darkorange",
                        fontweight="bold", ha="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    for name, island in state.refuge_islands.items():
        poly = island["polygon"]
        gpd.GeoSeries([poly]).plot(ax=ax, color="seagreen", alpha=0.6, zorder=5)
        gpd.GeoSeries([poly]).boundary.plot(ax=ax, color="darkgreen", linewidth=1, zorder=5)
        if dimension_labels:
            c = poly.centroid
            ax.annotate(f"refuge\n{island['width_ft']:.0f} ft", (c.x, c.y), fontsize=6.5, color="darkgreen",
                        ha="center", va="center", fontweight="bold")

    for name, poly in state.raised_crossings.items():
        gpd.GeoSeries([poly]).plot(ax=ax, color="slateblue", alpha=0.35, hatch="//", zorder=2)
        gpd.GeoSeries([poly]).boundary.plot(ax=ax, color="slateblue", linewidth=1, zorder=2)
        if dimension_labels:
            c = poly.centroid
            ax.annotate("raised\ncrossing", (c.x, c.y), fontsize=6.5, color="indigo",
                        ha="center", va="center", fontweight="bold")

    # Always resolved now, not just when a treatment needs them: the crosswalks ARE the
    # subject of this project, so a plan view that omits them isn't a reconstruction you
    # can check the 3D render against. Leaving them out is what let a mis-matched OSM
    # crossing (see src/render/crosswalks.py:_match_crossings_to_legs) sit at the dead
    # centre of E Broad & Princeton unnoticed - it was only visible in the 3D render.
    # fetch_crossings is disk-cached per (center, radius) in output/.cache, so the network
    # round-trip happens once per site, not once per run.
    if crossings is None:
        try:
            crossings = fetch_crossings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
        except RuntimeError as e:
            # Overpass unreachable and nothing cached. Don't fail the whole plan view for
            # context data - fall back to the geometric estimate, which is drawn in a
            # visibly different style and labeled as such, so it can't be mistaken for
            # surveyed placement.
            print(f"  WARNING: could not fetch OSM crossings ({e}) - crosswalk positions "
                  f"shown are geometric estimates, not surveyed.")
            crossings = []
    crosswalk_offsets = resolve_crosswalk_offsets(state, crossings)

    crosswalk_skews = resolve_crosswalk_skews(state, crossings)

    _draw_surveyed_crossings(ax, crossings)
    _draw_crosswalks(ax, model, state, crosswalk_offsets, crosswalk_skews, dimension_labels)
    _draw_props(ax, model, state, crosswalk_offsets, traffic_control, street_furniture, crossings,
                 dimension_labels)

    for leg_name, stripe_width_ft in state.lane_narrowing.items():
        line_only = leg_name in state.lane_narrowing_line_only
        sides = state.lane_narrowing_sides.get(leg_name, ("left", "right"))
        anchor_ft = leg_clearance_ft(leg_name, state.legs, state.corner_fillets)
        target_ft = crosswalk_offsets[leg_name][0] + CROSSWALK_CLEARANCE_FT
        leg = state.legs[leg_name]

        if line_only:
            for line in lane_narrowing_edge_lines_ft(leg, stripe_width_ft,
                                                      start_left_ft=anchor_ft, start_right_ft=anchor_ft,
                                                      sides=sides):
                gpd.GeoSeries([line]).plot(ax=ax, color="goldenrod", linewidth=1.5, zorder=3)
        else:
            for poly in lane_narrowing_polygons_ft(leg, stripe_width_ft,
                                                    start_left_ft=anchor_ft, start_right_ft=anchor_ft, sides=sides):
                gpd.GeoSeries([poly]).plot(ax=ax, color="gold", alpha=0.5, hatch="//", zorder=3)
                gpd.GeoSeries([poly]).boundary.plot(ax=ax, color="goldenrod", linewidth=1, zorder=3)
            for poly in lane_narrowing_taper_polygons_ft(leg, stripe_width_ft, anchor_ft, target_ft, sides=sides):
                gpd.GeoSeries([poly]).plot(ax=ax, color="gold", alpha=0.5, hatch="//", zorder=3)

        # The taper's own curve (src/geometry/model.py:lane_narrowing_taper_ft) - drawn either way,
        # since it's the boundary line itself (chevron-filled or not, per line_only above), the same
        # curve src/render/export.py feeds the 3D render so the two views can be checked against
        # each other directly instead of trying to eyeball it off the 3D render alone.
        for taper in lane_narrowing_taper_ft(leg, stripe_width_ft, anchor_ft, target_ft, sides=sides):
            gpd.GeoSeries([taper]).plot(ax=ax, color="goldenrod", linewidth=1.5, zorder=3)

        if dimension_labels:
            # One label PER SIDE actually narrowed, offset into that lane itself - not a single label
            # sitting on the centerline, which reads as "this road is one 11 ft lane" instead of what's
            # actually there (see sides above - not always both).
            lane_ft = leg.curb_to_curb_ft / 2 - stripe_width_ft
            along_dist = min(leg.centerline.length * 0.6, leg.centerline.length - 5)
            for side, sign in (("left", 1), ("right", -1)):
                if side not in sides:
                    continue
                lane_mid = leg.centerline.offset_curve(sign * lane_ft / 2).interpolate(along_dist)
                ax.annotate(f"lane {lane_ft:.1f} ft", (lane_mid.x, lane_mid.y), fontsize=6.5, color="goldenrod",
                            ha="center", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    for corner, depth_ft in state.corner_hatching.items():
        if "error" in state.corner_fillets[corner]:
            continue
        poly = corner_overlay_polygon(state.corner_fillets[corner], model.center_ft, depth_ft)
        gpd.GeoSeries([poly]).plot(ax=ax, color="gold", alpha=0.5, hatch="//", zorder=3)
        gpd.GeoSeries([poly]).boundary.plot(ax=ax, color="goldenrod", linewidth=1, zorder=3)

    for corner, extent_ft in state.corner_aprons.items():
        if "error" in state.corner_fillets[corner]:
            continue
        poly = corner_overlay_polygon(state.corner_fillets[corner], model.center_ft, extent_ft)
        gpd.GeoSeries([poly]).plot(ax=ax, color="peru", alpha=0.6, zorder=3)
        gpd.GeoSeries([poly]).boundary.plot(ax=ax, color="saddlebrown", linewidth=1, zorder=3)
        if dimension_labels:
            c = poly.centroid
            ax.annotate("mountable\napron", (c.x, c.y), fontsize=6, color="saddlebrown",
                        ha="center", va="center", fontweight="bold")

    for (leg_name, side), zone in state.parking_zones.items():
        leg = state.legs[leg_name]
        anchor_ft = leg_clearance_ft(leg_name, state.legs, state.corner_fillets)
        legal_start_ft = crosswalk_offsets[leg_name][0] + LEGAL_PARKING_SETBACK_FT
        parking_start_ft = max(anchor_ft, legal_start_ft)
        depth_ft, stall_length_ft, curb_offset_ft = zone["depth_ft"], zone["stall_length_ft"], zone["curb_offset_ft"]
        edge = parking_lane_edge_line_ft(leg, side, depth_ft, parking_start_ft, curb_offset_ft=curb_offset_ft)
        if edge is None:
            print(f"  NOTE: no room to mark parking on {leg_name} ({side}) - the corner return "
                  f"consumes the whole leg, so it is drawn without parking.")
            continue
        gpd.GeoSeries([edge]).plot(ax=ax, color="steelblue", linewidth=1.5, zorder=3)
        for divider in parking_stall_lines_ft(leg, side, depth_ft, stall_length_ft, parking_start_ft,
                                               curb_offset_ft=curb_offset_ft):
            gpd.GeoSeries([divider]).plot(ax=ax, color="steelblue", linewidth=1, zorder=3)
        if dimension_labels:
            n_stalls = parking_stall_count_ft(leg, stall_length_ft, parking_start_ft)
            mid = edge.interpolate(0.5, normalized=True)
            ax.annotate(f"parking\n{n_stalls} stalls ({depth_ft:.0f} ft)", (mid.x, mid.y), fontsize=6, color="steelblue",
                        ha="center", va="center", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

        # The striped no-parking buffer between the parking lane and the curb itself
        # (add_marked_parking's curb_offset_ft) - same chevron treatment as a lane-narrowing buffer,
        # just built with `sides=` restricted to this one side (see export.py's mirroring block).
        if curb_offset_ft:
            target_ft = crosswalk_offsets[leg_name][0] + CROSSWALK_CLEARANCE_FT
            for poly in lane_narrowing_polygons_ft(leg, curb_offset_ft, start_left_ft=anchor_ft,
                                                    start_right_ft=anchor_ft, sides=(side,)):
                gpd.GeoSeries([poly]).plot(ax=ax, color="gold", alpha=0.5, hatch="//", zorder=3)
                gpd.GeoSeries([poly]).boundary.plot(ax=ax, color="goldenrod", linewidth=1, zorder=3)
            for poly in lane_narrowing_taper_polygons_ft(leg, curb_offset_ft, anchor_ft, target_ft, sides=(side,)):
                gpd.GeoSeries([poly]).plot(ax=ax, color="gold", alpha=0.5, hatch="//", zorder=3)
            for taper in lane_narrowing_taper_ft(leg, curb_offset_ft, anchor_ft, target_ft, sides=(side,)):
                gpd.GeoSeries([taper]).plot(ax=ax, color="goldenrod", linewidth=1.5, zorder=3)
            if (leg_name, side) in state.parking_buffer_bollards:
                spacing_ft = state.parking_buffer_bollards[(leg_name, side)]
                points = bollard_points_ft(leg, curb_offset_ft, anchor_ft, spacing_ft, sides=(side,))
                if points:
                    xs, ys = zip(*points)
                    ax.scatter(xs, ys, color="darkorange", marker="o", s=10, zorder=6)

    for leg_name, spacing_ft in state.bollard_lines.items():
        stripe_width_ft = state.lane_narrowing[leg_name]
        start_ft = leg_clearance_ft(leg_name, state.legs, state.corner_fillets)
        points = bollard_points_ft(state.legs[leg_name], stripe_width_ft, start_ft, spacing_ft)
        if points:
            xs, ys = zip(*points)
            ax.scatter(xs, ys, color="darkorange", marker="o", s=10, zorder=6)

    ax.scatter([model.center_ft.x], [model.center_ft.y], color="blue", zorder=6, s=40)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    zoom_ft = 110
    ax.set_xlim(model.center_ft.x - zoom_ft, model.center_ft.x + zoom_ft)
    ax.set_ylim(model.center_ft.y - zoom_ft, model.center_ft.y + zoom_ft)
    ax.set_xlabel("Feet (EPSG:3424)")


def legend_handles():
    return [
        Line2D([0], [0], color="black", lw=2, label="Curb line - FIELD-MEASURED width"),
        Line2D([0], [0], color="darkviolet", lw=2, ls="-.", label="Curb line - OSM-derived width"),
        Line2D([0], [0], color="crimson", lw=2, ls="--", label="Curb line - estimated width"),
        Line2D([0], [0], color="steelblue", lw=1, ls=(0,(4,2)), label="OSM sidewalk centerline"),
        Line2D([0], [0], color="darkviolet", lw=1, ls=":", label="OSM crossing way (as surveyed)"),
        Line2D([0], [0], color="darkorange", lw=2.5, label="Corner fillet (radius labeled)"),
        Line2D([0], [0], color="seagreen", lw=6, alpha=0.6, label="Pedestrian refuge island"),
        Line2D([0], [0], color="slateblue", lw=6, alpha=0.35, label="Raised crossing"),
        Patch(facecolor="gold", alpha=0.5, hatch="//", edgecolor="goldenrod", label="Lane narrowing / corner hatching"),
        Line2D([0], [0], color="goldenrod", lw=1.5, label="Lane narrowing - line only (no chevron fill)"),
        Patch(facecolor="peru", alpha=0.6, edgecolor="saddlebrown", label="Mountable apron"),
        Line2D([0], [0], marker="o", color="darkorange", lw=0, label="Bollard"),
        Line2D([0], [0], color="steelblue", lw=1.5, label="Marked parking lane + stalls"),
        Patch(facecolor="white", edgecolor="darkviolet", label="Crosswalk - OSM-surveyed position"),
        Patch(facecolor="white", edgecolor="crimson", ls="--", label="Crosswalk - estimated position"),
        Line2D([0], [0], color="grey", lw=0.7, ls=":", label="Unmarked leg (no crosswalk today)"),
        Line2D([0], [0], color="dimgrey", lw=3, label="Stop bar (entering half only)"),
        Line2D([0], [0], marker="o", color="limegreen", markeredgecolor="black", lw=0,
                label="Traffic signal pole + mast arm"),
        Line2D([0], [0], marker="s", color="limegreen", markeredgecolor="black", lw=0,
                label="Pedestrian signal head"),
        Line2D([0], [0], marker="H", color="red", lw=0, label="Stop sign (unsignalized)"),
        Line2D([0], [0], marker="s", color="white", markeredgecolor="red", lw=0, label="No turn on red"),
        Line2D([0], [0], marker="*", color="dimgrey", lw=0, label="Streetlight"),
        Line2D([0], [0], marker="P", color="gold", markeredgecolor="black", lw=0,
                label="Pedestrian pushbutton (OSM)"),
        Patch(facecolor="darkorange", edgecolor="black", label="Tactile paving / curb ramp (OSM)"),
        Line2D([0], [0], color="black", lw=2.2, label="Traced kerb (OSM barrier=kerb)"),
        Line2D([0], [0], marker="D", color="gold", markeredgecolor="black", lw=0, label="RRFB beacon (OSM)"),
        Line2D([0], [0], marker="P", color="firebrick", lw=0, label="Fire hydrant (OSM)"),
        Line2D([0], [0], color="saddlebrown", lw=1.5, label="Corner parcel"),
        Line2D([0], [0], marker="o", color="blue", lw=0, label="Intersection"),
    ]
