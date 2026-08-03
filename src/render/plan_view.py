"""Plan-view rendering: draws an IntersectionModel + DesignState to a matplotlib axis."""
import geopandas as gpd
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import LineString, Polygon

from shapely.ops import substring

from src.geometry.model import (build_pavement_polygon, inset_point_at_station,
                                trimmed_curb_lines)
from src.geometry.paint import curbside_paint_ft
from src.geometry.intersection import IntersectionModel, kerb_lines_with_tags_ft
from src.geometry.treatments import DEFAULT_CENTERLINE_STYLE, DesignState
from src.provenance import PLOT_STYLE, built_width_provenance
from src.render.props import DRAWN_BY_PAINT, build_props, signalization_conflicts
from src.render.coords import FT_TO_M, wgs84_to_state_plane
from src.render.crosswalks import (CROSSWALK_DEPTH_M, STOP_BAR_PLAN_DEPTH_FT, centerline_start_ft,
                                   crosswalk_reaches_ft,
                                   crosswalk_band_ft, crosswalk_bands_ft, stop_bar_bands_ft,
                                   resolve_crosswalk_offsets,
                                   resolve_crosswalk_skews, resolve_stop_bar_offsets,
                                   stop_bar_band_geometry_ft, stop_bar_width_ft)
from src.sources.osm_context import (fetch_crossings, fetch_kerbs, fetch_sidewalks,
                                     fetch_stop_lines, fetch_street_furniture,
                                     fetch_traffic_control)

# Matches TACTILE_PAD_RED in scripts/blender/blender_props.py - the plan view and the 3D
# render must not disagree about what a detectable warning surface looks like.
TACTILE_PAD_COLOR = "#8c1f14"

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



# One colour for a flex-post wherever it is drawn from - the treatment layer's own bollard
# pieces, the daylight-zone props, and legend_handles(). Named so a test can count markers of
# this colour rather than trusting that the dispatch has a branch for them at all.
BOLLARD_PLAN_COLOR = "darkorange"


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

    Bollards tagged DRAWN_BY_PAINT are skipped, because the treatment layer above already
    drew them from state.bollard_lines. The ones that are NOT so tagged - the daylight-zone
    posts from protect_daylight_zone - exist only as props, and skipping every bollard the
    way this used to meant the plan view of a bollard proposal showed no bollards while the
    3D render of the same scenario showed thirteen.
    """
    kerb_lines = kerb_lines_with_tags_ft(model.center_wgs84, model.center_ft)
    props = build_props(model, state, crosswalk_offsets, model.center_ft, traffic_control,
                         street_furniture, crossings, fetch_kerbs(model.center_wgs84, radius_m=120))
    for line, _tags in kerb_lines:
        gpd.GeoSeries([line]).plot(ax=ax, color='black', linewidth=2.2, zorder=6)
    signal_count = 0
    for prop in props:
        kind = prop["type"]
        if kind == "bollard" and prop.get(DRAWN_BY_PAINT):
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
            gpd.GeoSeries([pad]).plot(ax=ax, color=TACTILE_PAD_COLOR, alpha=0.85, zorder=8)
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
        elif kind == "bollard":
            # Same marker the treatment layer uses for a bollard it drew itself, and the one
            # legend_handles() advertises. Without this branch a daylight-zone post fell
            # through to the generic "extras" case below and came out as a goldenrod
            # TRIANGLE - drawn, but not as the thing the legend says it is.
            ax.scatter([x], [y], color=BOLLARD_PLAN_COLOR, marker="o", s=14,
                        edgecolors="black", linewidths=0.4, zorder=7)
        else:  # site- or scenario-specific extras (school zone signs, etc.)
            ax.scatter([x], [y], color="darkgoldenrod", marker="^", s=30, zorder=7)

    if dimension_labels:
        control = (f"SIGNALIZED - {signal_count} signal pole(s)" if signal_count
                    else "NOT signalized - stop/yield control")
        ax.annotate(control, xy=(0.5, 0.005), xycoords="axes fraction", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="black" if signal_count else "dimgrey",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9))
    # Handed to the invariant pass rather than rebuilt there: build_props is the most
    # expensive thing in the plan view, and checking a DIFFERENT set of props from the one
    # drawn would defeat the point of checking at all.
    return props


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
                      crosswalk_skews: dict, dimension_labels: bool, pavement=None):
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
        band = crosswalk_band_ft(leg, offset_ft, depth_ft, crosswalk_skews.get(leg_name, 0.0),
                                  roadway=pavement)
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
        stop_lines = fetch_stop_lines(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
        for leg_name, stop_offset_ft in resolve_stop_bar_offsets(
                state, crosswalk_offsets, stop_lines).items():
            # A stop bar covers only the ENTERING half of the roadway - a driver stops
            # in their own lanes, never across the opposing ones. Sized and positioned
            # from the same shared rule blender_crosswalks.add_stop_bar uses, so the 2D
            # bar is the bar that gets rendered. It also inherits the crosswalk's
            # surveyed skew, being painted parallel to it.
            leg = state.legs[leg_name]
            span_ft, lateral_ft = stop_bar_band_geometry_ft(stop_bar_width_ft(state, leg_name))
            bar = crosswalk_band_ft(leg, stop_offset_ft, STOP_BAR_PLAN_DEPTH_FT,
                                    crosswalk_skews.get(leg_name, 0.0),
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
        pavement = None

    # Real OSM sidewalk centerlines, drawn behind everything else. These are what the
    # crossing ways actually connect to, and they bound where the curb can possibly be
    # (src/geometry/model.py:sidewalk_span_ft) - so having them on the plot is what makes
    # an over-wide leg visible instead of merely arguable.
    for walk in sidewalk_lines_ft(sidewalks):
        gpd.GeoSeries([walk]).plot(ax=ax, color="steelblue", linewidth=1.0, linestyle=(0, (4, 2)),
                                    alpha=0.65, zorder=2)

    # Curb lines as the corners trim them. The raw lines overshoot into the junction on
    # purpose (fillet material), so drawing them raw would draw curb across the middle of
    # the intersection - marking a curb that isn't there and isn't in the 3D render.
    curbs_by_leg = trimmed_curb_lines(state.legs, state.corner_fillets)
    for name, leg in state.legs.items():
        tier = built_width_provenance(leg, model.config["legs"][name])
        style_kw = PLOT_STYLE[tier]
        color = style_kw["color"]
        for curb in curbs_by_leg[name].values():
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
    _draw_crosswalks(ax, model, state, crosswalk_offsets, crosswalk_skews, dimension_labels,
                      pavement)
    props = _draw_props(ax, model, state, crosswalk_offsets, traffic_control, street_furniture,
                         crossings, dimension_labels)

    # Every painted marking comes from src/geometry/paint.py - the same builder the 3D
    # export draws from and src/checks.py inspects. This block used to assemble the paint
    # itself, in parallel with export.py doing the same, and the two had already drifted on
    # where a parking buffer's taper starts and on whether taper fill is cut around a
    # crossing. Both views now show the same geometry because it IS the same geometry.
    marked = set(model.config["intersection"].get("existing_marked_crosswalks", []))
    bands = crosswalk_bands_ft(state, crosswalk_offsets, crosswalk_skews,
                                CROSSWALK_DEPTH_M / FT_TO_M, pavement,
                                crosswalk_reaches_ft(state, crosswalk_offsets, crosswalk_skews,
                                                      pavement, marked))
    paint = curbside_paint_ft(state, crosswalk_offsets, model.center_ft, bands, props,
                               marked_crosswalks=marked)

    STYLE = {   # kind -> (matplotlib kwargs, whether it's a filled/hatched zone)
        "lane_narrowing_fill":  dict(color="gold", alpha=0.5, hatch="//", zorder=3),
        "taper_fill":           dict(color="gold", alpha=0.5, hatch="//", zorder=3),
        "buffer_fill":          dict(color="gold", alpha=0.5, hatch="//", zorder=3),
        # The statutory no-parking zone at the corner (R.S. 39:4-138). Drawn in a distinct
        # colour from the ordinary buffer hatch because it is a different claim: not "this
        # asphalt is spare", but "parking here is illegal and this proposal marks it".
        "daylight_fill":        dict(color="orangered", alpha=0.40, hatch="xx", zorder=3),
        "corner_hatch_fill":    dict(color="gold", alpha=0.5, hatch="//", zorder=3),
        "apron":                dict(color="peru", alpha=0.6, zorder=3),
        "lane_edge_line":       dict(color="goldenrod", linewidth=1.5, zorder=3),
        "taper_line":           dict(color="goldenrod", linewidth=1.5, zorder=3),
        "buffer_edge_line":     dict(color="goldenrod", linewidth=1.5, zorder=3),
        "daylight_edge_line":   dict(color="orangered", linewidth=1.5, zorder=3),
        "crossing_rim_line":    dict(color="orangered", linewidth=1.5, zorder=3),
        # The square end of a zone with no crossing to be cut by and no room to taper.
        "zone_end_line":        dict(color="goldenrod", linewidth=1.5, zorder=3),
        "parking_edge_line":    dict(color="steelblue", linewidth=1.5, zorder=3),
        "stall_divider":        dict(color="steelblue", linewidth=1, zorder=3),
    }
    EDGE = {"gold": "goldenrod", "peru": "saddlebrown", "orangered": "orangered"}

    for piece in paint:
        if piece.kind == "bollard":
            c = piece.geometry.centroid
            ax.scatter([c.x], [c.y], color=BOLLARD_PLAN_COLOR, marker="o", s=10, zorder=6)
            continue
        style = STYLE.get(piece.kind)
        if style is None:
            continue
        series = gpd.GeoSeries([piece.geometry])
        series.plot(ax=ax, **style)
        if piece.is_fill:
            series.boundary.plot(ax=ax, color=EDGE[style["color"]], linewidth=1, zorder=3)

    if dimension_labels:
        _label_paint(ax, state, paint, crosswalk_offsets)
        _label_parking_legality(ax, model, state)

    ax.scatter([model.center_ft.x], [model.center_ft.y], color="blue", zorder=6, s=40)

    _draw_centerlines(ax, state, crosswalk_offsets, stop_bar_offsets_for(model, state, crossings))

    violations = _mark_violations(ax, model, state, crossings, props, paint, pavement)

    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    zoom_ft = 110
    ax.set_xlim(model.center_ft.x - zoom_ft, model.center_ft.x + zoom_ft)
    ax.set_ylim(model.center_ft.y - zoom_ft, model.center_ft.y + zoom_ft)
    ax.set_xlabel("Feet (EPSG:3424)")
    return violations


# MUTCD/AASHTO proportions, matching scripts/blender/blender_crosswalks.py's
# add_double_yellow_centerline: ~6 in stripes with a ~4 in gap between them.
DOUBLE_YELLOW_GAP_FT = 0.1 / FT_TO_M


def _draw_centerlines(ax, state, crosswalk_offsets, stop_bar_offsets):
    """The leg centerline (the measurement datum) and the painted centerline on top of it.

    Both were missing. The datum matters because every width in this drawing - the 11 ft
    lane, the 8 ft stall, the depth of a hatched zone - is an OFFSET FROM IT, and without it
    drawn there is nothing to check those offsets against by eye. The painted centerline
    matters because the 3D render draws one (blender_scene.py, from state.centerline_styles)
    and this view is supposed to show what that render will show.

    The paint starts where src/render/crosswalks.py:centerline_start_ft says, which is at the
    stop bar - the same rule the export uses, not a second copy of it.
    """
    for leg_name, leg in state.legs.items():
        # The datum: thin, grey, dotted, the full length of the leg. Deliberately
        # unobtrusive - it is a construction line, not a marking on the road.
        ax.plot(*leg.centerline.xy, color="#3b6ea5", lw=0.9, ls=(0, (7, 3, 1, 3)), alpha=0.9,
                zorder=4)

        style = state.centerline_styles.get(leg_name, DEFAULT_CENTERLINE_STYLE)
        if style == "none" or leg_name not in crosswalk_offsets:
            continue
        start_ft = centerline_start_ft(crosswalk_offsets[leg_name][0],
                                        stop_bar_offsets.get(leg_name))
        if start_ft >= leg.centerline.length:
            continue
        painted = substring(leg.centerline, start_ft, leg.centerline.length)
        if style == "double_yellow":
            for sign in (1, -1):
                ax.plot(*painted.offset_curve(sign * DOUBLE_YELLOW_GAP_FT / 2).xy,
                        color="gold", lw=1.2, zorder=4)
        else:   # single_yellow_dashed
            ax.plot(*painted.xy, color="gold", lw=1.2, ls=(0, (6, 6)), zorder=4)


# What OSM says about kerbside parking, and what that produced. Colour is the OSM statement
# alone, so a kerb the surveyor tagged and a kerb nobody has tagged never look the same.
PARKING_LEGALITY_COLOR = {"restricted": "#b3261e", "allowed": "#1b7f3b", "untagged": "#6b6b6b"}



def _label_parking_legality(ax, model, state):
    """Per side of every leg: the OSM parking tag, and what the design did with it.

    Without this the drawing cannot answer the question it most often provokes - "why is that
    kerb hatched?" - because three different situations produce identical hatching: OSM says
    no parking, OSM says parking is fine but the road has less than one stall's width spare,
    and nobody has tagged it at all. The first is a restriction being marked; the other two
    are this design's own arithmetic. Only the tag distinguishes them, so the tag is drawn.
    """
    from src.geometry.intersection import parking_restriction_by_side, parking_is_restricted
    from src.geometry.model import curb_point_at_station
    from src.geometry.treatments import MIN_MARKED_PARKING_DEPTH_FT, TARGET_LANE_WIDTH_FT

    osm_tags = getattr(model, "leg_osm_tags", {})
    aligned_by_leg = getattr(model, "leg_osm_aligned", {})
    for leg_name, leg in state.legs.items():
        if leg.curb_to_curb_ft is None:
            continue
        sides = parking_restriction_by_side(osm_tags.get(leg_name, {}),
                                             aligned_by_leg.get(leg_name, True))
        allowance_ft = leg.curb_to_curb_ft / 2 - TARGET_LANE_WIDTH_FT
        for side in ("left", "right"):
            restriction = sides[side]
            if parking_is_restricted(restriction):
                kind, says = "restricted", restriction
            elif restriction == "none":
                kind, says = "allowed", "none (parking OK)"
            else:
                kind, says = "untagged", "untagged"

            if (leg_name, side) in state.parking_zones:
                drew = "stalls"
            elif (leg_name in state.lane_narrowing
                  and side in state.lane_narrowing_sides.get(leg_name, ("left", "right"))):
                drew = ("hatched" if kind == "restricted"
                        else f"hatched: only {allowance_ft:.1f} ft spare, "
                             f"under a {MIN_MARKED_PARKING_DEPTH_FT:.0f} ft stall")
            else:
                drew = (f"nothing: {allowance_ft:.1f} ft spare beside an "
                        f"{TARGET_LANE_WIDTH_FT:.0f} ft lane")

            point = curb_point_at_station(leg, side, leg.centerline.length * 0.42)
            if point is None:
                continue
            outward = 1 if side == "left" else -1
            here = inset_point_at_station(leg, leg.centerline.length * 0.42,
                                           outward * (abs(_offset_of(leg, point)) + 9.0))
            ax.annotate(f"OSM parking: {says}\n-> {drew}", (here[0], here[1]),
                        fontsize=5.2, color=PARKING_LEGALITY_COLOR[kind], ha="center",
                        va="center", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                  ec=PARKING_LEGALITY_COLOR[kind], lw=0.6, alpha=0.9))


def _offset_of(leg, point):
    import numpy as np

    from src.geometry.model import station_offset_many

    _stations, offsets = station_offset_many(leg.centerline, np.asarray([point], dtype=float))
    return float(offsets[0])


def _label_paint(ax, state, paint, crosswalk_offsets):
    """Dimension labels for the curbside paint: what each treatment actually measures.

    One lane label PER SIDE narrowed, offset into that lane - not a single label on the
    centerline, which reads as "this road is one 11 ft lane" rather than what is there
    (a leg is not always narrowed on both sides).
    """
    for leg_name, stripe_width_ft in state.lane_narrowing.items():
        leg = state.legs[leg_name]
        lane_ft = leg.curb_to_curb_ft / 2 - stripe_width_ft
        along_ft = min(leg.centerline.length * 0.6, leg.centerline.length - 5)
        for side, sign in (("left", 1), ("right", -1)):
            if side not in state.lane_narrowing_sides.get(leg_name, ("left", "right")):
                continue
            at = leg.centerline.offset_curve(sign * lane_ft / 2).interpolate(along_ft)
            ax.annotate(f"lane {lane_ft:.1f} ft", (at.x, at.y), fontsize=6.5, color="goldenrod",
                        ha="center", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    # One label per RUN of stalls, not per side: a hydrant mid-block splits a kerb into two
    # separate runs, and a single label would be counting stalls that are not in one place.
    for piece in paint:
        if piece.kind != "parking_edge_line":
            continue
        zone = state.parking_zones.get((piece.leg, piece.side))
        if zone is None:
            continue
        n_stalls = int(piece.geometry.length // zone["stall_length_ft"])
        mid = piece.geometry.interpolate(0.5, normalized=True)
        ax.annotate(f"parking\n{n_stalls} stalls ({zone['depth_ft']:.0f} ft)", (mid.x, mid.y),
                    fontsize=6, color="steelblue", ha="center", va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))


def stop_bar_offsets_for(model, state, crossings):
    """{leg: offset} for a signalized junction, else {} - the same resolution the export and
    the invariant check use, so all three agree on where the bar is."""
    if not model.config.get("signals"):
        return {}
    return resolve_stop_bar_offsets(
        state, resolve_crosswalk_offsets(state, crossings),
        fetch_stop_lines(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M))


def _mark_violations(ax, model, state, crossings, props, paint, pavement=None):
    """Run the scene invariants and draw whatever failed, right where it failed.

    The plan view reports rather than raises, and the phase script asserts after saving -
    so a failure always arrives with a picture of itself. Reading "tactile pad 40% in the
    roadway at (419160, 566742)" next to a red ring around that exact pad is the difference
    between one round trip and several.
    """
    from src.checks import check_scene

    offsets = resolve_crosswalk_offsets(state, crossings)
    skews = resolve_crosswalk_skews(state, crossings)
    stop_offsets = (resolve_stop_bar_offsets(
        state, offsets, fetch_stop_lines(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M))
        if model.config.get("signals") else {})

    violations = check_scene(
        model, state, props, pavement,
        crosswalk_bands=crosswalk_bands_ft(state, offsets, skews, CROSSWALK_DEPTH_M / FT_TO_M,
                                            pavement),
        stop_bars=stop_bar_bands_ft(state, stop_offsets, skews),
        paint=paint, crosswalk_offsets=offsets)

    located = [v for v in violations if v.where]
    if located:
        xs, ys = zip(*(v.where for v in located))
        ax.scatter(xs, ys, s=260, facecolors="none", edgecolors="red", linewidths=2.0, zorder=10)
        ax.annotate(f"{len(violations)} INVARIANT FAILURE(S) - see console",
                    xy=(0.5, 0.975), xycoords="axes fraction", ha="center", va="top",
                    fontsize=9, fontweight="bold", color="red",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.95))
    for violation in violations:
        label = "INVARIANT FAILED" if violation.fatal else "SOURCE CONFLICT"
        print(f"  {label}: {violation}")
    return violations


def legend_handles():
    return [
        Line2D([0], [0], color="black", lw=2, label="Curb line - FIELD-MEASURED width"),
        Line2D([0], [0], color="darkviolet", lw=2, ls="-.", label="Curb line - OSM-derived width"),
        Line2D([0], [0], color="crimson", lw=2, ls="--", label="Curb line - estimated width"),
        Line2D([0], [0], color="steelblue", lw=1, ls=(0,(4,2)), label="OSM sidewalk centerline"),
        Line2D([0], [0], color="#3b6ea5", lw=0.9, ls=(0,(7,3,1,3)), label="Leg centerline (widths measured from this)"),
        Line2D([0], [0], color="gold", lw=1.2, label="Centerline paint (double yellow / dashed)"),
        Patch(facecolor="white", edgecolor=PARKING_LEGALITY_COLOR["restricted"],
               label="OSM: parking restricted"),
        Patch(facecolor="white", edgecolor=PARKING_LEGALITY_COLOR["allowed"],
               label="OSM: parking allowed"),
        Patch(facecolor="white", edgecolor=PARKING_LEGALITY_COLOR["untagged"],
               label="OSM: parking untagged"),
        Line2D([0], [0], color="darkviolet", lw=1, ls=":", label="OSM crossing way (as surveyed)"),
        Line2D([0], [0], color="darkorange", lw=2.5, label="Corner fillet (radius labeled)"),
        Line2D([0], [0], color="seagreen", lw=6, alpha=0.6, label="Pedestrian refuge island"),
        Line2D([0], [0], color="slateblue", lw=6, alpha=0.35, label="Raised crossing"),
        Patch(facecolor="gold", alpha=0.5, hatch="//", edgecolor="goldenrod", label="Lane narrowing / corner hatching"),
        Line2D([0], [0], color="goldenrod", lw=1.5, label="Lane narrowing - line only (no chevron fill)"),
        Patch(facecolor="orangered", alpha=0.40, hatch="xx", edgecolor="orangered",
               label="Daylighting - no parking (R.S. 39:4-138)"),
        Patch(facecolor="peru", alpha=0.6, edgecolor="saddlebrown", label="Mountable apron"),
        Line2D([0], [0], marker="o", color=BOLLARD_PLAN_COLOR, lw=0, label="Bollard"),
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
        Patch(facecolor=TACTILE_PAD_COLOR, edgecolor="black", label="Tactile paving / curb ramp (OSM)"),
        Line2D([0], [0], color="black", lw=2.2, label="Traced kerb (OSM barrier=kerb)"),
        Line2D([0], [0], marker="D", color="gold", markeredgecolor="black", lw=0, label="RRFB beacon (OSM)"),
        Line2D([0], [0], marker="P", color="firebrick", lw=0, label="Fire hydrant (OSM)"),
        Line2D([0], [0], color="saddlebrown", lw=1.5, label="Corner parcel"),
        Line2D([0], [0], marker="o", color="blue", lw=0, label="Intersection"),
    ]
