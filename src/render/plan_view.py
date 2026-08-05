"""Plan-view rendering: draws an IntersectionModel + DesignState to a matplotlib axis."""
import geopandas as gpd
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import LineString

from shapely.ops import substring

from src.geometry.model import inset_point_at_station, trimmed_curb_lines
from src.geometry.intersection import IntersectionModel, kerb_lines_with_tags_ft
from src.geometry.kerbs import KerbType
from src.geometry.treatments import DesignState, RaiseCrossing, RefugeIsland
from src.provenance import PLOT_STYLE, built_width_provenance
from src.geometry import markings
from src.geometry.markings import require_every_kind
from src.render.props import (DRAWN_BY_PAINT, TACTILE_PAD_DEPTH_FT, TACTILE_PAD_WIDTH_FT,
                               build_props, pad_polygon, signalization_conflicts)
from src.render.coords import FT_TO_M, wgs84_to_state_plane
from src.render.crosswalks import centerline_start_ft
from src.render.frame import junction_frame
from src.render.scene import SceneGeometry
from src.sources.osm_context import (fetch_crossings, fetch_kerbs, fetch_sidewalks,
                                     fetch_street_furniture, fetch_traffic_control)

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


def _draw(ax, geometries, boundary=None, **style) -> None:
    """Draw a group of same-styled geometries as ONE matplotlib collection.

    Every draw in this module used to be `gpd.GeoSeries([one_geometry]).plot(...)`, which
    makes a collection per geometry, and the cost of adding a collection grows with how many
    are already on the axes. A plan view of a proposal builds ~170 of them, and measured on
    this repo's own figures that is 2.44 s against 0.016 s for the same geometry drawn as one
    collection per style - 156x, and it was the single largest cost in a 2D build.

    `boundary` is the kwargs for outlining filled shapes, drawn from the same series, so a
    fill and its outline stay one pair rather than two independent draws.
    """
    geometries = [g for g in geometries if g is not None and not g.is_empty]
    if not geometries:
        return
    series = gpd.GeoSeries(geometries)
    series.plot(ax=ax, **style)
    if boundary is not None:
        series.boundary.plot(ax=ax, **boundary)


def _scatter_groups(ax, points_by_style: dict) -> None:
    """One ax.scatter per marker style rather than one per prop, for the same reason."""
    for style, points in points_by_style.items():
        if not points:
            continue
        xs, ys = zip(*points)
        ax.scatter(xs, ys, **dict(style))


# One colour for a flex-post wherever it is drawn from - the treatment layer's own bollard
# pieces, the daylight-zone props, and legend_handles(). Named so a test can count markers of
# this colour rather than trusting that the dispatch has a branch for them at all.
BOLLARD_PLAN_COLOR = "darkorange"

# How each prop type is marked in plan. Data rather than an if/elif chain, so every marker
# style is one dict entry and every prop of a type can be scattered in one call - and so a
# new prop type is a row here instead of another branch that might be forgotten (a
# daylight-zone bollard once fell through to the generic case and came out as a goldenrod
# triangle, drawn but not as the thing the legend said it was).
PROP_MARKERS = {
    "traffic_signal_pole":    (dict(color="black", marker="o", s=46, zorder=7),
                               dict(color="limegreen", marker="o", s=16, zorder=7)),
    "pedestrian_signal_head": (dict(color="limegreen", marker="s", s=22, edgecolors="black",
                                    linewidths=0.6, zorder=7),),
    "stop_sign":              (dict(color="red", marker="H", s=44, edgecolors="white",
                                    linewidths=0.6, zorder=7),),
    "pedestrian_pushbutton":  (dict(color="gold", marker="P", s=30, edgecolors="black",
                                    linewidths=0.5, zorder=8),),
    "rrfb":                   (dict(color="gold", marker="D", s=30, edgecolors="black",
                                    linewidths=0.6, zorder=8),),
    "fire_hydrant":           (dict(color="firebrick", marker="P", s=34, zorder=7),),
    "yield_sign":             (dict(color="white", marker="v", s=40, edgecolors="red",
                                    linewidths=1.2, zorder=7),),
    "no_turn_on_red_sign":    (dict(color="white", marker="s", s=26, edgecolors="red",
                                    linewidths=1.2, zorder=7),),
    "streetlight":            (dict(color="dimgrey", marker="*", s=34, zorder=6),),
    "bollard":                (dict(color=BOLLARD_PLAN_COLOR, marker="o", s=14,
                                    edgecolors="black", linewidths=0.4, zorder=7),),
}
# How each kind of traced kerb is drawn. Raised is the solid black line this view has always
# drawn; a LOWERED kerb is where a vehicle crosses - a driveway or a yard entrance - and the
# whole point of distinguishing them is that the kerbside markings break over one and not the
# other, so the drawing has to show which is which or the gap in the paint looks like a mistake.
# Every one of the 95 kerbs mapped here is tagged, so UNKNOWN is drawn only if that stops being
# true - and drawn distinctly rather than as raised, because "nobody said" is not "raised".
# Named linestyles, not dash tuples: these go through GeoSeries.plot to a LineCollection, and a
# (offset, (on, off)) tuple there is read as per-element data - "inhomogeneous shape" from numpy
# rather than a dashed line.
KERB_STYLE = {
    KerbType.RAISED:  dict(color="black", linewidth=2.2, zorder=6),
    KerbType.LOWERED: dict(color="black", linewidth=1.1, linestyle="--", zorder=6),
    KerbType.FLUSH:   dict(color="black", linewidth=1.1, linestyle=":", zorder=6),
    KerbType.UNKNOWN: dict(color="dimgrey", linewidth=1.6, linestyle="-.", zorder=6),
}


# A driveway is PAVING, so it is drawn as paving: a filled strip under everything else, in a
# browner grey than the roadway so it reads as private access rather than carriageway. It was a
# thin dashed centreline, which on a drawing already carrying parcel lines, sidewalk centrelines
# and leg centrelines was indistinguishable from them - the thing it exists to explain (the gap in
# the markings at its mouth) needs to be visibly a surface a car drives on.
DRIVEWAY_STYLE = dict(color="#8a7a68", alpha=0.55, zorder=2)
DRIVEWAY_EDGE = dict(color="#5d5044", linewidth=0.8, zorder=2)


def _draw_driveways(ax, driveways) -> None:
    """The junction's mapped driveways, off the MODEL - already projected and already widened into
    the strip both views draw, so the plan view and the 3D render cannot disagree about where a
    driveway is or how wide it is. See src/geometry/intersection.py:Driveway."""
    _draw(ax, [drive.surface for drive in driveways or () if drive.surface is not None],
          boundary=DRIVEWAY_EDGE, **DRIVEWAY_STYLE)


def _draw_kerbs(ax, kerb_lines) -> None:
    """The traced kerbs, grouped by what OSM says each one is.

    One collection per kerb type rather than per line, the same reason _draw groups everything
    else. Grouped by TYPE and not drawn uniformly because a dropped kerb is why a marking stops:
    src/geometry/paint.py:kerb_opening_bands breaks the kerbside paint over exactly these, and a
    reader looking at a gap in a bike lane needs to see the driveway that caused it.
    """
    by_type: dict = {}
    for line, tags, _way_id in kerb_lines:
        by_type.setdefault(KerbType.from_tags(tags), []).append(line)
    for kerb, lines in sorted(by_type.items(), key=lambda kv: str(kv[0])):
        _draw(ax, lines, **KERB_STYLE[kerb])


# Site- or scenario-specific extras (school zone signs, RRFB relocations, ...) have no
# dedicated marker: they are whatever a config or a proposal named.
EXTRA_PROP_MARKER = dict(color="darkgoldenrod", marker="^", s=30, zorder=7)

# How each marking is drawn in plan. Styling is a real per-marking choice - what colour says
# "this asphalt is spare" versus "parking here is illegal" is a judgement, not something
# derivable - so this table is written by hand. require_every_kind is what makes forgetting an
# entry impossible: a marking declared in src/geometry/markings.py with no style here raises on
# import, rather than being silently absent from the plan view while the 3D render draws it. An
# OBJECT is exempt - a flex post is drawn as a marker, not as paint (see BOLLARD_PLAN_COLOR).
PAINT_STYLE = require_every_kind({
    markings.LANE_NARROWING_FILL: dict(color="gold", alpha=0.5, hatch="//", zorder=3),
    markings.TAPER_FILL:          dict(color="gold", alpha=0.5, hatch="//", zorder=3),
    markings.BUFFER_FILL:         dict(color="gold", alpha=0.5, hatch="//", zorder=3),
    # The statutory no-parking zone at the corner (R.S. 39:4-138). Drawn in a distinct
    # colour from the ordinary buffer hatch because it is a different claim: not "this
    # asphalt is spare", but "parking here is illegal and this proposal marks it".
    markings.DAYLIGHT_FILL:       dict(color="orangered", alpha=0.40, hatch="xx", zorder=3),
    markings.CORNER_HATCH_FILL:   dict(color="gold", alpha=0.5, hatch="//", zorder=3),
    markings.APRON:               dict(color="peru", alpha=0.6, zorder=3),
    # A green bike lane's asphalt. Under the stripes' zorder so the white edge lines read on
    # top of it, exactly as they do on the street and in the render.
    markings.BIKE_LANE_SURFACE:   dict(color="mediumseagreen", alpha=0.45, zorder=2),
    markings.LANE_EDGE_LINE:      dict(color="goldenrod", linewidth=1.5, zorder=3),
    markings.TAPER_LINE:          dict(color="goldenrod", linewidth=1.5, zorder=3),
    markings.BUFFER_EDGE_LINE:    dict(color="goldenrod", linewidth=1.5, zorder=3),
    markings.DAYLIGHT_EDGE_LINE:  dict(color="orangered", linewidth=1.5, zorder=3),
    # The square end of a zone with no crossing to be cut by and no room to taper.
    markings.ZONE_END_LINE:       dict(color="goldenrod", linewidth=1.5, zorder=3),
    markings.PARKING_EDGE_LINE:   dict(color="steelblue", linewidth=1.5, zorder=3),
    markings.STALL_DIVIDER:       dict(color="steelblue", linewidth=1, zorder=3),
    # An exclusive bike lane. Green, because that is what a bike lane is coloured on a real
    # street and in every other agency's drawings - and a colour of its own is the point: this
    # is the one treatment here that says a vehicle BELONGS in the strip, where the gold
    # hatching says nothing does.
    markings.BIKE_LANE_EDGE_LINE: dict(color="seagreen", linewidth=1.6, zorder=3),
    # Drawn identically to the continuous line, because it IS that line: the breaks are in the
    # geometry rather than in a dash pattern, so what the plan view draws here is a row of short
    # stripes - the same row the render extrudes. Styling it differently would say the paint is a
    # different colour across a driveway, which it is not.
    markings.BIKE_LANE_DOTTED_EXTENSION: dict(color="seagreen", linewidth=1.6, zorder=3),
    markings.BIKE_BUFFER_FILL:    dict(color="mediumseagreen", alpha=0.35, hatch="\\\\", zorder=3),
}, "plan_view.PAINT_STYLE")
# Outline colour for each filled zone's own fill colour.
PAINT_FILL_EDGE = {"gold": "goldenrod", "peru": "saddlebrown", "orangered": "orangered",
                   "mediumseagreen": "seagreen"}


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
    drew them - LaneNarrowingBollards and ParkingBufferBollards each emit their own paint
    pieces. The ones that are NOT so tagged - the daylight-zone posts from
    ProtectDaylightZone - exist only as props, and skipping every bollard the
    way this used to meant the plan view of a bollard proposal showed no bollards while the
    3D render of the same scenario showed thirteen.
    """
    kerb_lines = kerb_lines_with_tags_ft(model.center_wgs84, model.center_ft)
    props = build_props(model, state, crosswalk_offsets, model.center_ft, traffic_control,
                         street_furniture, crossings, fetch_kerbs(model.center_wgs84, radius_m=120))
    _draw_driveways(ax, model.driveways)
    _draw_kerbs(ax, kerb_lines)

    # Grouped, then drawn once per group. See _draw / _scatter_groups.
    marker_points: dict[tuple, list] = {}
    pads, arms = [], []
    signal_count = 0
    for prop in props:
        kind = prop["type"]
        if kind == "bollard" and prop.get(DRAWN_BY_PAINT):
            continue
        x, y = prop["position_ft"]
        if kind == "tactile_paving_pad":
            # Drawn at its true size and orientation, not as a marker: whether the pad
            # sits wholly on the footway or spills into the roadway is exactly the kind
            # of thing the plan view exists to make checkable.
            pads.append(pad_polygon(x, y, prop["heading_deg"],
                                      depth_ft=prop.get("pad_depth_ft", TACTILE_PAD_DEPTH_FT),
                                      width_ft=prop.get("pad_width_ft", TACTILE_PAD_WIDTH_FT)))
            continue
        if kind == "traffic_signal_pole":
            signal_count += 1
            # The mast arm is the part that reaches out over the roadway, and its length
            # is derived from a real leg width - worth seeing in plan, since it's the
            # most visually dominant thing in the 3D render.
            arm_deg, arm_ft = prop.get("arm_heading_deg"), prop.get("arm_length_ft")
            if arm_deg is not None and arm_ft:
                arms.append(LineString([(x, y),
                                        (x + np.cos(np.radians(arm_deg)) * arm_ft,
                                         y + np.sin(np.radians(arm_deg)) * arm_ft)]))
        for style in PROP_MARKERS.get(kind, (EXTRA_PROP_MARKER,)):
            marker_points.setdefault(tuple(sorted(style.items())), []).append((x, y))

    _draw(ax, arms, color="black", linewidth=1.6, capstyle="round", zorder=7)
    _draw(ax, pads, color=TACTILE_PAD_COLOR, alpha=0.85, zorder=8,
          boundary=dict(color="black", linewidth=0.5, zorder=8))
    _scatter_groups(ax, marker_points)

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
    lines = sidewalk_lines_ft(crossings)
    _draw(ax, lines, color="darkviolet", linewidth=1.0, linestyle=":", alpha=0.8, zorder=5)
    ends = [line.coords[i] for line in lines for i in (0, -1)]
    if ends:
        ax.scatter(*zip(*ends), color="darkviolet", s=8, marker="o", zorder=5)


def _draw_crosswalks(ax, scene: SceneGeometry, dimension_labels: bool):
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

    Both footprints come from `scene`, which is the point. This used to rebuild them: the
    crossing band without the two-pass reaches the paint and the invariants use (15 sq ft
    out at W Broad & Louellen), and the stop bar without the skew stretch
    stop_bar_bands_ft applies (3.8 ft out on Louellen's -44 deg crossing). Both were drawn
    in a place neither the render nor the checks agreed with, which is the one failure this
    view exists to catch rather than commit.
    """
    state = scene.state
    raised = {t.target.leg for t in state.treatments_of(RaiseCrossing)}
    # Grouped by how each band gets drawn, which turns on two independent facts: whether it is
    # painted at all, and whether its position is surveyed or estimated.
    painted_surveyed, painted_estimated, unpainted = [], [], []

    for leg_name in state.legs:
        offset = scene.crosswalk_offsets[leg_name]
        painted = leg_name in scene.marked_crosswalks and leg_name not in raised
        band = scene.crosswalk_bands[leg_name]
        if not painted:
            unpainted.append(band)
        elif offset.is_surveyed:
            painted_surveyed.append(band)
        else:
            painted_estimated.append(band)

        if dimension_labels:
            centroid = band.centroid
            note = "" if painted else "\n(unmarked)"
            edge = "darkviolet" if offset.is_surveyed else "crimson"
            ax.annotate(f"{offset.offset_ft:.0f} ft\n{'OSM' if offset.is_surveyed else 'est.'}{note}",
                        (centroid.x, centroid.y), fontsize=5.5,
                        color=edge if painted else "grey", ha="center", va="center", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7))

    for bands, edge, dash in ((painted_surveyed, "darkviolet", "-"),
                              (painted_estimated, "crimson", "--")):
        _draw(ax, bands, color="white", alpha=0.95, zorder=4,
              boundary=dict(color=edge, linewidth=1.4, linestyle=dash, zorder=4))
    # Unpainted legs get the outline only - "this leg has no marked crossing" is itself a
    # finding, and a wrong offset on one is a latent error until a proposal marks it.
    if unpainted:
        gpd.GeoSeries(unpainted).boundary.plot(ax=ax, color="grey", linewidth=0.7,
                                                linestyle=":", zorder=4)

    # Stop bars, on the same terms the export uses: signalized sites only, which is what
    # leaves scene.stop_bar_bands empty everywhere else. A bar covers only the ENTERING half
    # of the roadway - a driver stops in their own lanes, never across the opposing ones -
    # and inherits the crossing's surveyed skew, being painted parallel to it.
    _draw(ax, scene.stop_bar_bands.values(), color="dimgrey", alpha=0.9, zorder=4)


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

    # Always resolved, not just when a treatment needs them: the crosswalks ARE the subject
    # of this project, so a plan view that omits them isn't a reconstruction you can check
    # the 3D render against. Leaving them out is what let a mis-matched OSM crossing (see
    # src/render/crosswalks.py:_match_crossings_to_legs) sit at the dead centre of E Broad &
    # Princeton unnoticed - it was only visible in the 3D render. fetch_crossings is
    # disk-cached per (center, radius), so the network round-trip happens once per site.
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
    # Once, for the whole figure: the pavement, every crossing and stop bar footprint, and
    # the offsets/skews everything else is measured from. This function used to resolve them
    # three times over and build the crossing bands twice from different arguments - see
    # src/render/scene.py for what that cost.
    scene = SceneGeometry.resolve(model, state, crossings)
    pavement = scene.pavement

    model.parcels.boundary.plot(ax=ax, color="tan", linewidth=0.6, zorder=1)
    model.corner_parcels.boundary.plot(ax=ax, color="saddlebrown", linewidth=1.5, zorder=1)

    _draw(ax, [pavement], color="#d9d9d9", zorder=2)

    # Real OSM sidewalk centerlines, drawn behind everything else. These are what the
    # crossing ways actually connect to, and they bound where the curb can possibly be
    # (src/geometry/model.py:sidewalk_span_ft) - so having them on the plot is what makes
    # an over-wide leg visible instead of merely arguable.
    _draw(ax, sidewalk_lines_ft(sidewalks), color="steelblue", linewidth=1.0,
          linestyle=(0, (4, 2)), alpha=0.65, zorder=2)

    # Curb lines as the corners trim them. The raw lines overshoot into the junction on
    # purpose (fillet material), so drawing them raw would draw curb across the middle of
    # the intersection - marking a curb that isn't there and isn't in the 3D render.
    # Grouped by provenance tier, since that is what decides the style.
    curbs_by_leg = trimmed_curb_lines(state.legs, state.corner_fillets)
    curbs_by_tier: dict[str, list] = {}
    for name, leg in state.legs.items():
        tier = built_width_provenance(leg, model.config["legs"][name])
        curbs_by_tier.setdefault(tier, []).extend(curbs_by_leg[name].values())
        if dimension_labels:
            mid = leg.centerline.interpolate(min(leg.centerline.length * 0.85, leg.centerline.length - 5))
            ax.annotate(f"{leg.curb_to_curb_ft:.1f} ft", (mid.x, mid.y), fontsize=7,
                        color=PLOT_STYLE[tier]["color"], ha="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    for tier, curbs in curbs_by_tier.items():
        _draw(ax, curbs, linewidth=2, zorder=3, **PLOT_STYLE[tier])

    arcs = []
    for corner, pieces in state.corner_fillets.items():
        if "error" in pieces:
            continue
        arcs.append(pieces["arc"])
        # `is not None` as well as `in`: a corner that is not a corner - two legs of one street
        # running through the junction - has no radius, and reaching for one crashed the build.
        if dimension_labels and pieces.get("radius_ft") is not None:
            mid = pieces["arc"].interpolate(0.5, normalized=True)
            ax.annotate(f"R={pieces['radius_ft']:.0f} ft", (mid.x, mid.y), fontsize=7, color="darkorange",
                        fontweight="bold", ha="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    _draw(ax, arcs, color="darkorange", linewidth=2.5, zorder=4)

    # Asked of the treatments, which build their polygon against this design - see
    # src/render/export.py's note on why these two are not materialised onto the state.
    islands = [(island, island.polygon(state)) for island in state.treatments_of(RefugeIsland)]
    _draw(ax, [polygon for _island, polygon in islands],
          color="seagreen", alpha=0.6, zorder=5,
          boundary=dict(color="darkgreen", linewidth=1, zorder=5))
    if dimension_labels:
        for island, polygon in islands:
            c = polygon.centroid
            ax.annotate(f"refuge\n{island.width_ft:.0f} ft", (c.x, c.y), fontsize=6.5, color="darkgreen",
                        ha="center", va="center", fontweight="bold")

    raised_bands = [t.polygon(state) for t in state.treatments_of(RaiseCrossing)]
    _draw(ax, raised_bands, color="slateblue", alpha=0.35, hatch="//", zorder=2,
          boundary=dict(color="slateblue", linewidth=1, zorder=2))
    if dimension_labels:
        for poly in raised_bands:
            c = poly.centroid
            ax.annotate("raised\ncrossing", (c.x, c.y), fontsize=6.5, color="indigo",
                        ha="center", va="center", fontweight="bold")

    _draw_surveyed_crossings(ax, crossings)
    _draw_crosswalks(ax, scene, dimension_labels)
    props = _draw_props(ax, model, state, scene.crosswalk_offsets, traffic_control,
                         street_furniture, crossings, dimension_labels)

    # Every painted marking comes from src/geometry/paint.py - the same builder the 3D
    # export draws from and src/checks.py inspects. This block used to assemble the paint
    # itself, in parallel with export.py doing the same, and the two had already drifted on
    # where a parking buffer's taper starts and on whether taper fill is cut around a
    # crossing. Both views now show the same geometry because it IS the same geometry.
    #
    # props comes back extended with the bollards the paint places (see
    # SceneGeometry.build_paint_and_posts). Nothing more is drawn for them here - the loop
    # below already draws them from the paint - but the invariant pass has to see the same
    # props the export will, or the check that they reach the 3D render can only fail there.
    paint, props = scene.build_paint_and_posts(props)

    # All pieces of one kind in one collection. A proposal builds well over a hundred, and
    # adding a matplotlib collection costs more the more are already there - drawn one at a
    # time this was the single most expensive thing in a 2D build. See _draw.
    by_kind: dict[markings.PaintKind, list] = {}
    bollards = []
    for piece in paint:
        # A flex post is an object, not paint, and is drawn as a marker below. Asked of the
        # marking rather than matched against its name - see markings.Role.
        if piece.kind.is_object:
            bollards.append(piece.geometry.centroid)
        else:
            by_kind.setdefault(piece.kind, []).append(piece.geometry)
    for kind, geometries in by_kind.items():
        style = PAINT_STYLE[kind]
        # A zone that covers ground gets its outline drawn too; a line has no boundary. Asked
        # of the marking, not of the geometry: a bollard is stored as a degenerate polygon, so
        # the geometry test answered "fill" for something that is neither.
        edge = (dict(color=PAINT_FILL_EDGE[style["color"]], linewidth=1, zorder=3)
                if kind.covers_area else None)
        _draw(ax, geometries, boundary=edge, **style)
    if bollards:
        ax.scatter([p.x for p in bollards], [p.y for p in bollards],
                   color=BOLLARD_PLAN_COLOR, marker="o", s=10, zorder=6)

    if dimension_labels:
        _label_paint(ax, state, paint)
        _label_parking_legality(ax, model, state)

    ax.scatter([model.center_ft.x], [model.center_ft.y], color="blue", zorder=6, s=40)

    _draw_centerlines(ax, scene)

    violations = _mark_violations(ax, scene, props, paint)

    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    # The frame the 3D render is pointed at as well - see src/render/frame.py for the 1.15-1.57x
    # the two views used to disagree by, and why it is measured from the model rather than from
    # this DesignState (a before/after pair has to share one frame).
    xmin, xmax, ymin, ymax = junction_frame(model).bounds_ft()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Feet (EPSG:3424)")
    return violations


# MUTCD/AASHTO proportions, matching scripts/blender/blender_crosswalks.py's
# add_double_yellow_centerline: ~6 in stripes with a ~4 in gap between them.
DOUBLE_YELLOW_GAP_FT = 0.1 / FT_TO_M


def _draw_centerlines(ax, scene: SceneGeometry):
    """The leg centerline (the measurement datum) and the painted centerline on top of it.

    Both were missing. The datum matters because every width in this drawing - the 11 ft
    lane, the 8 ft stall, the depth of a hatched zone - is an OFFSET FROM IT, and without it
    drawn there is nothing to check those offsets against by eye. The painted centerline
    matters because the 3D render draws one (blender_scene.py, from the same
    DesignState.centerline_style this reads) and this view is supposed to show what that
    render will show.

    The paint starts where src/render/crosswalks.py:centerline_start_ft says, which is at the
    stop bar - the same rule the export uses, not a second copy of it.
    """
    state = scene.state
    for leg_name, leg in state.legs.items():
        # The datum: thin, grey, dotted, the full length of the leg. Deliberately
        # unobtrusive - it is a construction line, not a marking on the road.
        ax.plot(*leg.centerline.xy, color="#3b6ea5", lw=0.9, ls=(0, (7, 3, 1, 3)), alpha=0.9,
                zorder=4)

        style = state.centerline_style(leg_name)
        if style == "none" or leg_name not in scene.crosswalk_offsets:
            continue
        start_ft = centerline_start_ft(scene.crosswalk_offsets[leg_name].offset_ft,
                                        scene.stop_bar_offsets.get(leg_name),
                                        leg_name in scene.marked_crosswalks)
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

    Read per STRETCH of kerb, not per leg. A restriction covering only the approach to the
    junction is how OSM records "no parking for the first 100 ft", and a label that reduced the
    kerb to one value could only report one end of it - which is how East Broad came to be
    labelled "parking OK" over a kerb whose first 80 ft are tagged no_parking. See
    src/geometry/treatments.py:RestrictionSummary.
    """
    from src.geometry.model import curb_point_at_station
    from src.geometry.targets import BOTH_SIDES, LegSide, LegTarget
    from src.geometry.treatments import (AddBikeLane, LaneNarrowing, MarkedParking,
                                         MIN_MARKED_PARKING_DEPTH_FT, TARGET_LANE_WIDTH_FT,
                                         _restriction_summary)

    for leg_name, leg in state.legs.items():
        if leg.curb_to_curb_ft is None:
            continue
        allowance_ft = leg.curb_to_curb_ft / 2 - TARGET_LANE_WIDTH_FT
        narrowing = state.treatment_for(LaneNarrowing, LegTarget(leg_name))
        for side in BOTH_SIDES:
            kerb = LegSide(leg_name, side)
            bike_lane = state.treatment_for(AddBikeLane, kerb)
            at = _restriction_summary(state, leg_name, side, leg.centerline.length)
            if at.restricted_throughout:
                kind, says = "restricted", at.worst_value
            elif at.restricted_in_part:
                # The case a single value cannot express. Named on the drawing with the stretch
                # it covers, because "which 80 ft" is the whole content of the fact.
                kind = "restricted"
                says = at.describe().replace("OSM says ", "").replace("'", "")
            elif at.stated_ft > 0:
                kind, says = "allowed", "none (parking OK)"
            else:
                kind, says = "untagged", "untagged"

            if bike_lane is not None:
                drew = f"bike lane, {bike_lane.width_ft:.0f} ft"
            elif state.treatment_for(MarkedParking, kerb) is not None:
                # Naming the carve-out matters on a partly-restricted kerb: "stalls" alone, next
                # to a label saying no_parking over the first 80 ft, reads as a contradiction
                # rather than as the two facts it is.
                drew = ("stalls beyond it" if at.restricted_in_part else "stalls")
            elif narrowing is not None and side in narrowing.sides:
                # Three reasons a kerb ends up hatched, and the label may only claim the one
                # that applies. It used to attribute every unrestricted hatched kerb to
                # insufficient width, which on Broad St reads "only 15.0 ft spare, under a 8 ft
                # stall" - self-contradictory, because 15 is not under 8. What is really going
                # on there is that the borough ordinance prohibits parking where OSM carries no
                # tag at all, so the scenario hatched it deliberately; the label cannot know
                # which, so it stops asserting and says what it can see.
                if kind == "restricted":
                    drew = "hatched"
                elif allowance_ft < MIN_MARKED_PARKING_DEPTH_FT:
                    drew = (f"hatched: only {allowance_ft:.1f} ft spare, under a "
                            f"{MIN_MARKED_PARKING_DEPTH_FT:.0f} ft stall")
                else:
                    drew = (f"hatched by this proposal, though {allowance_ft:.1f} ft is spare "
                            f"- see the scenario for why")
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


def _label_paint(ax, state, paint):
    """Dimension labels for the curbside paint: what each treatment actually measures.

    One lane label PER SIDE narrowed, offset into that lane - not a single label on the
    centerline, which reads as "this road is one 11 ft lane" rather than what is there
    (a leg is not always narrowed on both sides).
    """
    from src.geometry.targets import LegSide
    from src.geometry.treatments import LaneNarrowing, MarkedParking

    for narrowing in state.treatments_of(LaneNarrowing):
        leg_name = narrowing.target.leg
        leg = state.legs[leg_name]
        lane_ft = leg.curb_to_curb_ft / 2 - narrowing.stripe_width_ft
        along_ft = min(leg.centerline.length * 0.6, leg.centerline.length - 5)
        for side in narrowing.sides:
            sign = side.sign
            at = leg.centerline.offset_curve(sign * lane_ft / 2).interpolate(along_ft)
            ax.annotate(f"lane {lane_ft:.1f} ft", (at.x, at.y), fontsize=6.5, color="goldenrod",
                        ha="center", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    # One label per RUN of stalls, not per side: a hydrant mid-block splits a kerb into two
    # separate runs, and a single label would be counting stalls that are not in one place.
    for piece in paint:
        if piece.kind is not markings.PARKING_EDGE_LINE:
            continue
        parking = state.treatment_for(MarkedParking, LegSide(piece.leg, piece.side))
        if parking is None:
            continue
        n_stalls = int(piece.geometry.length // parking.stall_length_ft)
        mid = piece.geometry.interpolate(0.5, normalized=True)
        ax.annotate(f"parking\n{n_stalls} stalls ({parking.depth_ft:.0f} ft)", (mid.x, mid.y),
                    fontsize=6, color="steelblue", ha="center", va="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))


def _mark_violations(ax, scene: SceneGeometry, props, paint):
    """Run the scene invariants and draw whatever failed, right where it failed.

    The plan view reports rather than raises, and the phase script asserts after saving -
    so a failure always arrives with a picture of itself. Reading "tactile pad 40% in the
    roadway at (419160, 566742)" next to a red ring around that exact pad is the difference
    between one round trip and several.

    Checked against `scene`, so this validates the geometry the figure above actually drew.
    It used to re-resolve everything from `crossings` and rebuild the crossing bands without
    the mutual-exclusion reaches, which made it a check on a third set of geometry that
    neither the 2D view nor the 3D render used.
    """
    violations = scene.check(props, paint)

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
        Line2D([0], [0], color="black", lw=2.2, label="Traced kerb - RAISED (OSM kerb=raised)"),
        Line2D([0], [0], color="black", lw=1.1, ls="--",
               label="Traced kerb - LOWERED: a vehicle crosses, so the paint opens"),
        Patch(facecolor="#8a7a68", alpha=0.55, edgecolor="#5d5044",
               label="Driveway (OSM service=driveway) - width DRAWN is assumed"),
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
        # A curb extension's tightened face is drawn as this same arc - it IS a corner fillet,
        # solved at the radius the extension presents (src/geometry/treatments.py).
        Line2D([0], [0], color="darkorange", lw=2.5,
                label="Corner fillet / curb extension face (radius labeled)"),
        Line2D([0], [0], color="seagreen", lw=6, alpha=0.6, label="Pedestrian refuge island"),
        Line2D([0], [0], color="slateblue", lw=6, alpha=0.35, label="Raised crossing"),
        Patch(facecolor="gold", alpha=0.5, hatch="//", edgecolor="goldenrod", label="Lane narrowing / corner hatching"),
        Line2D([0], [0], color="goldenrod", lw=1.5, label="Lane narrowing - line only (no chevron fill)"),
        Patch(facecolor="orangered", alpha=0.40, hatch="xx", edgecolor="orangered",
               label="Daylighting - no parking (R.S. 39:4-138)"),
        Patch(facecolor="peru", alpha=0.6, edgecolor="saddlebrown", label="Mountable apron"),
        # One row for both kinds: the dotted extension is the same paint, and the dashes are in
        # the geometry rather than in the line style, so a second swatch would look identical.
        Line2D([0], [0], color="seagreen", lw=1.6,
               label="Bike lane - edge lines (dotted across a driveway)"),
        Patch(facecolor="mediumseagreen", alpha=0.45, edgecolor="seagreen",
               label="Bike lane - green surface"),
        Patch(facecolor="mediumseagreen", alpha=0.35, hatch="\\\\", edgecolor="seagreen",
               label="Bike lane buffer"),
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
        Line2D([0], [0], marker="D", color="gold", markeredgecolor="black", lw=0, label="RRFB beacon (OSM)"),
        Line2D([0], [0], marker="P", color="firebrick", lw=0, label="Fire hydrant (OSM)"),
        Line2D([0], [0], color="saddlebrown", lw=1.5, label="Corner parcel"),
        Line2D([0], [0], marker="o", color="blue", lw=0, label="Intersection"),
    ]
