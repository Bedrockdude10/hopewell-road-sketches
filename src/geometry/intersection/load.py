"""load_intersection_model: config.yaml in, IntersectionModel out.

The one place the order of operations is written down - match, snap, fit, centre, join, corner - and
that order is load-bearing, because each step measures against what the last one produced."""


import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import substring

from src.sources.data_loader import load_parcels_near, load_road_network
from src.geometry.cross_streets import cross_streets_ft
from src.render.frame import frame_scale
from src.geometry.model import (
    Leg,
    assign_kerbs_to_corners,
    build_corner_fillets,
    build_pavement_polygon,
    corner_radii_from_kerbs,
    buffer_point_wgs84,
    clip_to_radius,
    label_quadrants,
    nearest_per_quadrant,
    reproject_to_state_plane,
    split_leg_centerlines,
)
from src.site import load_site_config
from src.geometry.intersection.junction import ROOT_DIR, IntersectionModel
from src.geometry.intersection.kerb_sources import (_kerb_lines_ft,
                                                    kerb_lines_with_tags_ft)
from src.geometry.intersection.fitting import (_centre_legs_on_traced_kerbs,
                                               _extend_curbs_with_far_tracing,
                                               _fit_legs_to_traced_kerbs, _join_through_legs,
                                               _widths_from_traced_kerbs)
from src.geometry.intersection.osm_roads import (SNAP_REPORT_THRESHOLD_FT,
                                                 _assign_leg_pieces,
                                                 _match_legs_to_osm_roads,
                                                 _snap_distance_ft, _snap_to_center)
from src.geometry.intersection.paved import _paved_surfaces_ft



def _build_corners(legs: dict, radius_ft: float, corner_radii: dict, kerb_lines: list) -> dict:
    """Corner geometry, preferring the surveyor's traced kerb over a fitted fillet.

    A traced kerb IS the curb, so it is used as the corner directly and the leg curb lines
    are trimmed to meet its ends. Fitting a circle to it and redrawing that circle off our
    own estimated curb lines kept the curvature but lost the position - the synthesised
    arcs sat 0.2-5.9 ft from the mapped kerb at Broad & Greenwood.

    Falls back to the fitted fillet for the WHOLE junction if the traced corners can't
    close into a valid pavement ring. A traced kerb can be too short, or stop before the
    tangent point, and half a ring built from real geometry is worse than a consistent
    approximation - so the fallback is all-or-nothing and says so.
    """
    corner_arcs = assign_kerbs_to_corners(legs, kerb_lines)
    fillets = build_corner_fillets(legs, radius_ft, corner_radii, corner_arcs)
    traced = [k for k, v in fillets.items() if v.get("source") == "traced_kerb"]
    if not traced:
        return fillets
    try:
        build_pavement_polygon(fillets)
    except ValueError as e:
        print(f"  NOTE: traced kerbs don't close into a valid pavement ring here ({e}). Falling back to "
              f"fitted fillets for this junction; the corners will sit a few feet off the mapped kerb.")
        return build_corner_fillets(legs, radius_ft, corner_radii)
    names = ", ".join("/".join(sorted(c)) for c in traced)
    print(f"  NOTE: corner geometry taken directly from the traced OSM kerb at: {names}.")
    return fillets


def _corner_radii_from_osm(center_wgs84: Point, center_ft: Point, legs: dict, fallback_ft: float) -> dict:
    """Per-corner radii from traced kerbs, reporting what was and wasn't usable."""
    radii, notes = corner_radii_from_kerbs(legs, _kerb_lines_ft(center_wgs84, center_ft), fallback_ft)
    for note in notes:
        print(f"  NOTE: {note}")
    return radii


def load_intersection_model(config: dict | None = None, site: str | None = None) -> IntersectionModel:
    """Pass either a pre-loaded `config` dict, or a `site` name to load it fresh
    (defaults to src.site.DEFAULT_SITE if neither is given)."""
    if config is None:
        from src.site import DEFAULT_SITE
        config = load_site_config(site or DEFAULT_SITE)

    lon, lat = config["intersection"]["center_wgs84"]
    center = Point(lon, lat)
    center_ft = gpd.GeoSeries([center], crs="EPSG:4326").to_crs("EPSG:3424").iloc[0]

    data_sources = config.get("data_sources", {})
    road_network_path = ROOT_DIR / data_sources["road_network"]
    parcels_path = ROOT_DIR / data_sources["parcels"]

    clip_radius_m = config["intersection"]["clip_radius_m"]
    bbox = buffer_point_wgs84(center, clip_radius_m * 1.3)
    network = load_road_network(bbox=bbox, path=road_network_path)
    clipped = clip_to_radius(network, center, clip_radius_m)
    clipped_ft = reproject_to_state_plane(clipped)

    working_len = config["intersection"]["leg_working_length_ft"]
    legs_cfg = config["legs"]
    # How far to carry each leg, PER LEG. The site-wide value is the default; a leg may
    # override it with legs.<name>.working_length_ft. Legs are not interchangeable in how far
    # they can honestly be drawn - a curb is traced as far as somebody traced it, and past
    # that curb_line_from_points extrapolates from a bearing. So the length that shows the
    # most ground truth on an arterial is longer than the one that starts inventing kerb on
    # the cross street. One global forces the shorter answer on both.
    # What the SITE says each leg is worth drawing. Kept separately from the scaled figure below
    # because it is the surveyed answer: it is what the frame is measured against (so a zoom-out
    # does not compound), and what src/metrics.py splits measured from projected on.
    surveyed_leg_lengths = {name: float(cfg.get("working_length_ft", working_len))
                            for name, cfg in legs_cfg.items()}
    # Carried out with the frame, so a treatment runs as far as the picture draws the street
    # rather than stopping a third of the way along it. Three things make this safe rather than
    # a licence to invent:
    #   * the width fit is bounded at stations TRACED_SECTION_START_FT..TRACED_SECTION_END_FT
    #     "whatever working_length_ft says", so no measured width moves;
    #   * a leg is cut from the NJDOT alignment with substring(), so it simply stops where the
    #     surveyed alignment stops - asking for more than exists yields what exists;
    #   * past the traced kerb, curb_line_from_points already extrapolates from a bearing, and
    #     that is drawn with the provenance it always was.
    # What it does NOT make safe is the arithmetic downstream - a leg twice as long carries twice
    # the kerb to mark parking along - so SceneMetrics reports the projected part separately
    # rather than letting a stall count move with a camera setting.
    scale = frame_scale()
    leg_lengths = {name: length * scale for name, length in surveyed_leg_lengths.items()}
    sri_to_leg_names: dict[str, list[str]] = {}
    for name, leg_cfg in legs_cfg.items():
        sri_to_leg_names.setdefault(leg_cfg["sri"], []).append(name)

    legs: dict[str, Leg] = {}
    for sri, leg_names in sri_to_leg_names.items():
        rows = clipped_ft[clipped_ft["SRI"] == sri]
        if rows.empty:
            print(f"  WARNING: SRI {sri} not found in clipped network - skipping legs {leg_names}.")
            continue
        line = rows.iloc[0].geometry
        # NJDOT digitization sometimes has sub-foot vertex noise near
        # intersections; that noise creates enough local curvature to make
        # offset_curve() self-intersect (returns a MultiLineString) at typical
        # curb-to-curb widths. A few feet of simplification removes it without
        # affecting real road geometry at this scale.
        # Split to the LONGEST length wanted on this SRI - which leg a piece is cannot be
        # known until _assign_leg_pieces has matched it by bearing, so cutting to a specific
        # leg's length here would truncate the wrong one. Each piece is trimmed to its own
        # leg's length below, after it has a name.
        split_len = max(leg_lengths[n] for n in leg_names)
        pieces = [p.simplify(3.0) for p in split_leg_centerlines(line, center_ft, split_len)]
        pieces = [_snap_to_center(p, center_ft) for p in pieces]
        snap_ft = _snap_distance_ft(line, center_ft)
        if snap_ft > SNAP_REPORT_THRESHOLD_FT:
            print(f"  NOTE: SRI {sri} centerline passes {snap_ft:.1f} ft from the resolved intersection "
                  f"node - snapped onto it (legs {', '.join(leg_names)}).")
        for name, piece in _assign_leg_pieces(pieces, leg_names, legs_cfg, center_ft, sri).items():
            # Trimmed from the snapped junction end outward, so a shortened leg keeps the
            # station-0 origin every measurement in the project is taken from.
            if piece.length > leg_lengths[name]:
                piece = substring(piece, 0, leg_lengths[name])
            legs[name] = Leg(name=name, centerline=piece, curb_to_curb_ft=legs_cfg[name].get("curb_to_curb_ft"))

    kerb_lines = _kerb_lines_ft(center, center_ft)
    for name, width_ft in _widths_from_traced_kerbs(legs, kerb_lines, legs_cfg).items():
        legs[name] = Leg(name=name, centerline=legs[name].centerline, curb_to_curb_ft=width_ft)

    # The NEAR set for the fit: the ways around the junction, which is what a width and a
    # corner radius are measured from. _extend_curbs_with_far_tracing adds the rest afterwards,
    # once the widths can no longer be moved by them.
    kerb_ways = kerb_lines_with_tags_ft(center, center_ft)
    # Twice, deliberately. The first pass only has to be good enough to collect each leg's
    # traced vertices; that gives _resize_and_centre_from_traced_kerbs a real cross-section
    # to measure, and the corrected width and centre then change which vertices belong to
    # which leg side, so the assignment is redone in the corrected frame. Silent the first
    # time round - the coverage it reports is about the final geometry, not the scaffold.
    near_coverage = _fit_legs_to_traced_kerbs(legs, kerb_ways, center_ft, legs_cfg)
    _extend_curbs_with_far_tracing(legs, center, center_ft, near_coverage)
    # Last, on the settled widths and the fullest tracing: the alignment is bent onto the
    # carriageway centre over the WHOLE leg. Everything below measures in this frame - the
    # road match, the cross streets, the corner fillets - so it happens before any of them.
    _centre_legs_on_traced_kerbs(legs)
    # ...and then the two halves of a through street are made to agree with each other, which
    # centring each on its own kerbs cannot do. Both run before anything below measures in
    # this frame.
    _join_through_legs(legs)
    leg_road_spans = _match_legs_to_osm_roads(legs, center, center_ft)
    # The way covering MOST of the leg carries its whole-leg tags. Not the way nearest the
    # leg's midpoint, which is what this used to pick: on a split leg those differ, and the
    # nearest-to-midpoint rule has no claim to describing the leg as a whole.
    dominant = {name: max(spans, key=lambda span: span.length_ft)
                for name, spans in leg_road_spans.items()}
    leg_osm_tags = {name: span.tags for name, span in dominant.items()}
    leg_osm_aligned = {name: span.aligned for name, span in dominant.items()}
    for name, spans in sorted(leg_road_spans.items()):
        if len(spans) < 2:
            continue
        # A split leg is worth saying out loud: it is how OSM records a fact that changes part
        # way along a street, and reading only one of the ways is how such a fact disappears.
        detail = "; ".join(f"way {s.way_id} over {s.start_ft:.0f}-{s.end_ft:.0f} ft" for s in spans)
        print(f"  NOTE: {name} is covered by {len(spans)} OSM ways ({detail}). Kerbside parking "
              f"is read per span; whole-leg tags come from the longest.")

    radius_ft = config["treatments"]["existing_corner_radius_ft"]
    # Traced OSM kerbs give a real measured radius per corner where they exist; the
    # config value stays as the fallback for untraced corners. Nothing in OSM carries a
    # corner radius as a tag, so a traced kerb line is the only way to source this.
    corner_radii = {}
    corner_fillets = {}
    if radius_ft:
        corner_radii = _corner_radii_from_osm(center, center_ft, legs, radius_ft)
        corner_fillets = _build_corners(legs, radius_ft, corner_radii, kerb_lines)

    parcels = load_parcels_near(center, radius_ft=300, path=parcels_path)
    corner_parcels = nearest_per_quadrant(label_quadrants(parcels, center_ft))

    return IntersectionModel(
        config=config,
        center_wgs84=center,
        center_ft=center_ft,
        legs=legs,
        corner_fillets=corner_fillets,
        leg_road_spans=leg_road_spans,
        leg_osm_tags=leg_osm_tags,
        leg_osm_aligned=leg_osm_aligned,
        parcels=parcels,
        corner_parcels=corner_parcels,
        paved_surfaces=_paved_surfaces_ft(center, corner_fillets),
        surveyed_leg_lengths=surveyed_leg_lengths,
        cross_streets=cross_streets_ft(center, center_ft, legs),
    )
