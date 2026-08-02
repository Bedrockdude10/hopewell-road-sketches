"""Assemble the full intersection model (legs, curb lines, corner fillets, parcels)
from a site's config.yaml + the data sources it points to. Shared by every phase
script, for every site - nothing in this module is specific to any one
intersection (see sites/README.md for what a site provides instead)."""
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point

from src.sources.data_loader import load_parcels_near, load_road_network
from src.render.coords import wgs84_to_state_plane
from src.sources.osm_context import fetch_kerbs
from src.geometry.model import (
    Leg,
    build_corner_fillets,
    corner_radii_from_kerbs,
    buffer_point_wgs84,
    clip_to_radius,
    label_quadrants,
    nearest_per_quadrant,
    reproject_to_state_plane,
    split_leg_centerlines,
)
from src.site import load_site_config

ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # src/geometry/intersection.py -> repo root


@dataclass
class IntersectionModel:
    config: dict
    center_wgs84: Point
    center_ft: Point
    legs: dict[str, Leg]
    corner_fillets: dict
    parcels: gpd.GeoDataFrame
    corner_parcels: gpd.GeoDataFrame


def _bearing_deg(from_pt, to_pt) -> float:
    """Compass bearing (0=N, 90=E, clockwise) from from_pt to to_pt."""
    dx, dy = to_pt[0] - from_pt[0], to_pt[1] - from_pt[1]
    return (90 - np.degrees(np.arctan2(dy, dx))) % 360


def _bearing_diff(a: float, b: float) -> float:
    """Smallest angular difference between two compass bearings, in [0, 180]."""
    return abs((a - b + 180) % 360 - 180)


# How far a road network centerline may sit from the resolved intersection node before
# the snap below is worth reporting. Sub-foot gaps are digitizing noise; anything larger
# is a real disagreement between the two sources and worth seeing in the phase output.
SNAP_REPORT_THRESHOLD_FT = 2.0
KERB_CONTEXT_RADIUS_M = 120  # fetch radius, metres - generous enough to catch a whole return
KERB_NEAR_JUNCTION_FT = 80   # but a return belonging to THIS junction is within this of centre
KERB_PLAUSIBLE_HALF_WIDTH_FT = (8.0, 45.0)  # a kerb this far off a centerline is that leg's kerb


def _snap_distance_ft(line, center_ft: Point) -> float:
    """Perpendicular distance from the resolved intersection node to a road centerline."""
    return line.distance(center_ft)


def _snap_to_center(piece, center_ft: Point):
    """Translate a leg centerline piece so it starts exactly at the resolved intersection
    node, instead of at the nearest point on the road network's own line.

    WHY: the intersection LOCATION comes from OSM (a real shared junction node,
    cross-checked against the NJDOT SLD milepost - see data_loader.geocode_intersection),
    and so does every piece of context placed against it: the surveyed pedestrian
    crossings (src/render/crosswalks.py), buildings, and mapped footways. The leg
    CENTERLINES come from NJDOT's SRI linear-referencing layer. Those two frames don't
    agree, and on the state/county routes the disagreement is large and systematic:

        Route 518 at Greenwood Ave    8.4 ft        Greenwood Ave (local)   0.3 ft
        Route 518 at Princeton Ave    8.7 ft        Princeton Ave (CR 569)  1.4 ft
        CR 654 at Louellen St        16.3 ft        Columbia Ave (local)    1.8 ft

    ~8.5 ft on Route 518 at two independent junctions is a parallel offset in NJDOT's
    route alignment, not noise - an SRI line is a linear-referencing alignment, not a
    surveyed physical centerline, and some of these diagrams are 15 years old. Left
    uncorrected it shifts the whole modelled roadway sideways relative to the OSM
    crossings: at E Broad & Princeton it put 100% of the Princeton crosswalk inside the
    E Broad roadway. Snapping the centerline onto the node drops that to 22%, and the
    remainder is the leg's own (estimated) width.

    TRADE-OFF: this translates the entire piece, so its far end moves off NJDOT's
    alignment by the same amount (up to ~8 ft at 130 ft out). That's accepted
    deliberately - accuracy at the intersection, which is the whole subject of this
    project, beats accuracy at the far end of a leg that's only there for context. The
    offset is reported above whenever it exceeds SNAP_REPORT_THRESHOLD_FT so it's never
    silent. Bearings, lengths and widths are all unchanged; only position moves.
    """
    x0, y0 = piece.coords[0]
    return affinity.translate(piece, xoff=center_ft.x - x0, yoff=center_ft.y - y0)


def _assign_leg_pieces(pieces: list, leg_names: list[str], legs_cfg: dict, center_ft: Point) -> dict[str, object]:
    """
    Match centerline pieces (all sharing one SRI, split at the intersection) to
    the configured leg names that reference that SRI, by nearest compass bearing.
    Generalizes to any number of pieces per SRI (2 for a through road, 1 for a
    dead-end/stub leg) and any intersection shape - nothing here assumes a
    4-way or perpendicular roads, only that each leg's config entry has an
    accurate `bearing_deg`.
    """
    assigned = {}
    remaining_names = list(leg_names)
    for piece in pieces:
        far_bearing = _bearing_deg((center_ft.x, center_ft.y), piece.coords[-1])
        best_name = min(remaining_names, key=lambda n: _bearing_diff(far_bearing, legs_cfg[n]["bearing_deg"]))
        assigned[best_name] = piece
        remaining_names.remove(best_name)
    return assigned


def kerb_lines_with_tags_ft(center_wgs84: Point, center_ft: Point) -> list:
    """[(LineString, tags)] for traced kerbs near the junction - geometry plus what OSM
    says about each (kerb=lowered, tactile_paving=yes, wheelchair=yes)."""
    try:
        kerbs = fetch_kerbs(center_wgs84, radius_m=KERB_CONTEXT_RADIUS_M)
    except RuntimeError:
        return []
    out = []
    for kerb in kerbs:
        coords = kerb.get("coords_wgs84")
        if not coords:
            continue
        xs, ys = wgs84_to_state_plane.transform([c[0] for c in coords], [c[1] for c in coords])
        line = LineString(zip(xs, ys))
        if line.distance(center_ft) <= KERB_NEAR_JUNCTION_FT:
            out.append((line, kerb.get("tags", {})))
    return out


def _kerb_lines_ft(center_wgs84: Point, center_ft: Point) -> list[LineString]:
    """Traced OSM kerb ways near this junction, in state-plane feet.

    Only kerbs within KERB_NEAR_JUNCTION_FT of the centre are kept. The fetch radius has
    to be generous enough to catch a whole corner return, but at 120 m it also pulls in
    kerbs belonging to NEIGHBOURING junctions - and those were being assigned to this
    junction's corners, producing a nonsense 7.9-30.2 ft spread at Columbia & Princeton.
    A corner return sits within a few tens of feet of the junction it belongs to.
    """
    try:
        kerbs = fetch_kerbs(center_wgs84, radius_m=KERB_CONTEXT_RADIUS_M)
    except RuntimeError as e:
        print(f"  WARNING: could not fetch OSM kerbs ({e}) - corner radii stay placeholders.")
        return []

    lines = []
    for kerb in kerbs:
        coords = kerb.get("coords_wgs84")
        if not coords:
            continue  # a lone barrier=kerb node has no arc to fit
        xs, ys = wgs84_to_state_plane.transform([c[0] for c in coords], [c[1] for c in coords])
        line = LineString(zip(xs, ys))
        if line.distance(center_ft) <= KERB_NEAR_JUNCTION_FT:
            lines.append(line)
    return lines


def _widths_from_traced_kerbs(legs: dict, kerb_lines: list, legs_cfg: dict) -> dict:
    """Leg half-widths measured off traced OSM kerbs, where they beat what we had.

    A corner return is TANGENT to its leg's curb line, so the closest approach of a traced
    kerb to a leg centerline is that leg's half-width. (Away from the tangent point the
    return curves off around the corner, so the measurement is an upper bound - the real
    curb is at most this far out, never further.)

    That one-sidedness is what makes it safe to apply: a traced kerb closer to the
    centerline than our modelled curb PROVES the modelled road is too wide. It is only
    used to narrow a leg, never to widen one, and never against a field measurement -
    those win outright (src/provenance.py), though a conflict is reported.

    This is what stops the render disagreeing with the mapping: previously the sidewalk
    band and pavement edge came from estimated widths, so they sat several feet off the
    kerb the surveyor had actually traced.
    """
    from src.provenance import FIELD_MEASURED, leg_width_provenance

    updates = {}
    for name, leg in legs.items():
        if leg.curb_to_curb_ft is None:
            continue
        candidates = []
        for kerb in kerb_lines:
            distance = kerb.distance(leg.centerline)
            nearest_along = leg.centerline.project(kerb.interpolate(kerb.project(leg.centerline.centroid)))
            if not (0 < nearest_along < leg.centerline.length):
                continue
            if KERB_PLAUSIBLE_HALF_WIDTH_FT[0] <= distance <= KERB_PLAUSIBLE_HALF_WIDTH_FT[1]:
                candidates.append(distance)
        if not candidates:
            continue

        measured_half_ft = min(candidates)
        current_half_ft = leg.curb_to_curb_ft / 2
        if measured_half_ft >= current_half_ft - 0.5:
            continue  # traced kerb agrees, or sits outside our curb - nothing proven

        tier = leg_width_provenance(legs_cfg.get(name, {}))
        if tier == FIELD_MEASURED:
            print(f"  CONFLICT: {name} is field-measured at {leg.curb_to_curb_ft:.1f} ft, but the traced "
                  f"OSM kerb comes within {measured_half_ft:.1f} ft of its centerline "
                  f"({measured_half_ft * 2:.1f} ft curb-to-curb). The measurement stands; check whether "
                  f"it included a shoulder or parking zone beyond the kerb.")
            continue
        updates[name] = measured_half_ft * 2
        print(f"  NOTE: {name} narrowed {leg.curb_to_curb_ft:.1f} -> {measured_half_ft * 2:.1f} ft from the "
              f"traced OSM kerb (osm_derived; the kerb is tangent to the curb line, so this is where the "
              f"curb actually is).")
    return updates


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
        pieces = [p.simplify(3.0) for p in split_leg_centerlines(line, center_ft, working_len)]
        pieces = [_snap_to_center(p, center_ft) for p in pieces]
        snap_ft = _snap_distance_ft(line, center_ft)
        if snap_ft > SNAP_REPORT_THRESHOLD_FT:
            print(f"  NOTE: SRI {sri} centerline passes {snap_ft:.1f} ft from the resolved intersection "
                  f"node - snapped onto it (legs {', '.join(leg_names)}).")
        for name, piece in _assign_leg_pieces(pieces, leg_names, legs_cfg, center_ft).items():
            legs[name] = Leg(name=name, centerline=piece, curb_to_curb_ft=legs_cfg[name].get("curb_to_curb_ft"))

    kerb_lines = _kerb_lines_ft(center, center_ft)
    for name, width_ft in _widths_from_traced_kerbs(legs, kerb_lines, legs_cfg).items():
        legs[name] = Leg(name=name, centerline=legs[name].centerline, curb_to_curb_ft=width_ft)

    radius_ft = config["treatments"]["existing_corner_radius_ft"]
    # Traced OSM kerbs give a real measured radius per corner where they exist; the
    # config value stays as the fallback for untraced corners. Nothing in OSM carries a
    # corner radius as a tag, so a traced kerb line is the only way to source this.
    corner_radii = {}
    if radius_ft:
        corner_radii = _corner_radii_from_osm(center, center_ft, legs, radius_ft)
    corner_fillets = build_corner_fillets(legs, radius_ft, corner_radii) if radius_ft else {}

    parcels = load_parcels_near(center, radius_ft=300, path=parcels_path)
    corner_parcels = nearest_per_quadrant(label_quadrants(parcels, center_ft))

    return IntersectionModel(
        config=config,
        center_wgs84=center,
        center_ft=center_ft,
        legs=legs,
        corner_fillets=corner_fillets,
        parcels=parcels,
        corner_parcels=corner_parcels,
    )
