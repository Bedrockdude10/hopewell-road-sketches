"""Geometry operations: WGS84 buffering, radius clipping, CRS reprojection, and
curb-line / corner-fillet construction from centerlines + widths."""
from dataclasses import dataclass, field
from functools import lru_cache

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import substring
from shapely.validation import explain_validity

WGS84 = "EPSG:4326"
NJ_STATE_PLANE_FT = "EPSG:3424"  # NAD83(HARN) / New Jersey (ftUS)


@lru_cache(maxsize=64)
def _utm_crs_at(lon: float, lat: float):
    """The local UTM CRS for a WGS84 point.

    Cached because geopandas' estimate_utm_crs() is the single most expensive call in this
    pipeline and it is asked the same question over and over: it queries the PROJ database for
    every candidate CRS and takes ~38 ms, and a scenario export made 43 of those - 1.67 s out
    of a 2.76 s export, all of them about one intersection. The answer depends only on the
    point, and UTM zones are 6 degrees wide, so this is a pure function being called with a
    handful of distinct arguments.

    Keyed on (lon, lat) rather than the Point, because a Point is unhashable and two Points at
    the same place are different objects.
    """
    return gpd.GeoSeries([Point(lon, lat)], crs=WGS84).estimate_utm_crs()


@lru_cache(maxsize=256)
def _buffer_bounds_wgs84(lon: float, lat: float, radius_m: float) -> tuple[float, float, float, float]:
    utm_crs = _utm_crs_at(lon, lat)
    point_gs = gpd.GeoSeries([Point(lon, lat)], crs=WGS84)
    buffered = point_gs.to_crs(utm_crs).buffer(radius_m).to_crs(WGS84)
    return tuple(buffered.total_bounds)


def buffer_point_wgs84(point: Point, radius_m: float) -> tuple[float, float, float, float]:
    """Buffer a WGS84 point by radius_m meters (via a local UTM projection) and
    return a WGS84 bbox as (minx, miny, maxx, maxy).

    Memoized on (lon, lat, radius): src/sources/osm_context.py calls this twice per OSM fetch
    (once to bound the layer, once in assert_within_snapshot) and there are eight fetchers
    called repeatedly per site, all about the same centre. See _utm_crs_at.
    """
    return _buffer_bounds_wgs84(point.x, point.y, float(radius_m))


def clip_to_radius(gdf: gpd.GeoDataFrame, center: Point, radius_m: float) -> gpd.GeoDataFrame:
    """Clip a WGS84 GeoDataFrame to a circular radius (meters) around center,
    trimming feature geometry (not just filtering by bbox)."""
    center_gs = gpd.GeoSeries([center], crs=WGS84)
    utm_crs = _utm_crs_at(center.x, center.y)
    center_utm = center_gs.to_crs(utm_crs).iloc[0]
    circle_wgs84 = gpd.GeoSeries([center_utm.buffer(radius_m)], crs=utm_crs).to_crs(WGS84).iloc[0]

    clipped = gdf[gdf.intersects(circle_wgs84)].copy()
    clipped["geometry"] = clipped.intersection(circle_wgs84)
    return clipped[~clipped.geometry.is_empty]


def reproject_to_state_plane(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to NJ State Plane, NAD83(HARN) (feet)."""
    return gdf.to_crs(NJ_STATE_PLANE_FT)


def label_quadrants(gdf_ft: gpd.GeoDataFrame, center_ft: Point) -> gpd.GeoDataFrame:
    """Label each feature's compass quadrant (NE/NW/SE/SW) relative to a center
    point, plus its distance in feet - used to locate corner parcels."""
    out = gdf_ft.copy()
    out["dist_ft"] = out.geometry.distance(center_ft)
    centroids = out.geometry.centroid
    out["quadrant"] = [
        ("N" if cy > center_ft.y else "S") + ("E" if cx > center_ft.x else "W")
        for cx, cy in zip(centroids.x, centroids.y)
    ]
    return out


def nearest_per_quadrant(gdf_ft: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Given the output of label_quadrants, return the closest feature per quadrant."""
    return gdf_ft.sort_values("dist_ft").groupby("quadrant", as_index=False).first()


def split_leg_centerlines(line: LineString, center: Point, working_length_ft: float) -> list[LineString]:
    """
    Split a line at the point on it nearest `center`, returning up to two pieces
    that each start at that snapped point and extend outward (trimmed to at most
    working_length_ft) - one piece per side of the split.
    """
    snap_dist = line.project(center)
    total = line.length
    legs = []
    if snap_dist > 0:
        head = substring(line, 0, snap_dist)
        head = LineString(list(head.coords)[::-1])  # start at the snap point, head outward
        legs.append(substring(head, 0, min(working_length_ft, head.length)))
    if snap_dist < total:
        tail = substring(line, snap_dist, total)  # already starts at the snap point
        legs.append(substring(tail, 0, min(working_length_ft, tail.length)))
    return legs


@dataclass
class Leg:
    """One approach to an intersection: a centerline plus (if known) a curb-to-curb
    width, from which parallel curb lines are derived automatically."""
    name: str
    centerline: LineString  # starts at the point nearest the intersection, extends outward
    curb_to_curb_ft: float | None = None
    left_curb: LineString | None = None
    right_curb: LineString | None = None
    # Sides whose curb line is the surveyor's traced kerb rather than a centerline offset.
    # The corner between two traced sides is traced too, so it is joined and smoothed
    # instead of being replaced by a fitted fillet.
    traced_sides: set = field(default_factory=set)
    # Tier of curb_to_curb_ft AS BUILT, when that is better than what the config claimed.
    # A width measured between two traced kerbs is osm_derived however the config describes
    # its own estimate, and the phase summary and the plan view's curb styling both have to
    # say which one they are showing - "ESTIMATE / PLACEHOLDER" over a line drawn from a
    # surveyor's trace is the project's own principle stated backwards.
    width_provenance: str | None = None

    def __post_init__(self):
        if self.curb_to_curb_ft is not None:
            half = self.curb_to_curb_ft / 2
            self.left_curb = self.centerline.offset_curve(half)
            self.right_curb = self.centerline.offset_curve(-half)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


# Beyond this, two adjacent legs are the same street running through the junction, and the
# pair of curbs facing away from the stem is one continuous kerb with no corner in it. The
# original 179 deg only caught a perfectly straight through road; W Broad kinks 9.1 deg at
# Louellen, and rounding that "corner" is meaningless - the two curb rays converge so slowly
# that their crossing point lands 47 ft up the street, dragging the fillet's tangent points
# and the whole pavement ring with it. These are old streets; a through road that bends a
# few degrees at a side street is the normal case, not a corner.
THROUGH_STREET_ANGLE_DEG = 160.0
# 160 rather than 165 so that W Broad, which kinks 17.3 deg at Louellen (162.7 deg between
# the legs), counts as one street passing through - which is what it is. Its outer kerb runs
# unbroken past the junction and carries no crossing. Raising the tolerance meant the two
# legs' zones met at an angle and overlapped by 5.6 sq ft in the wedge between their frames;
# curbside_paint_ft's `shares_a_kerb` now has them butt instead. At E Broad & Princeton the
# pair is 179.9 deg apart, where the wedge is negligible and neither issue arises.


def fillet_curb_corner(
    curb_a: LineString, curb_b: LineString, radius_ft: float, n_points: int = 24
) -> tuple[LineString, LineString, LineString]:
    """
    Round the corner where two curb lines would otherwise meet at a sharp point.
    Each curb line is treated as a ray from its first vertex, in the direction of
    its first segment - so pass in curb lines that start near the intersection
    corner and extend outward (as produced by Leg / offset_curve).

    Returns (trimmed_curb_a, arc, trimmed_curb_b): concatenate the three pieces,
    in that order, for one continuous rounded curb path.

    Two curb lines meeting at ~180 degrees are not a corner at all - they're one
    straight run of curb. That is the normal case on the far side of a T or Y
    junction, where the through road's two legs are collinear and the pair of curbs
    facing away from the stem never actually turns (e.g. e_broad_st_east's left curb
    and e_broad_st_west's right curb, the continuous north edge of E Broad St at
    E Broad & Princeton, at 179.9 degrees). There's nothing to round, and no true
    corner vertex to round it about - the two curb rays are parallel, so solving for
    their crossing point is singular. Joined with a straight bridge instead.
    """
    pa, da = np.array(curb_a.coords[0]), _unit(np.array(curb_a.coords[1]) - np.array(curb_a.coords[0]))
    pb, db = np.array(curb_b.coords[0]), _unit(np.array(curb_b.coords[1]) - np.array(curb_b.coords[0]))

    theta = np.arccos(np.clip(np.dot(da, db), -1, 1))
    if theta < np.radians(1):
        # The curbs double back along each other - a real geometry problem, not a
        # flat corner. Still an error.
        raise ValueError(f"Curb lines meet at an implausible angle ({np.degrees(theta):.1f} deg) - check inputs.")
    if theta > np.radians(THROUGH_STREET_ANGLE_DEG):
        # Collinear: no rounding, no trimming. The "arc" is the straight bridge
        # across the small gap between the two curb lines' start points, which keeps
        # the (trimmed_a, arc, trimmed_b) contract - and build_pavement_polygon's
        # ring walk - working unchanged.
        bridge = LineString([tuple(pa), tuple(pb)])
        return curb_a, bridge, curb_b

    # true square-corner vertex: intersection of the two curb lines, extended
    a_matrix = np.array([da, -db]).T
    t, _s = np.linalg.solve(a_matrix, pb - pa)
    vertex = pa + t * da

    tangent_dist = radius_ft / np.tan(theta / 2)
    center_dist = radius_ft / np.sin(theta / 2)
    bisector = _unit(da + db)

    t1 = vertex + da * tangent_dist
    t2 = vertex + db * tangent_dist
    center = vertex + bisector * center_dist

    a1 = np.arctan2(t1[1] - center[1], t1[0] - center[0])
    a2 = np.arctan2(t2[1] - center[1], t2[0] - center[0])
    delta = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi  # shorter angular sweep, bulging toward the vertex
    angles = a1 + np.linspace(0, delta, n_points)
    arc = LineString([(center[0] + radius_ft * np.cos(a), center[1] + radius_ft * np.sin(a)) for a in angles])

    trimmed_a = substring(curb_a, curb_a.project(Point(*t1)), curb_a.length)
    trimmed_b = substring(curb_b, curb_b.project(Point(*t2)), curb_b.length)
    return trimmed_a, arc, trimmed_b


def _leg_bearing(leg: "Leg") -> float:
    p0 = np.array(leg.centerline.coords[0])
    p1 = np.array(leg.centerline.coords[1])
    d = p1 - p0
    return np.arctan2(d[1], d[0])


def _through_street(leg_a, leg_b) -> bool:
    """Whether these two legs are one street running through the junction rather than two
    streets meeting at a corner.

    Measured between the leg CENTERLINES, not between the first segments of their traced
    curbs. The curbs' first segments are wherever the surveyor's tracing happens to begin,
    which on a partially-traced side is somewhere up the block; and using each leg's chord
    rather than its near end matters at W Broad & Louellen, where louellen_st_west leaves the
    junction on a 15 ft stub bearing 239 deg before settling onto 269. By the stub it reads as
    178.6 deg from w_broad_st_northeast - a through street - and by the chord as 149.2, which
    is the truth: the route turns there, and the traced kerbs show a real 14 ft return.
    """
    theta = np.arccos(np.clip(np.dot(_line_direction(leg_a.centerline),
                                     _line_direction(leg_b.centerline)), -1, 1))
    return np.degrees(theta) > THROUGH_STREET_ANGLE_DEG


def build_corner_fillets(legs: dict, radius_ft, corner_radii: dict | None = None,
                          corner_arcs: dict | None = None) -> dict:
    """
    Given >=2 Legs with curb lines already computed, sort them by compass bearing
    and fillet the corner between each pair of angularly-adjacent legs (wrapping
    around). For a leg A immediately followed (counter-clockwise) by leg B, the
    corner between them is bounded by A's left curb and B's right curb.

    Returns {(name_a, name_b): {"trimmed_a", "arc", "trimmed_b"}} for each corner,
    or {"error": ...} in place of a corner whose fillet couldn't be built.
    """
    usable = {name: leg for name, leg in legs.items() if leg.left_curb is not None}
    if len(usable) < 2:
        return {}

    ordered = sorted(usable.items(), key=lambda kv: _leg_bearing(kv[1]))
    n = len(ordered)
    results = {}
    for i in range(n):
        name_a, leg_a = ordered[i]
        name_b, leg_b = ordered[(i + 1) % n]
        corner_key = frozenset((name_a, name_b))

        # A pair of legs that are the same street running THROUGH the junction has no corner
        # between them, so neither branch below applies: there is no return to trace and
        # nothing to round. Tested first because both of them would otherwise happily invent
        # one. e_broad_st_east and e_broad_st_west are 179.9 deg apart - the continuous north
        # edge of E Broad St, opposite the stem of the T - and traced_corner_join drew a
        # diagonal from one curb to the other whose start sat 67.1 ft up the leg. That became
        # the leg's corner-return "tangent point", which held the kerbside hatching 75 ft out
        # from a junction whose surveyed stop bar is at 52.9 ft. fillet_curb_corner has had
        # this test since the fitted path was the only path; it just never ran for a traced
        # corner, because the traced branches return before reaching it.
        if _through_street(leg_a, leg_b):
            results[(name_a, name_b)] = {
                "trimmed_a": leg_a.left_curb,
                "arc": LineString([leg_a.left_curb.coords[0], leg_b.right_curb.coords[0]]),
                "trimmed_b": leg_b.right_curb,
                # No radius key at all, rather than a None one: there is no corner here to
                # have a radius, and the plan view labels a corner's radius wherever the key
                # is present. A None slipped straight past that guard and crashed the 2D
                # build on an f-string.
                "source": "through_street", "through_street": True,
            }
            continue

        # Both sides traced means the corner between them is traced too - the return's own
        # vertices are already the inner ends of these two curbs. Nothing to fit: walk from
        # one to the other and smooth the seam. Fitting a circle here and redrawing it off
        # our own curb lines is what put the synthesised arcs 0.2-5.9 ft from the mapped
        # kerb at Broad & Greenwood.
        if "left" in leg_a.traced_sides and "right" in leg_b.traced_sides:
            trimmed_a, arc, trimmed_b = traced_corner_join(leg_a.left_curb, leg_b.right_curb)
            results[(name_a, name_b)] = {
                "trimmed_a": trimmed_a, "arc": arc, "trimmed_b": trimmed_b,
                "radius_ft": (corner_radii or {}).get(corner_key, radius_ft),
                "source": "traced_kerb",
            }
            continue

        traced = traced_corner_arc((corner_arcs or {}).get(corner_key, []),
                                    leg_a.left_curb, leg_b.right_curb)
        if traced is not None:
            try:
                trimmed_a = substring(leg_a.left_curb, leg_a.left_curb.project(Point(*traced.coords[0])),
                                       leg_a.left_curb.length)
                trimmed_b = substring(leg_b.right_curb, leg_b.right_curb.project(Point(*traced.coords[-1])),
                                       leg_b.right_curb.length)
                if not trimmed_a.is_empty and not trimmed_b.is_empty:
                    results[(name_a, name_b)] = {
                        "trimmed_a": trimmed_a, "arc": traced, "trimmed_b": trimmed_b,
                        "radius_ft": (corner_radii or {}).get(corner_key, radius_ft),
                        "source": "traced_kerb",
                    }
                    continue
            except (ValueError, IndexError):
                pass  # fall through to the fitted fillet below

        # A traced kerb gives this specific corner its own measured radius; anything
        # untraced falls back to the site-wide placeholder (see corner_radii_from_kerbs).
        this_radius = corner_radii.get(corner_key, radius_ft) if corner_radii else radius_ft
        try:
            trimmed_a, arc, trimmed_b = fillet_curb_corner(leg_a.left_curb, leg_b.right_curb, this_radius)
            results[(name_a, name_b)] = {"trimmed_a": trimmed_a, "arc": arc, "trimmed_b": trimmed_b,
                                          "radius_ft": this_radius}
        except ValueError as e:
            results[(name_a, name_b)] = {"error": str(e)}
    return results


# A leg whose bearing is within this of the reverse of another's is that leg's continuation
# across the junction, not a street crossing it. Broad St's two legs do not make each other's
# crosswalk longer; Greenwood's do. Same threshold THROUGH_STREET_ANGLE_DEG uses, and for the
# same reason - see _through_street.
CROSS_STREET_MAX_ANGLE_DEG = 150.0
# How far beyond the cross street's kerb line a crosswalk actually sits. MEASURED, not chosen:
# fitted against the 11 OSM-surveyed crossings at the four sites, which give a mean setback of
# 8.3 ft with a standard deviation of 2.4 ft (range 5.1-13.9). See
# tests/test_sites.py:test_the_crosswalk_estimate_reproduces_the_surveyed_crossings.
CROSSWALK_SETBACK_FT = 8.3


def crosswalk_estimate_ft(leg_name: str, legs: dict) -> float:
    """Where a crosswalk goes on a leg with no surveyed crossing to copy.

    The controlling dimension is the CROSS street's half-width, not this leg's own kerb: a
    crosswalk sits just outside the box the intersecting roadway occupies, and the corner
    return it also has to clear scales with that same roadway. Two other candidate rules were
    tried against the 11 surveyed crossings first and both failed - the fillet tangent point
    this replaces (leg_clearance_ft, which is what a crossing used to be placed on) scattered
    -31.5 to +41.7 ft, and projecting the cross street's kerb lines onto this leg's centerline
    scattered -38.0 to -2.3 ft and returned 119.7 ft for W Broad's northeast leg, where the
    near-parallel through street's kerbs meet it at a shallow angle far up the road.

    This rule reproduces all 11 to a standard deviation of 2.4 ft.

    The failure it fixes is not subtle. At W Broad & Louellen the fillet-tangent rule put the
    southwest leg's crossing 67.8 ft out - past the far kerb of the cross street, into the
    middle of the block - and the northeast leg's 11.5 ft out, inside a corner return whose
    kerb is still 25.4 ft off the centerline against a 17.6 ft half-width. One crossing too
    far out and its opposite too far in, at the same junction, from the same rule.
    """
    leg = legs[leg_name]
    bearing = _leg_bearing_deg(leg)
    widest_cross_half_ft = 0.0
    for other_name, other in legs.items():
        if other_name == leg_name:
            continue
        apart = abs(_leg_bearing_deg(other) - bearing) % 360
        apart = min(apart, 360 - apart)
        if apart > CROSS_STREET_MAX_ANGLE_DEG:
            continue        # this leg's own continuation across the junction
        widest_cross_half_ft = max(widest_cross_half_ft, other.curb_to_curb_ft / 2)
    return widest_cross_half_ft + CROSSWALK_SETBACK_FT


def _leg_bearing_deg(leg) -> float:
    """Compass bearing from the junction outward along a leg, from its chord.

    The chord and not the first segment: a bent leg's opening segment points somewhere the
    leg as a whole does not, which is the error that put a crosswalk 29.4 deg off square at
    Louellen (see crosswalk_axes).
    """
    (x0, y0), (x1, y1) = leg.centerline.coords[0], leg.centerline.coords[-1]
    return float(np.degrees(np.arctan2(x1 - x0, y1 - y0)) % 360)


def leg_clearance_ft(leg_name: str, legs: dict, corner_fillets: dict, buffer_ft: float = 3.0,
                     side: str | None = None) -> float:
    """
    Distance from a leg's near point out past BOTH of its corner fillets'
    tangent points, plus a small buffer - the point beyond which the leg's
    curb lines run straight rather than curving through the corner. Use this
    to place crosswalks / raised crossings outside the curve, not inside it -
    a fixed small offset from the intersection center lands inside the curve
    for any leg wide enough or with a generous enough corner radius.

    `side` narrows it to the corners that constrain THAT KERB. A corner return belongs to one
    side of each leg it touches - build_corner_fillets pairs leg A's LEFT curb with leg B's
    RIGHT curb - so a per-leg maximum holds one kerb back for a curve that is on the other.
    At E Broad & Princeton the stem runs south, so both corners constrain the south curbs and
    the north side of E Broad has no return on it at all; the per-leg figure held its kerbside
    paint 28-32 ft out from a curve that is not there:

        e_broad_st_east  left  (north)   per-side  3.0 ft   per-leg 32.1 ft
        e_broad_st_east  right (south)   per-side 32.1 ft   per-leg 32.1 ft

    Without `side` the answer is the per-leg maximum, which is what a CROSSWALK wants - it
    spans kerb to kerb, so it has to clear the returns on both sides. Only paint that belongs
    to one kerb should ask per side.
    """
    # Project onto the centerline (not raw Euclidean distance from the near
    # point) - the tangent point lives on the CURB line, laterally offset from
    # the centerline by half the leg's width, so a plain .distance() call
    # conflates that lateral offset with the actual along-the-road distance,
    # wildly overshooting for wide legs (a 68 ft leg has a 34 ft half-width,
    # which alone would dominate the distance even with zero along-leg offset).
    centerline = legs[leg_name].centerline
    max_along_dist = 0.0
    for (leg_a, leg_b), pieces in corner_fillets.items():
        if "error" in pieces or pieces.get("through_street"):
            # A through-street join is not a corner return: the curb does not curve there, so
            # it constrains nothing. Its "tangent points" are just wherever the two curb lines
            # happen to start, which on a partially-traced side is far up the leg.
            continue
        # trimmed_a is leg_a's LEFT curb, trimmed_b is leg_b's RIGHT curb - see
        # build_corner_fillets. That is what makes a corner side-specific.
        if leg_a == leg_name and side in (None, "left"):
            max_along_dist = max(max_along_dist, centerline.project(Point(pieces["trimmed_a"].coords[0])))
        if leg_b == leg_name and side in (None, "right"):
            max_along_dist = max(max_along_dist, centerline.project(Point(pieces["trimmed_b"].coords[0])))
    return max_along_dist + buffer_ft


# How finely a curbside strip is sampled along the leg. The curb is a traced polyline whose
# vertices fall wherever the surveyor clicked, so the strip is resampled on a regular station
# grid instead: both of its boundaries then have matching vertices at matching stations, and
# the strip stays a strip rather than a wedge.
STRIP_SAMPLE_FT = 2.0


@lru_cache(maxsize=256)
def _curb_in_leg_frame(centerline: LineString, curb: LineString):
    """A curb's own vertices as (stations, offsets) in the leg's frame, sorted by station.

    Cached on the two geometries, for the same reason and with the same safety as
    _polyline_frame: this is a fact about a pair of immutable lines, and everything that
    measures against a real kerb needs it. Uncached it was recomputed from scratch on every
    query - once per bollard, once per parking-stall divider, once per zone end line - each
    time re-projecting every traced vertex (a traced kerb can carry 30+) to answer about one
    station.

    Sorted here rather than by each caller, because np.interp requires it and two callers
    were each doing their own argsort of the same array.
    """
    stations, offsets = station_offset_many(centerline, np.asarray(curb.coords, dtype=float))
    order = np.argsort(stations)
    stations, offsets = stations[order], offsets[order]
    stations.flags.writeable = False
    offsets.flags.writeable = False
    return stations, offsets


def _traced_curb_frame(leg: "Leg", side: str):
    """(stations, offsets) for a side's real curb, or None where that side has none."""
    curb = getattr(leg, f"{side}_curb", None)
    if curb is None or curb.is_empty:
        return None
    return _curb_in_leg_frame(leg.centerline, curb)


def curb_offsets_at_stations(leg: "Leg", side: str, stations: np.ndarray) -> np.ndarray | None:
    """Signed offsets of a side's real curb at the given CENTERLINE stations.

    The vectorized form of curb_point_at_station's interpolation - see that function for why
    a curb cannot be addressed by its own arc length.
    """
    frame = _traced_curb_frame(leg, side)
    if frame is None:
        return None
    curb_stations, curb_offsets = frame
    return np.interp(stations, curb_stations, curb_offsets)


def curb_station_span(leg: "Leg", side: str) -> tuple[float, float] | None:
    """The stations along the leg that this side's curb was actually traced across, clipped
    to the leg itself.

    A traced kerb runs on its own terms: it starts 13-47 ft out (the corner return is traced
    separately) and several of them carry on 11-78 ft PAST the end of the 130 ft leg, because
    the tracing continues down the block. Paint has to be built inside that span - outside it
    there is no curb to measure from, only extrapolation.
    """
    frame = _traced_curb_frame(leg, side)
    if frame is None:
        return None
    stations, _offsets = frame
    lo = max(float(stations[0]), 0.0)      # sorted by _curb_in_leg_frame
    hi = min(float(stations[-1]), leg.centerline.length)
    return (lo, hi) if hi > lo else None


def narrowest_half_width_ft(leg: "Leg", side: str, from_ft: float = 0.0,
                             to_ft: float | None = None) -> float:
    """The least room this side has between the centerline and the real kerb, over a span.

    The bound a cross-section has to fit if it is to be promised for the whole of a leg rather
    than at one station. The nominal half-width is a summary and is routinely the wrong number
    for this: broad_st_east's nominal half is 26.0 ft, and its kerbs come within 22.8 ft of the
    alignment somewhere along the traced run. A 48 ft parking-protected section sized off the
    nominal figure would be drawn 5.2 ft over the kerb at that point.

    Falls back to the nominal half-width where the side has no traced kerb, since then there is
    no measurement to prefer.
    """
    half_ft = leg.curb_to_curb_ft / 2 if leg.curb_to_curb_ft is not None else 0.0
    span = curb_station_span(leg, side)
    if span is None:
        return half_ft
    lo = max(span[0], from_ft)
    hi = min(span[1], leg.centerline.length if to_ft is None else to_ft)
    if hi - lo < STRIP_SAMPLE_FT:
        return half_ft
    n = max(int(np.ceil((hi - lo) / STRIP_SAMPLE_FT)) + 1, 2)
    offsets = curb_offsets_at_stations(leg, side, np.linspace(lo, hi, n))
    return float(np.abs(offsets).min())


def curbside_strip_polygon(leg: "Leg", side: str, inner_offset_ft: float,
                            start_ft: float, end_ft: float | None = None) -> Polygon | None:
    """The strip of roadway between a leg's real curb and a line inner_offset_ft from its
    centerline, between two centerline stations.

    Both boundaries are sampled at the SAME stations, which is the whole point. The previous
    construction paired `substring(curb, start_ft, curb.length)` with
    `substring(inner, start_ft, inner.length)`, and those two substrings have nothing to do
    with each other: substring measures arc length along each line from that line's own
    start, so the curb was cut at a station 20-30 ft from where the inner line was cut, and
    the far ends differed by as much as 49 ft where the tracing ran past the leg. Closing
    that ring produced a wedge with two long diagonal ends instead of a strip, which is what
    fragmented the hatching and pushed paint outside the curb.

    Returns None where there is no room - the curb comes inside inner_offset_ft (paint would
    be outside the roadway) or the span is empty.
    """
    span = curb_station_span(leg, side)
    if span is None:
        return None
    lo, hi = span
    lo = max(lo, start_ft)
    hi = min(hi, leg.centerline.length if end_ft is None else end_ft)
    if hi - lo < STRIP_SAMPLE_FT:
        return None

    n = max(int(np.ceil((hi - lo) / STRIP_SAMPLE_FT)) + 1, 2)
    stations = np.linspace(lo, hi, n)
    curb_offsets = curb_offsets_at_stations(leg, side, stations)
    if curb_offsets is None:
        return None

    # The inner edge never crosses outside the real curb. On several sides the traced kerb
    # comes inside the nominal half-width (broad_st_east left is traced at 22.7 ft against a
    # nominal 24.2 ft), and taking the nominal figure on faith is how paint ended up over the
    # kerb. Where the curb is inside the lane edge the strip simply pinches to nothing.
    sign = 1.0 if side == "left" else -1.0
    inner = sign * np.minimum(inner_offset_ft, np.abs(curb_offsets))

    # The outer boundary is the kerb, so it uses the KERB'S OWN coordinates rather than a
    # resampling of them - see curb_edge_by_station for why that difference is load-bearing near
    # a centerline bend. The inner boundary has no such geometry to borrow and is placed in the
    # measuring frame instead.
    outer_pts = curb_edge_by_station(leg, side, lo, hi)
    if outer_pts is None:
        return None
    inner_pts = _place_no_further_in_than(leg.centerline, stations, inner)
    poly = Polygon(list(outer_pts) + list(reversed(inner_pts)))
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if (not poly.is_empty and poly.area > 1e-6) else None


def lane_narrowing_polygons_ft(leg: "Leg", stripe_width_ft: float,
                                start_left_ft: float = 0.0, start_right_ft: float = 0.0,
                                sides: tuple = ("left", "right"),
                                end_ft: float | None = None) -> list[Polygon]:
    """Two thin paint-only strips just inside each curb line - a visual lane
    narrowing treatment achieved with paint, NOT a curb_to_curb_ft change (no
    pavement/curb geometry is touched). Used by paint-only proposals - see
    src/geometry/treatments.py:add_lane_narrowing.

    start_left_ft/start_right_ft trim each strip to begin past the point
    where it tapers into the corner (see lane_narrowing_taper_ft) - a real
    painted lane line doesn't stop in a straight cut at the crosswalk/
    stop-bar clearance line, it tapers back out to meet the real curb closer
    to the intersection, so this needs to line up exactly with wherever that
    taper starts on each side (which can differ between the leg's left and
    right side - each is trimmed independently). Without this, the strip's
    straight, untrimmed curb/offset lines run all the way to the
    intersection's own center point, crossing straight through the open
    intersection box where no paint actually exists.

    sides restricts which curb(s) to build a strip for - e.g. a marked-
    parking buffer (src/geometry/treatments.py:add_marked_parking's
    curb_offset_ft) only ever needs one side of one leg, not the symmetric
    both-sides narrowing a travel lane gets."""
    half = leg.curb_to_curb_ft / 2
    inner_half = max(half - stripe_width_ft, 0.5)
    polys = []
    for start_ft, side in ((start_left_ft, "left"), (start_right_ft, "right")):
        if side not in sides:
            continue
        poly = curbside_strip_polygon(leg, side, inner_half, start_ft, end_ft)
        if poly is not None:
            polys.append(poly)
    return polys


def lane_narrowing_edge_lines_ft(leg: "Leg", stripe_width_ft: float,
                                  start_left_ft: float = 0.0, start_right_ft: float = 0.0,
                                  sides: tuple = ("left", "right"),
                                  keep_inside_ft: float = 0.0) -> list[LineString]:
    """The solid line marking the new, narrower travel lane's outer edge on
    each side - the same inner boundary lane_narrowing_polygons_ft's buffer
    zone starts from (11 ft from centerline for this site's proposals - see
    TARGET_LANE_WIDTH_FT in sites/broad_st_greenwood/scenarios.py) - drawn
    explicitly so the lane width actually reads on the render, rather than
    only being implied by wherever the diagonal hatching happens to start.
    start_left_ft/start_right_ft match lane_narrowing_polygons_ft's (see its
    docstring) so this line, the hatch fill, and the corner taper
    (lane_narrowing_taper_ft) all begin at the same point with no gap.
    sides - see lane_narrowing_polygons_ft's docstring."""
    half = leg.curb_to_curb_ft / 2
    inner_half = max(half - stripe_width_ft, 0.5)
    lines = []
    for start_ft, side in ((start_left_ft, "left"), (start_right_ft, "right")):
        if side not in sides:
            continue
        line = inset_line_ft(leg, side, inner_half, start_ft, keep_inside_ft=keep_inside_ft)
        if line is not None:
            lines.append(line)
    return lines


def inset_line_ft(leg: "Leg", side: str, offset_ft: float,
                   start_ft: float, end_ft: float | None = None,
                   keep_inside_ft: float = 0.0) -> LineString | None:
    """A line offset_ft from the centerline on one side, over the stations where that side's
    curb exists - the inner boundary of curbside_strip_polygon, drawn on its own.

    Built on the same station grid as the strip so the two cannot disagree, and clamped
    inside the real curb for the same reason. NOT `offset_curve(...).interpolate(d)`: an
    offset curve's arc length differs from the centerline's, so `d` there is not station `d`,
    which is what let the parking stall ticks drift along the leg.

    keep_inside_ft is how far short of the kerb the line must stop when it gets clamped
    there - half the painted stripe's width, so the stripe sits inside the road instead of
    straddling the kerb. Clamping the AXIS to the kerb hung half the paint over it wherever
    the road was narrower than the offset asked for.
    """
    span = curb_station_span(leg, side)
    if span is None:
        return None
    lo, hi = span
    lo = max(lo, start_ft)
    hi = min(hi, leg.centerline.length if end_ft is None else end_ft)
    if hi - lo < STRIP_SAMPLE_FT:
        return None
    n = max(int(np.ceil((hi - lo) / STRIP_SAMPLE_FT)) + 1, 2)
    stations = np.linspace(lo, hi, n)
    curb_offsets = curb_offsets_at_stations(leg, side, stations)
    sign = 1.0 if side == "left" else -1.0
    room = np.maximum(np.abs(curb_offsets) - keep_inside_ft, 0.0)
    inner = sign * np.minimum(offset_ft, room)
    # Same measured-frame placement curbside_strip_polygon uses, and for the same reason - this
    # line is the inner boundary of that strip and the two must not disagree about where it is.
    return LineString(_place_no_further_in_than(leg.centerline, stations, inner))


def offset_band_polygon(leg: "Leg", side: str, inner_offset_ft: float, outer_offset_ft: float,
                         start_ft: float, end_ft: float | None = None,
                         keep_inside_ft: float = 0.0) -> Polygon | None:
    """The strip of roadway between TWO lateral offsets from the centerline, on one side.

    For a marking whose own two boundaries are what define it - a bike lane's green surface
    sits between the lane's two edge stripes and is exactly as wide as the lane. Built on the
    same station grid and with the same kerb clamping inset_line_ft uses, so the band and the
    two stripes drawn at its edges cannot disagree about where those edges are.

    NOT the difference of two curbside_strip_polygons, which is what this replaced and which
    was subtly wrong: both of those are bounded by the stations where the TRACED KERB exists,
    so wherever the kerb is unmapped the inner strip still reached the nominal half-width while
    the outer one contributed nothing to subtract, and the leftover spilled past the marking's
    own outer edge - 6.6 ft past it on broad_st_west's right bike lane, onto asphalt that is not
    the lane. Nothing reported it, because the ground it spilled onto has no other paint on it
    to collide with and no traced kerb to be outside of.

    Returns None where there is no room or no span to draw over, like its siblings.
    """
    span = curb_station_span(leg, side)
    if span is None:
        return None
    lo, hi = span
    lo = max(lo, start_ft)
    hi = min(hi, leg.centerline.length if end_ft is None else end_ft)
    if hi - lo < STRIP_SAMPLE_FT:
        return None
    n = max(int(np.ceil((hi - lo) / STRIP_SAMPLE_FT)) + 1, 2)
    stations = np.linspace(lo, hi, n)
    curb_offsets = curb_offsets_at_stations(leg, side, stations)
    sign = 1.0 if side == "left" else -1.0
    room = (np.maximum(np.abs(curb_offsets) - keep_inside_ft, 0.0) if curb_offsets is not None
            else np.full(stations.shape, abs(leg.curb_to_curb_ft) / 2 - keep_inside_ft))
    inner = sign * np.minimum(inner_offset_ft, room)
    outer = sign * np.minimum(outer_offset_ft, room)
    inner_pts = _place_in_measured_frame(leg.centerline, stations, inner)
    outer_pts = _place_in_measured_frame(leg.centerline, stations, outer)
    band = Polygon(list(inner_pts) + list(reversed(list(outer_pts))))
    if not band.is_valid:
        band = band.buffer(0)
    return band if not band.is_empty and band.area > 0 else None


def _corner_bulge_normal(leg: "Leg", role: str) -> np.ndarray:
    """Unit normal pointing from a leg's curb toward where a real corner
    fillet's arc bulges - the same direction that role's own curb is already
    offset from centerline ('left' for the leg_a corner role, 'right' for
    leg_b - see build_corner_fillets), just continuing further outward.
    Confirmed empirically against this project's real corner arcs (a corner
    fillet's arc sits further from centerline than the straight curb it's
    replacing, on the same side, not the opposite one)."""
    c0, c1 = np.array(leg.centerline.coords[0]), np.array(leg.centerline.coords[1])
    u = _unit(c1 - c0)
    return np.array([-u[1], u[0]]) if role == "left" else np.array([u[1], -u[0]])



def curb_point_at_station(leg: "Leg", side: str, station_ft: float) -> np.ndarray | None:
    """The point on a leg's real curb at `station_ft` ALONG THE CENTERLINE.

    Not `curb.interpolate(station_ft)`. That measures distance along the curb line from the
    curb line's own start, which coincides with the centerline station only while the curb
    is a symmetric offset of the centerline starting at the junction. Since the curbs became
    traced kerbs neither holds - they start 14-47 ft out and run at their own bearing - and
    asking for station 40 landed anywhere from 51 to 86 ft down the leg.
    """
    offsets = curb_offsets_at_stations(leg, side, np.asarray([station_ft], dtype=float))
    if offsets is None:
        return None
    return np.asarray(_point_at(leg.centerline, station_ft, float(offsets[0])), dtype=float)


def inset_point_at_station(leg: "Leg", station_ft: float, offset_ft: float) -> np.ndarray:
    """A point offset laterally from the centerline at a given station - exactly, via the
    leg frame, rather than by interpolating along an offset_curve whose own arc length
    differs from the centerline's."""
    return np.asarray(_point_at(leg.centerline, station_ft, offset_ft), dtype=float)


def _taper_arc_points(leg: "Leg", role: str, sign: int, inner_half_ft: float,
                       anchor_ft: float, target_ft: float, n_points: int) -> list[tuple] | None:
    """The taper arc on ONE side of a leg, as a list of points, or None where there is none.

    Tangent to the straight inset line at anchor_ft and passing exactly through the real curb
    at target_ft. Tangent-at-one-point + passes-through-another-point + a common circle centre
    uniquely determines the radius - solved directly, not guessed or borrowed from elsewhere:
    for chord d = target - anchor and outward unit normal n, R = |d|^2 / (2 * dot(d, n)).

    Extracted because lane_narrowing_taper_ft and lane_narrowing_taper_polygons_ft each carried
    a verbatim copy of it - the LINE and the FILL either side of one seam, solved twice. Two
    copies of the arc that the fill's whole purpose is to sit inside is the drift this project
    keeps paying for; they cannot disagree now.

    Held out of the travel lane at the end, because the arc is solved in WORLD space while the
    lane edge it leaves from is a fixed offset in the LEG's frame. Those are the same line only
    while the centerline is straight. Once the alignment bends onto the carriageway
    (intersection._centre_legs_on_traced_kerbs) a leg that curves toward the paint lets the arc
    cut 0.16 ft inside the 11 ft mark just after it leaves the tangent - which is real paint in
    a real travel lane, and check_paint_stays_out_of_the_travel_lane duly caught it on
    w_broad_st_northeast. Clamping the offset is the same move inset_line_ft makes against the
    kerb: the arc keeps its shape everywhere it was already outside the line.
    """
    p1 = inset_point_at_station(leg, anchor_ft, sign * inner_half_ft)
    p2 = curb_point_at_station(leg, role, target_ft)
    if p2 is None:
        return None
    normal = _corner_bulge_normal(leg, role)
    d = p2 - p1
    denom = 2 * np.dot(d, normal)
    if abs(denom) < 1e-6:
        return None     # p2 already (near enough) on the tangent line - no taper needed
    radius_ft = np.dot(d, d) / denom
    center = p1 + radius_ft * normal
    a1 = np.arctan2(p1[1] - center[1], p1[0] - center[0])
    a2 = np.arctan2(p2[1] - center[1], p2[0] - center[0])
    delta = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
    angles = a1 + np.linspace(0, delta, n_points)
    arc = np.array([(center[0] + radius_ft * np.cos(t), center[1] + radius_ft * np.sin(t))
                    for t in angles])
    stations, offsets = station_offset_many(leg.centerline, arc)
    inside = np.abs(offsets) < inner_half_ft
    if not inside.any():
        return [tuple(p) for p in arc]
    offsets[inside] = sign * inner_half_ft
    return _place_in_measured_frame(leg.centerline, stations, offsets)


# A taper runs from the straight run's start INWARD to the curb. When target_ft is further out
# than anchor_ft there is no room between the corner return and the crosswalk for one, and
# solving the arc anyway sweeps it backwards - which is what mangled the hatching on Princeton
# Ave's north leg (anchor 27.5 ft, target 28.6 ft) while the south leg, whose target sits
# properly inside its anchor, looked fine.
def _taper_fits(anchor_ft: float, target_ft: float) -> bool:
    return target_ft < anchor_ft


def lane_narrowing_taper_ft(leg: "Leg", stripe_width_ft: float, anchor_ft: float, target_ft: float,
                             n_points: int = 16, sides: tuple = ("left", "right")) -> list[LineString]:
    """Tapers a lane-narrowing buffer's straight edge line, on both sides of
    the leg, from anchor_ft (the stop-bar/clearance point where the straight
    run ends) back out to meet the REAL curb at target_ft (a point safely
    clear of the crosswalk, closer to the intersection than anchor_ft) - a
    same-leg taper, like a parking lane curving back to the curb before an
    intersection, NOT a sweep around the intersection corner to the cross
    leg. A sweep like that was tried first and doesn't work: the cross leg's
    own crosswalk sits right at the corner by definition, so any curve
    reaching all the way to the cross leg's curb inevitably cuts through it
    - there's no radius that avoids that, because the destination itself is
    inside the excluded zone. Terminating on the SAME leg, before its OWN
    crosswalk, sidesteps the problem entirely.

    The taper is tangent to the straight inset line at anchor_ft (so it
    continues the buffer's edge with no visible seam - the very thing an
    independently-computed curve, e.g. built from build_corner_fillets'
    fillet math with an unrelated radius, got wrong) and passes exactly
    through the real curb at target_ft. Tangent-at-one-point + passes-
    through-another-point + a common circle center uniquely determines the
    radius - solved directly, not guessed or borrowed from elsewhere: for
    chord d = target - anchor and outward unit normal n, R = |d|^2 / (2 *
    dot(d, n)). (For this site this R lands within ~1 ft of the real corner's
    own 20 ft radius anyway, for what it's worth - not a coincidence, just
    two ways of describing similarly-scaled curves at the same corner.)"""
    inner_half = max(leg.curb_to_curb_ft / 2 - stripe_width_ft, 0.5)
    if not _taper_fits(anchor_ft, target_ft):
        return []
    tapers = []
    for sign, role in ((1, "left"), (-1, "right")):
        if role not in sides:
            continue
        arc = _taper_arc_points(leg, role, sign, inner_half, anchor_ft, target_ft, n_points)
        if arc is not None:
            tapers.append(LineString(arc))
    return tapers


def lane_narrowing_taper_polygons_ft(leg: "Leg", stripe_width_ft: float, anchor_ft: float, target_ft: float,
                                      n_points: int = 16, sides: tuple = ("left", "right")) -> list[Polygon]:
    """The paint-only buffer's fill zone WITHIN the taper itself - same real
    source photo this whole treatment is modeled on (see lane_narrowing_taper_ft's
    docstring/PR history) shows the diagonal chevron paint continuing in the
    same pattern all the way around the curve to the curb, not stopping dead
    where the straight run ends. Bounded by the taper arc (lane_narrowing_taper_ft,
    tangent to the straight inset line at anchor_ft, passing through the real
    curb at target_ft) on one side and the real curb itself, from target_ft
    back to anchor_ft, on the other - the same curb/inset pairing
    lane_narrowing_polygons_ft uses for the straight run, just curved instead
    of straight, so hatch_lines_ft can fill it with the identical pattern and
    the two zones read as one continuous stripe with no visible seam."""
    inner_half = max(leg.curb_to_curb_ft / 2 - stripe_width_ft, 0.5)
    if not _taper_fits(anchor_ft, target_ft):
        return []
    polys = []
    for sign, role in ((1, "left"), (-1, "right")):
        if role not in sides:
            continue
        # The SAME arc lane_narrowing_taper_ft draws as the line - see _taper_arc_points. The
        # fill's entire job is to sit inside that line, so solving it twice was asking for the
        # two to disagree.
        arc_pts = _taper_arc_points(leg, role, sign, inner_half, anchor_ft, target_ft, n_points)
        if arc_pts is None:
            continue    # no taper (see lane_narrowing_taper_ft) - nothing extra to fill
        # The curb run back from target_ft to anchor_ft, sampled by STATION. `substring(curb,
        # target_ft, anchor_ft)` was arc length along the traced kerb from the kerb's own
        # start - the same confusion curb_point_at_station exists to avoid - which put this
        # edge somewhere else entirely and left the taper fill 2.3 ft over the kerb on
        # broad_st_west. arc_pts already ends at p2 (the curb at target_ft), so that
        # duplicate is dropped; Polygon() closes the ring with the segment back to p1.
        n_curb = max(int(np.ceil((anchor_ft - target_ft) / STRIP_SAMPLE_FT)) + 1, 2)
        curb_stations = np.linspace(target_ft, anchor_ft, n_curb)
        curb_offsets = curb_offsets_at_stations(leg, role, curb_stations)
        curb_forward = [_point_at(leg.centerline, s, float(o))
                        for s, o in zip(curb_stations, curb_offsets)][1:]
        ring = arc_pts + curb_forward
        if len(ring) < 3:
            continue
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        # The arc is a circle solved through two points; between them it is free to bulge
        # past the kerb, which no amount of care about the endpoints prevents. Clipping it
        # to the roadway on this side is what actually guarantees the fill stays on the road.
        roadway = curbside_strip_polygon(leg, role, 0.0, target_ft, anchor_ft)
        if roadway is not None:
            poly = poly.intersection(roadway)
        for part in getattr(poly, "geoms", [poly]):
            if part.geom_type == "Polygon" and not part.is_empty and part.area > 1e-6:
                polys.append(part)
    return polys


def bollard_points_ft(leg: "Leg", stripe_width_ft: float, start_ft: float,
                       spacing_ft: float = 10.0, sides: tuple = ("left", "right")) -> list[tuple[float, float]]:
    """Points down the center of a paint-only buffer strip that's stripe_width_ft
    wide, next to the curb (same inner_half math as lane_narrowing_polygons_ft,
    so a bollard line always sits centered in the buffer that's actually
    painted, not a separately-guessed offset) - one line per requested side,
    starting start_ft along the centerline (past the corner fillet curve, same
    clearance convention as crosswalks/stop bars/trees - see leg_clearance_ft)
    and spaced spacing_ft apart to the end of the leg. Used by
    src/geometry/treatments.py:add_bollards (both sides, centered in a
    lane-narrowing buffer) and add_parking_buffer_bollards (one side,
    centered in the curb_offset_ft buffer between a marked-parking lane and
    the curb - same "centered in a strip" math either way, just a different
    strip)."""
    half = leg.curb_to_curb_ft / 2
    inner_half = max(half - stripe_width_ft, 0.5)
    points = []
    for side, sign in (("left", 1), ("right", -1)):
        if side not in sides:
            continue
        span = curb_station_span(leg, side)
        if span is None or span[1] < start_ft:
            continue
        # Every station at once. The kerb was previously read one bollard at a time, and each
        # read re-projected the whole traced kerb into the leg frame to answer about a single
        # station. Counted rather than accumulated with +=, so the last bollard's station is
        # start + n*spacing exactly instead of the sum of n additions.
        n = int((span[1] - start_ft) // spacing_ft) + 1
        stations = start_ft + np.arange(n) * spacing_ft
        curb_off = np.abs(curb_offsets_at_stations(leg, side, stations))
        # Centered between the strip's two real boundaries at EACH station, so a bollard sits
        # in the buffer that is actually painted even where the traced kerb comes inside the
        # nominal half-width (see curbside_strip_polygon).
        lateral = (curb_off + np.minimum(inner_half, curb_off)) / 2
        points.extend(tuple(_point_at(leg.centerline, float(s), sign * float(o)))
                      for s, o in zip(stations, lateral))
    return points


def points_at_offset_ft(leg: "Leg", side: str, offset_ft: float, start_ft: float,
                         end_ft: float | None = None, spacing_ft: float = 10.0) -> list[tuple]:
    """Points at a FIXED lateral offset from the centerline, spaced along the leg.

    For anything standing in a strip whose position is measured from the centerline rather than
    from the kerb - a delineator in a bike lane's buffer, say. bollard_points_ft centres its
    points between the kerb and a lane edge, which is the right rule for a kerbside buffer and
    the wrong one for a buffer sitting between two lanes: it would drift outward with the kerb
    instead of holding the line the paint holds.

    Clipped to the stations this side's kerb actually covers, so nothing is placed where there is
    no measured roadway, and never outside the kerb itself.
    """
    span = curb_station_span(leg, side)
    if span is None:
        return []
    lo = max(span[0], start_ft)
    hi = min(span[1], leg.centerline.length if end_ft is None else end_ft)
    if hi <= lo or spacing_ft <= 0:
        return []
    n = int((hi - lo) // spacing_ft) + 1
    stations = lo + np.arange(n) * spacing_ft
    curb_offsets = np.abs(curb_offsets_at_stations(leg, side, stations))
    sign = 1.0 if side == "left" else -1.0
    lateral = np.minimum(offset_ft, curb_offsets)
    return [tuple(_point_at(leg.centerline, float(s), sign * float(o)))
            for s, o in zip(stations, lateral)]


def parking_stall_count_ft(leg: "Leg", stall_length_ft: float, start_ft: float, end_ft: float | None = None) -> int:
    """How many full stall_length_ft stalls fit between start_ft and end_ft
    (defaults to the leg's own far end) - shared by parking_stall_lines_ft
    (which places the actual divider lines) and any caller that just wants
    the count for a label/note, so the two can never disagree."""
    end_ft = leg.centerline.length if end_ft is None else end_ft
    return max(int((end_ft - start_ft) // stall_length_ft), 0)


def parking_lane_edge_line_ft(leg: "Leg", side: str, depth_ft: float, start_ft: float,
                               end_ft: float | None = None, curb_offset_ft: float = 0.0) -> LineString | None:
    """The line marking the inner edge of a curbside marked-parking lane -
    depth_ft in from the curb (or from curb_offset_ft in from the curb, if
    the parking lane doesn't hug the curb directly - see below) on the given
    side, same start/end convention (past the corner fillet curve, see
    leg_clearance_ft) as lane_narrowing_edge_lines_ft. Real curbside parking
    doesn't always have this line painted (sometimes it's just the
    perpendicular stall ticks - see parking_stall_lines_ft), but drawing it
    makes the lane's real depth read clearly on both the plan view and the
    3D render, the same reasoning lane_narrowing_edge_lines_ft's docstring
    gives for its own edge line.

    curb_offset_ft > 0 pulls the whole parking lane in from the curb by that
    much (see src/geometry/treatments.py:add_marked_parking) - e.g. a striped
    no-parking buffer between the parking lane and the curb itself, so
    parking sits directly against the active travel lane instead of against
    the curb. Defaults to 0 (the lane starts right at the curb, as before)."""
    half = leg.curb_to_curb_ft / 2
    inner_half = max(half - curb_offset_ft - depth_ft, 0.5)
    # No room left on this leg to mark parking. Happens where the corner return eats the
    # whole leg: at W Broad & Louellen's acute Y, leg_clearance_ft comes out at 133 ft on a
    # 130 ft leg, so parking would start past the end of the road. Callers that interpolate
    # along an empty geometry fail with an unhelpful shapely type error, so inset_line_ft
    # says "nothing here" explicitly instead.
    return inset_line_ft(leg, side, inner_half, start_ft, end_ft)


def parking_stall_lines_ft(leg: "Leg", side: str, depth_ft: float, stall_length_ft: float, start_ft: float,
                            end_ft: float | None = None, curb_offset_ft: float = 0.0) -> list[LineString]:
    """Perpendicular divider lines bounding each marked parallel-parking
    stall along one side of a leg - the standard MUTCD curbside-parking
    marking: a short tie line at each stall boundary, not a filled/hatched
    zone (this is a real parking lane, not a paint-only buffer like
    lane_narrowing/corner_hatching - a driver is meant to park a real vehicle
    inside each one). One extra divider beyond the last full stall closes it
    off, so n stalls always get n+1 lines (see parking_stall_count_ft for the
    same n used elsewhere, e.g. a dimension label).

    Each divider normally runs from the real curb to depth_ft in from it;
    curb_offset_ft > 0 (see parking_lane_edge_line_ft) shifts BOTH ends in by
    that much instead, so the divider spans the parking lane itself, not the
    no-parking buffer between it and the curb."""
    half = leg.curb_to_curb_ft / 2
    sign = 1 if side == "left" else -1
    outer_off = max(half - curb_offset_ft, 0.5)
    inner_off = max(half - curb_offset_ft - depth_ft, 0.5)
    span = curb_station_span(leg, side)
    if span is None:
        return []
    end_ft = min(span[1], leg.centerline.length if end_ft is None else end_ft)
    n_stalls = parking_stall_count_ft(leg, stall_length_ft, start_ft, end_ft)
    # Station, not distance along an offset curve - see inset_line_ft. A divider is a
    # cross-section of the parking lane, so both ends have to be at the same station. All the
    # stations are read from the kerb in one pass rather than one per divider.
    stations = start_ft + np.arange(n_stalls + 1) * stall_length_ft
    curb_off = np.abs(curb_offsets_at_stations(leg, side, stations))
    return [LineString([
                _point_at(leg.centerline, float(station), sign * min(outer_off, float(off))),
                _point_at(leg.centerline, float(station), sign * min(inner_off, float(off))),
            ])
            for station, off in zip(stations, curb_off)]


# ---------------------------------------------------------------------------
# Curb extensions (bulb-outs)
# ---------------------------------------------------------------------------
#
# A curb extension shortens a crossing by moving the KERB LINE laterally into the roadway
# near the junction and tapering it back out. That is the whole mechanism, and it is worth
# stating because the obvious-looking alternative does nothing: re-cutting the corner ARC at a
# smaller radius (set_corner_radius, formerly and misleadingly called bump_out) leaves both
# curb lines exactly where they were. Measured on broad_st_east x greenwood_ave_north at
# 29.2 -> 15.0 ft:
#
#     arc length     19.48 -> 3.51 ft     the arc really is re-cut
#     trimmed_a     156.19 -> 164.19 ft   the curb just extends to the new tangent point
#     pavement area  23,989.7 -> 23,989.5 sq ft      0.2 sq ft of 24,000
#     crossing spans unchanged to 0.00 ft on all four legs
#
# The crossings here sit 21-42 ft out, past the corner, so a radius change never reaches them.
#
# The extension is measured from the leg's NOMINAL half-width, not from the traced kerb at that
# station, and the difference matters. The traced kerb flares through the corner return -
# broad_st_east's kerbs are 39.4 and 31.6 ft off the centerline where its crossing is painted,
# against a 26.0 ft nominal half-width - so the crossing today spans 65.0 ft of pavement, not
# the 52.0 ft the cross-section suggests. Extending from the nominal half-width replaces that
# flare with the extension's own straight face, which is what a built bulb-out does, and it is
# why the crossing falls further than the extension alone would imply: 8 ft of extension per
# side takes broad_st_east from 65.0 ft to about 2 x (26.0 - 8) = 36 ft.
#
# How far a curb extension may be pushed is bounded by the travel lane it must leave behind,
# so every caller is checked against TARGET_LANE_WIDTH_FT - see
# src/geometry/treatments.py:add_curb_extension.

# How gently the extension returns to the real kerb: feet along the leg per foot of lateral
# shift. A DESIGN CHOICE, not a measured or standard figure - flagged like
# PARKING_BUFFER_DEFAULT_FT rather than dressed up as a citation. 5:1 is at the gentle end of
# what low-speed parking-lane transitions use, and the check that matters is not the rate but
# the total: face plus taper has to stay inside the length of kerb where parking is already
# prohibited, or the extension removes a space. See
# tests/test_curb_extensions.py:test_a_bulbout_fits_inside_the_ordinance_no_parking_length.
BULBOUT_TAPER_RATE = 5.0


def curb_extension_line(leg: "Leg", side: str, extension_ft: float, full_ft: float,
                         taper_ft: float) -> LineString | None:
    """One leg side's kerb with a curb extension built into it.

    Three stretches, in station order:

      0 -> full_ft                  the extension's face, straight, at the leg's nominal
                                    half-width less `extension_ft`
      full_ft -> full_ft + taper_ft the return to the real kerb
      beyond                        the traced kerb itself, vertex for vertex

    The taper is a raised-cosine blend between the two offsets, which is tangent to both ends
    by construction - no kink where the face meets it and none where it rejoins the tracing.
    (An arc solved through two points, which lane_narrowing_taper_ft uses for painted tapers,
    is free to bulge between them; for a KERB that bulge would be built concrete.)

    The face never sits outside the traced kerb. Where the real kerb is already inside the
    nominal half-width - which happens mid-block, broad_st_east's left kerb is traced at
    22.7 ft against a 24.2 ft nominal - the tracing wins and no extension is built there. An
    extension is only ever allowed to take roadway, never to invent it.
    """
    frame = _traced_curb_frame(leg, side)
    if frame is None or leg.curb_to_curb_ft is None:
        return None
    curb_stations, curb_offsets = frame
    sign = 1.0 if side == "left" else -1.0
    face_abs = leg.curb_to_curb_ft / 2 - extension_ft
    taper_end_ft = full_ft + taper_ft

    n = max(int(np.ceil(taper_end_ft / STRIP_SAMPLE_FT)) + 1, 2)
    stations = np.linspace(0.0, taper_end_ft, n)
    real_abs = np.abs(np.interp(stations, curb_stations, curb_offsets))
    # Raised cosine over the taper, 0 on the face, 1 once the real kerb governs again.
    ease = (1 - np.cos(np.pi * np.clip((stations - full_ft) / taper_ft, 0.0, 1.0))) / 2
    built_abs = np.minimum(face_abs, real_abs) * (1 - ease) + real_abs * ease

    points = _place_in_measured_frame(leg.centerline, stations, sign * built_abs)
    # The tracing itself past the taper, not a resampling of it: beyond the extension this
    # side's kerb is still the surveyor's, and it should stay vertex-for-vertex theirs.
    points += [_point_at(leg.centerline, float(s), float(o))
               for s, o in zip(curb_stations, curb_offsets) if s > taper_end_ft]
    return LineString(points) if len(points) >= 2 else None


# How many corrective passes _place_in_measured_frame takes. Two is enough at every leg here -
# the residual falls from 0.59 ft to under a thousandth - and a cap means a pathological frame
# ends the loop rather than spinning in it.
#
# Raising it is not the way to chase a stubborn residual, which is worth stating because it
# looks like the obvious lever and it is not: each pass re-asks at a corrected (station,
# offset) and keeps whichever attempt has the smallest COMBINED station-and-offset error, so
# another pass can legitimately trade station accuracy for offset accuracy and land somewhere
# else entirely. Going 2 -> 3 moved enough geometry to fail 18 tests across four junctions
# while still not fixing the fold that prompted it. Where one line must not drift a particular
# way, bias that line - see inset_line_ft.
_FRAME_CORRECTION_PASSES = 2


def _place_in_measured_frame(centerline: LineString, stations: np.ndarray,
                             offsets: np.ndarray) -> list[tuple]:
    """World points that MEASURE BACK as (station, offset), not merely that were built from it.

    _point_at and station_offset_many are inverses along a straight centerline and drift apart
    near a kink, because they resolve the ambiguity differently: _point_at extrapolates the frame
    of the segment the station falls on, while station_offset_many assigns a point to whichever
    segment it is perpendicular-nearest to. In the wedge outside a bend those are different
    segments, and the further from the centerline the wider the gap.

    It matters here and not for a traced kerb because a traced kerb's stations were DERIVED by
    station_offset_many from surveyed points, so they agree with it by construction. An extension
    imposes stations instead, at ±19 ft of offset, and broad_st_east's centerline kinks 4.5 deg
    43.1 ft out where NJDOT rounds the corner: a vertex placed at station 44.0 read back at
    41.59, and across the taper's 0.2 ft-per-ft slope that is 0.59 ft of offset - enough to put
    the kerbside hatching built against this kerb 0.6 ft over it, which check_paint_inside_the_curb
    duly caught.

    So the placement is corrected against the measuring frame rather than trusted: place, measure,
    move by the residual. Everything downstream - the paint, the crossing reach, the invariants -
    measures with station_offset_many, so that is the frame the geometry has to be right in.

    Corrected per point and only where it HELPS. The two frames do not merely drift: at an offset
    larger than the bend's radius of curvature the offset curve FOLDS, and inside the fold the
    station order reverses - on broad_st_east's right kerb, 19 ft in from the bend at station
    43.1, asking for station 42 lands at 44.35 and asking for 44 lands at 41.59. A correction step
    across that discontinuity overshoots instead of converging, so each point keeps whichever
    estimate measures closest to what was asked and the fold is left alone rather than chased.
    """
    target_s = np.asarray(stations, dtype=float)
    target_o = np.asarray(offsets, dtype=float)
    ask_s, ask_o = target_s.copy(), target_o.copy()
    best = np.array([_point_at(centerline, float(s), float(o)) for s, o in zip(ask_s, ask_o)])
    got_s, got_o = station_offset_many(centerline, best)
    best_error = np.hypot(got_s - target_s, got_o - target_o)
    for _ in range(_FRAME_CORRECTION_PASSES):
        ask_s = ask_s + (target_s - got_s)
        ask_o = ask_o + (target_o - got_o)
        trial = np.array([_point_at(centerline, float(s), float(o))
                          for s, o in zip(ask_s, ask_o)])
        got_s, got_o = station_offset_many(centerline, trial)
        error = np.hypot(got_s - target_s, got_o - target_o)
        better = error < best_error
        best[better], best_error[better] = trial[better], error[better]
    return [tuple(p) for p in best]


# How many times the outward bias below re-asks. One pass closes most folds; broad_st_west's
# is deep enough at a 2.0x frame that a single correction still left 0.076 ft of the parking
# edge line inside the travel lane, and the residual only showed at that ONE frame scale -
# 1.0, 2.2, 2.5 and 3.0 were all clean. Each pass shrinks what is left, and the loop stops as
# soon as nothing is short, so this costs nothing where there is no fold.
_OUTWARD_BIAS_PASSES = 4


def _place_no_further_in_than(centerline: LineString, stations: np.ndarray,
                               offsets: np.ndarray) -> list[tuple]:
    """_place_in_measured_frame, biased so no point lands INSIDE the offset it was asked for.

    A kerbside marking's two possible placement errors are not equivalent. This offset is the
    edge of a travel lane: landing a hair wide of it costs a hair of kerbside treatment, and
    landing a hair narrow puts paint in the lane, which is what PaintStaysOutOfTheTravelLane
    exists to catch. Inside a frame fold - broad_st_east's 7.2 degree kink 43 ft out - the
    placement settles 0.05 ft short, and 0.05 ft short is a reported violation.

    Only the points that fell short move. Re-placing the whole line instead shifts every OTHER
    point too, because _place_in_measured_frame searches from the ask and a changed ask
    anywhere reshuffles the lot: that moved enough geometry across all four junctions to fail
    18 tests, in service of 0.05 ft on one vertex of one leg.

    Used by curbside_strip_polygon AND inset_line_ft, which is not optional - the line IS the
    strip's inner boundary, and biasing one without the other breaks the property inset_line_ft
    exists to hold. Biased on its own it put the rim of a hatched zone 1.5 ft alongside the
    edge line it continues, far enough off to stop reading as the same stroke and near enough
    for MarkingsDoNotCollide to call it two.
    """
    offsets = np.asarray(offsets, dtype=float)
    placed = np.asarray(_place_in_measured_frame(centerline, stations, offsets), dtype=float)
    ask = offsets.copy()
    for _ in range(_OUTWARD_BIAS_PASSES):
        _stations, got = station_offset_many(centerline, placed)
        short = np.maximum(np.abs(offsets) - np.abs(got), 0.0)
        if not short.any():
            break
        # Eased into the neighbouring vertices at half height rather than applied to the short
        # one alone. A single vertex pushed out of line with the two either side of it is a
        # kink, and a kink in a line that is then clipped around crossings and driveways comes
        # back as overlapping fragments: a 1.5 ft offcut of w_broad_st_southwest's buffer edge
        # line lying on top of the 125 ft one it was cut from, which MarkingsDoNotCollide reads
        # - correctly - as two lines painted down the same stretch of road.
        padded = np.pad(short, 1, mode="edge")
        short = np.maximum(short, 0.5 * np.maximum(padded[:-2], padded[2:]))
        ask = ask + np.sign(offsets) * short
        nudged = np.asarray(_place_in_measured_frame(centerline, stations, ask), dtype=float)
        moved = short > 0
        placed[moved] = nudged[moved]
    return [tuple(p) for p in placed]


def curb_edge_by_station(leg: "Leg", side: str, lo_ft: float, hi_ft: float) -> list[tuple] | None:
    """The kerb's OWN world coordinates between two stations, with exact ends.

    For the outer boundary of anything that runs along a kerb. Resampling the kerb onto a station
    grid and re-placing it with _point_at was near enough while every kerb offset changed slowly,
    and stopped being so once a curb extension's taper made one change at 0.2 ft per ft: the
    placement drifts from the frame the checks measure in, and inside a fold (see
    _place_in_measured_frame) it cannot be corrected at all.

    Taking the kerb's real coordinates sidesteps the frame entirely. Nothing is interpolated
    except the two end vertices, which are held at exactly lo_ft and hi_ft so the strip's two
    boundaries still start and finish at the same stations - the property the resampling existed
    to guarantee, and the one that keeps a strip a strip rather than a wedge.
    """
    frame = _traced_curb_frame(leg, side)
    if frame is None:
        return None
    curb_stations, curb_offsets = frame
    coords = np.asarray(getattr(leg, f"{side}_curb").coords, dtype=float)
    order = np.argsort(np.asarray(_traced_curb_station_order(leg, side)))
    inside = [tuple(coords[order[i]]) for i in range(len(order))
              if lo_ft < curb_stations[i] < hi_ft]
    ends = _place_in_measured_frame(leg.centerline, np.array([lo_ft, hi_ft]),
                                    np.interp([lo_ft, hi_ft], curb_stations, curb_offsets))
    return [ends[0], *inside, ends[1]]


def _traced_curb_station_order(leg: "Leg", side: str) -> np.ndarray:
    """The kerb's vertex indices in station order - the same sort _curb_in_leg_frame applies."""
    curb = getattr(leg, f"{side}_curb")
    stations, _offsets = station_offset_many(leg.centerline, np.asarray(curb.coords, dtype=float))
    return np.argsort(stations)


def corner_apron_annulus(curb_a: LineString, curb_b: LineString, face_radius_ft: float,
                          swept_radius_ft: float, n_points: int = 24) -> Polygon | None:
    """The mountable ground between a tightened corner face and the radius a bus still needs.

    A curb extension presents a `face_radius_ft` corner to a passenger car. A bus tracking the
    same corner needs the radius the corner was BUILT to, which at these junctions is a traced,
    measured figure per corner (29.2 / 24.6 / 29.0 / 22.9 ft at Broad & Greenwood). The ground
    between the two arcs is the difference: paved and flush, so a bus rides over it, but read
    by a driver as corner rather than carriageway.

    That region is what makes the "swept path is preserved by construction" claim true rather
    than asserted, so it is built as the actual annulus between the two arcs - both solved by
    the same fillet math off the same two curb lines. corner_overlay_polygon, which the
    standalone add_mountable_apron uses, draws a fixed-depth kite off one arc instead; it is
    the right shape for "hatch this corner" and the wrong one for "a bus fits through here",
    because nothing ties its depth to the radius a bus needs.

    Returns None where there is nothing to pave: a face radius at or above the swept radius
    means the corner was not tightened.
    """
    if swept_radius_ft <= face_radius_ft:
        return None
    try:
        _a, face_arc, _b = fillet_curb_corner(curb_a, curb_b, face_radius_ft, n_points)
        _a, swept_arc, _b = fillet_curb_corner(curb_a, curb_b, swept_radius_ft, n_points)
    except (ValueError, np.linalg.LinAlgError):
        return None
    ring = list(face_arc.coords) + list(reversed(swept_arc.coords))
    if len(ring) < 3:
        return None
    polygon = Polygon(ring)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area <= 1e-6:
        return None
    return polygon


def corner_overlay_polygon(pieces: dict, center_ft: Point, depth_ft: float) -> Polygon:
    """A 'virtual bump-out' zone hugging a corner's fillet arc, extending
    depth_ft inward toward the intersection center - flush with the pavement,
    no elevation/curb change. Shared shape for two different render
    treatments: diagonal paint hatching (src/geometry/treatments.py:add_corner_hatching)
    and a textured mountable apron (add_mountable_apron) - same footprint,
    different surface finish.

    A clean 4-point kite (arc start -> arc mid -> arc end -> inner point), NOT
    every point along the arc: using all ~24 arc vertices here produced a
    self-intersecting ring for some corners (GEOS then rejected it) and, once
    patched, a jagged boundary that fragmented any hatch line clipped against
    it into many small pieces - a visibly "tessellated" paint pattern for no
    benefit, since 3 points already approximate this size of curve smoothly
    enough for a paint-only overlay."""
    arc = pieces["arc"]
    start, mid, end = (arc.interpolate(t, normalized=True) for t in (0.0, 0.5, 1.0))
    inward = np.array([center_ft.x - mid.x, center_ft.y - mid.y])
    norm = np.linalg.norm(inward)
    inward = inward / norm if norm > 1e-6 else np.array([0.0, 0.0])
    inner_pt = (mid.x + inward[0] * depth_ft, mid.y + inward[1] * depth_ft)
    return Polygon([start.coords[0], mid.coords[0], end.coords[0], inner_pt])


# A hatch stroke shorter than this is a clipping artifact, not paint. They appear where a
# stroke grazes a corner of the polygon or crosses the needle-thin tip of a taper, and they
# render as stubs - the "sheared in half" strokes. One came out 0.0 ft long.
MIN_HATCH_STROKE_FT = 1.0


def clip_paint_clear_of(geometry, keep_clear):
    """Cut `keep_clear` out of a piece of paint, returning the surviving pieces.

    Road markings are layered by priority, and a crosswalk outranks a buffer or a parking
    lane - export.py has said so in a comment since long before anything enforced it. Doing
    the subtraction on the GEOMETRY is what makes it true, rather than relying on the paint's
    start station being far enough out: a skewed crossing reaches further along one kerb than
    its centre offset suggests, which is how two hatch strokes ended up over Broad St's
    crossing while the arithmetic said they cleared it.
    """
    if keep_clear is None or keep_clear.is_empty:
        return [geometry]
    remainder = geometry.difference(keep_clear)
    if remainder.is_empty:
        return []
    parts = getattr(remainder, "geoms", [remainder])
    return [g for g in parts if g.geom_type == geometry.geom_type and not g.is_empty]


def hatch_lines_ft(polygon: Polygon, spacing_ft: float = 2.0, angle_deg: float = 45.0,
                    phase_origin: tuple[float, float] = (0.0, 0.0)) -> list[LineString]:
    """Diagonal hatch lines filling a polygon, clipped to its boundary - used
    to render paint-only diagonal/chevron marking (e.g. corner_hatching_polygon
    above) without any real curb/pavement geometry change.

    phase_origin fixes WHERE the family of parallel lines falls, in world coordinates. It
    matters because a buffer is not one polygon: the straight run, the taper into the
    corner, and whatever survives being cut around a crossing are separate polygons hatched
    separately. Phasing each family off its own bounding box centre - which is what this did
    - gave each piece an independent stroke position, so at every seam the strokes stepped
    sideways by some fraction of the spacing. Reading across the seam, one stroke looked
    sheared into two offset halves. Passing all the pieces of one treatment the same origin
    puts them on one continuous set of lines, and the seams disappear.
    """
    # A corner-hatch polygon built off a traced kerb can pinch to a point where the curb
    # doubles back on itself, which is a bowtie GEOS refuses to intersect against. buffer(0)
    # resolves it into the same area without moving any edge; an empty result means the
    # polygon had no area to hatch in the first place.
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
        if polygon.is_empty:
            return []

    minx, miny, maxx, maxy = polygon.bounds
    diag = ((maxx - minx) ** 2 + (maxy - miny) ** 2) ** 0.5
    theta = np.radians(angle_deg)
    u = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-u[1], u[0]])

    # Which lines of the (infinite, origin-anchored) family actually reach this polygon:
    # the range of the corners' distances along n, snapped outward to whole multiples of the
    # spacing. Anchoring on multiples of the spacing from a shared origin is what keeps
    # neighbouring pieces in phase.
    origin = np.asarray(phase_origin, dtype=float)
    corners = np.array([[minx, miny], [minx, maxy], [maxx, miny], [maxx, maxy]]) - origin
    along_n, along_u = corners @ n, corners @ u
    steps = np.arange(np.floor(along_n.min() / spacing_ft), np.ceil(along_n.max() / spacing_ft) + 1)

    # Every hatch line at once, and one clip against the polygon instead of one GEOS call
    # per line. Endpoints are built with numpy broadcasting; the whole family goes through
    # a single MultiLineString intersection. Each line must span the polygon's extent ALONG
    # u as well as sit at the right distance along n - the phase origin is the state-plane
    # origin, half a million feet away, so a segment merely centred on it never reaches.
    centers = origin + n * (steps * spacing_ft)[:, None]
    lo, hi = along_u.min() - diag, along_u.max() + diag
    ends = np.stack([centers + u * lo, centers + u * hi], axis=1)
    clipped = MultiLineString([tuple(map(tuple, pair)) for pair in ends]).intersection(polygon)

    if clipped.is_empty:
        return []
    pieces = clipped.geoms if hasattr(clipped, "geoms") else [clipped]
    return [g for g in pieces
            if g.geom_type == "LineString" and g.length >= MIN_HATCH_STROKE_FT]


def build_pavement_polygon(corner_fillets: dict) -> Polygon:
    """
    Stitch every corner's (trimmed curb, arc, trimmed curb) into one continuous
    ring: the full paved footprint of the intersection, rounded corners and all.
    Requires build_corner_fillets() to have succeeded for every corner (a full
    cycle - each leg's left curb feeds one corner, its right curb the next).
    """
    if any("error" in pieces for pieces in corner_fillets.values()):
        raise ValueError("Can't build a pavement polygon - at least one corner fillet failed.")

    order = []
    remaining = dict(corner_fillets)
    name_a0, name_b0 = next(iter(remaining))
    order.append(name_a0)
    current = name_b0
    while current != name_a0:
        order.append(current)
        next_pair = next(pair for pair in remaining if pair[0] == current)
        current = next_pair[1]

    n = len(order)
    ring: list[tuple[float, float]] = []
    for i in range(n):
        leg_a, leg_b = order[i - 1], order[i]
        leg_c = order[(i + 1) % n]
        trimmed_b = corner_fillets[(leg_a, leg_b)]["trimmed_b"]   # leg_b's right curb, t2 -> far
        trimmed_a_next = corner_fillets[(leg_b, leg_c)]["trimmed_a"]  # leg_b's left curb, t1 -> far
        arc_next = corner_fillets[(leg_b, leg_c)]["arc"]

        ring.extend(trimmed_b.coords)
        ring.extend(reversed(list(trimmed_a_next.coords)))
        ring.extend(list(arc_next.coords)[1:-1])

    polygon = Polygon(ring)
    if not polygon.is_valid:
        raise ValueError(
            "Pavement ring is self-intersecting: "
            f"{explain_validity(polygon)}. {_acute_corner_diagnosis(corner_fillets, order)}"
        )
    return polygon


def _acute_corner_diagnosis(corner_fillets: dict, order: list[str]) -> str:
    """Explain a self-intersecting pavement ring in terms of the legs that caused it.

    The corner-fillet model assumes each pair of angularly-adjacent legs meets at a
    distinct, roundable corner - which requires the two roads' pavement envelopes to
    be separate everywhere outside that corner. At a sharply acute junction (a Y, a
    skewed fork) that fails: two wide roads diverging at a narrow angle overlap near
    the junction, forming one continuous paved throat/gore rather than two roads with
    a corner between them. The ring then folds through itself, and no corner radius
    fixes it - the overlap is a function of the legs' widths and the angle only.

    W Broad St & Louellen St is the worked example: W Broad southwest (50 ft) and
    Louellen west (34 ft) diverge at 43.6 degrees, so their curb envelopes overlap
    within ~56 ft of the junction.
    """
    culprits = []
    for i, name_a in enumerate(order):
        name_b = order[(i + 1) % len(order)]
        pieces = corner_fillets.get((name_a, name_b))
        if pieces is None or "error" in pieces:
            continue
        # The fillet arc bulges toward the corner vertex; an acute corner is the one
        # whose trimmed curbs run far past where the opposite leg's curb already is.
        if pieces["trimmed_a"].intersects(pieces["trimmed_b"]):
            culprits.append(f"{name_a}/{name_b}")
    detail = (
        f" The curb lines of {' and '.join(culprits)} cross each other."
        if culprits else ""
    )
    return (
        "This usually means two legs meet at too acute an angle for their widths, so "
        "their pavement envelopes overlap and the intersection is really one merged "
        f"throat rather than a set of separate rounded corners.{detail} Check the "
        "legs' bearing_deg and curb_to_curb_ft in the site config; if the geometry is "
        "right, this junction shape is not representable by the corner-fillet model."
    )


# Where along a leg to probe for the flanking sidewalks. Far enough out to be clear of
# the corner returns (which curve the sidewalk in toward the crossing) but still within
# a typical leg_working_length_ft.
SIDEWALK_PROBE_DISTANCES_FT = (40.0, 60.0, 80.0)
SIDEWALK_PROBE_REACH_FT = 120.0


def sidewalk_span_ft(centerline: LineString, sidewalk_lines: list[LineString],
                      distances_ft=SIDEWALK_PROBE_DISTANCES_FT) -> dict | None:
    """Distance from a leg's centerline out to the sidewalk on each side.

    Casts a perpendicular ray both ways at each probe distance and takes the nearest
    sidewalk hit, then medians across probes so one gap in the sidewalk network (or one
    driveway apron mapped as a footway) can't skew the answer. Returns
    {"left_ft", "right_ft", "span_ft", "probes"} or None if either side never hit
    anything - a leg with sidewalk mapped on only one side gives no usable span.

    `span_ft` is sidewalk-centerline to sidewalk-centerline. It is an UPPER BOUND on
    curb-to-curb, never the width itself: the curb is somewhere inside it, by a verge
    that varies a lot in practice (11.8 ft/side vs 4.0 ft/side on the two field-measured
    legs in this project). See src/sources/osm_context.py:fetch_sidewalks.
    """
    left, right = [], []
    for dist in distances_ft:
        if dist >= centerline.length:
            continue
        point = centerline.interpolate(dist)
        ahead = centerline.interpolate(min(dist + 5, centerline.length))
        vx, vy = ahead.x - point.x, ahead.y - point.y
        norm = np.hypot(vx, vy)
        if norm == 0:
            continue
        px, py = -vy / norm, vx / norm
        for sign, bucket in ((1, left), (-1, right)):
            ray = LineString([
                (point.x, point.y),
                (point.x + sign * SIDEWALK_PROBE_REACH_FT * px, point.y + sign * SIDEWALK_PROBE_REACH_FT * py),
            ])
            nearest = None
            for walk in sidewalk_lines:
                hit = ray.intersection(walk)
                if hit.is_empty:
                    continue
                points = [hit] if hit.geom_type == "Point" else list(getattr(hit, "geoms", []))
                for candidate in points:
                    if candidate.geom_type != "Point":
                        continue
                    d = point.distance(candidate)
                    if nearest is None or d < nearest:
                        nearest = d
            if nearest is not None:
                bucket.append(nearest)

    if not left or not right:
        return None
    left_ft, right_ft = float(np.median(left)), float(np.median(right))
    return {"left_ft": left_ft, "right_ft": right_ft, "span_ft": left_ft + right_ft,
            "probes": min(len(left), len(right))}


# Gates on a circle fitted to a traced kerb, before its radius is trusted as a corner
# radius. A short arc barely constrains a circle: 8 ft of kerb spanning 36 degrees
# happened to fit well at Columbia & Princeton, but only because the tracing was careful.
MIN_KERB_ARC_SWEEP_DEG = 25.0
MAX_KERB_FIT_RESIDUAL_FT = 1.0
PLAUSIBLE_CORNER_RADIUS_FT = (5.0, 60.0)  # outside this it isn't a street corner return


def fit_circle_ft(line: LineString) -> dict | None:
    """Least-squares (Kasa) circle fit to a traced kerb line.

    Returns {"radius_ft", "center", "sweep_deg", "max_residual_ft"} or None if the fit is
    degenerate. `sweep_deg` is how much of the circle the trace actually covers and is the
    key quality signal - a wide sweep with a small residual is a trustworthy radius; a
    narrow sweep is a circle inferred from almost-straight input.
    """
    coords = np.asarray(line.coords)
    if len(coords) < 3:
        return None
    xs, ys = coords[:, 0], coords[:, 1]
    a_matrix = np.c_[2 * xs, 2 * ys, np.ones(len(xs))]
    try:
        cx, cy, c = np.linalg.lstsq(a_matrix, xs ** 2 + ys ** 2, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    inner = c + cx ** 2 + cy ** 2
    if inner <= 0:
        return None
    radius = float(np.sqrt(inner))
    residual = float(np.abs(np.hypot(xs - cx, ys - cy) - radius).max())
    angles = np.unwrap(np.sort(np.arctan2(ys - cy, xs - cx)))
    sweep = float(np.degrees(angles.max() - angles.min()))
    return {"radius_ft": radius, "center": (float(cx), float(cy)),
            "sweep_deg": sweep, "max_residual_ft": residual}


def kerb_radius_is_usable(fit: dict | None) -> bool:
    """Whether a circle fit is well-enough conditioned to use as a corner radius."""
    if fit is None:
        return False
    low, high = PLAUSIBLE_CORNER_RADIUS_FT
    return (fit["sweep_deg"] >= MIN_KERB_ARC_SWEEP_DEG
            and fit["max_residual_ft"] <= MAX_KERB_FIT_RESIDUAL_FT
            and low <= fit["radius_ft"] <= high)


def assign_kerbs_to_corners(legs: dict, kerb_lines_ft: list) -> dict:
    """{frozenset(leg_a, leg_b): [LineString, ...]} - traced kerbs grouped by the corner
    they belong to, matched by which two legs their midpoint sits closest to."""
    by_corner: dict[frozenset, list] = {}
    for line in kerb_lines_ft:
        midpoint = line.interpolate(0.5, normalized=True)
        ranked = sorted(legs.items(), key=lambda kv: kv[1].centerline.distance(midpoint))
        if len(ranked) < 2:
            continue
        by_corner.setdefault(frozenset((ranked[0][0], ranked[1][0])), []).append(line)
    return by_corner


# A traced kerb is hand-clicked, so its vertices carry the mapper's noise: used raw it
# renders as a visibly kinked corner. Smoothing replaces that noise while keeping the
# traced POSITION and endpoints, which is the whole point of using the tracing.
SMOOTHED_ARC_POINTS = 24
MAX_ARC_FIT_RESIDUAL_FT = 1.5  # beyond this the kerb isn't really circular; smooth it instead


def _chaikin(line: LineString, iterations: int = 5) -> LineString:
    """Chaikin corner-cutting. Endpoints are preserved exactly; interior vertices are
    repeatedly replaced by points 1/4 and 3/4 along each segment, which converges on a
    smooth curve. Used where a kerb isn't circular enough to fit an arc."""
    coords = np.asarray(line.coords, dtype=float)
    for _ in range(iterations):
        if len(coords) < 3:
            break
        # Both cut points for every segment in one shot; interleaved with reshape rather
        # than appended one at a time (the vertex count quadruples each iteration).
        starts, ends = coords[:-1], coords[1:]
        cuts = np.empty((2 * len(starts), 2))
        cuts[0::2] = 0.75 * starts + 0.25 * ends
        cuts[1::2] = 0.25 * starts + 0.75 * ends
        coords = np.vstack([coords[:1], cuts, coords[-1:]])
    return LineString(coords)


def smooth_traced_arc(line: LineString) -> LineString:
    """A clean curve following a traced kerb: same endpoints, same path, no click noise.

    Preferred method is to fit a circle to the traced points and redraw the arc between
    the traced ENDPOINTS along that circle. That is smooth by construction and still sits
    on the mapped kerb - unlike the old fitted fillet, which took only the radius and then
    redrew the arc off our own estimated curb lines, landing it feet away.

    Falls back to Chaikin smoothing where the kerb isn't circular enough to fit (a
    compound return, or a trace covering more than one curve).
    """
    fit = fit_circle_ft(line)
    if fit is None or fit["max_residual_ft"] > MAX_ARC_FIT_RESIDUAL_FT:
        return _chaikin(line)

    cx, cy = fit["center"]
    radius = fit["radius_ft"]
    start, end = np.array(line.coords[0]), np.array(line.coords[-1])
    mid = np.array(line.interpolate(0.5, normalized=True).coords[0])
    a0 = np.arctan2(start[1] - cy, start[0] - cx)
    a1 = np.arctan2(end[1] - cy, end[0] - cx)
    a_mid = np.arctan2(mid[1] - cy, mid[0] - cx)

    # Two ways round the circle; take the one that actually passes through the traced
    # midpoint, so a reflex return isn't silently replaced by the short way round.
    candidates = [(a1 - a0) % (2 * np.pi), (a1 - a0) % (2 * np.pi) - 2 * np.pi]
    def passes_mid(sweep):
        t = ((a_mid - a0) / sweep) if sweep else 0.0
        return 0.0 <= t <= 1.0
    sweep = next((c for c in candidates if passes_mid(c)), min(candidates, key=abs))

    angles = a0 + np.linspace(0, sweep, SMOOTHED_ARC_POINTS)
    # Deliberately NOT pinned back to the raw traced endpoints. a0/a1 are already those
    # endpoints projected onto the fitted circle, so the arc starts and ends within the
    # fit residual (<=1.5 ft) of where they were traced. Snapping the ends back to the
    # raw points put a kink at each end - it made the smoothed arc turn MORE sharply than
    # the trace it was meant to clean up. The curb lines trim to the arc's own ends, so
    # nothing downstream needs the raw endpoints.
    return LineString([(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles])


# How much of each traced curb either side of a corner is handed to the smoothing pass.
# The traced corner returns already live in the leg curbs (assign_curb_points_to_legs puts
# each return's vertices on the two sides it joins), so this only has to take the click
# noise off the join, not invent a curve.
CORNER_BLEND_FT = 8.0


def traced_corner_join(curb_a: LineString, curb_b: LineString) -> tuple[LineString, LineString, LineString]:
    """Join two traced curbs around the corner they share, smoothing the seam.

    Both curbs are the surveyor's own traced kerb, ending where the tracing ends - which for
    a mapped corner is partway around the return. So there is no corner to construct: the
    two ends are already at the corner and this walks from one to the other, taking the last
    CORNER_BLEND_FT of each, bridging whatever gap the tracing left, and Chaikin-smoothing
    the result. Returns the same (trimmed_a, arc, trimmed_b) contract as fillet_curb_corner,
    with the arc running from curb_a's side to curb_b's side.
    """
    blend_a = min(CORNER_BLEND_FT, curb_a.length / 2)
    blend_b = min(CORNER_BLEND_FT, curb_b.length / 2)
    head_a = substring(curb_a, 0, blend_a)
    head_b = substring(curb_b, 0, blend_b)
    seam = LineString(list(head_a.coords)[::-1] + list(head_b.coords))
    return substring(curb_a, blend_a, curb_a.length), _chaikin(seam), substring(curb_b, blend_b, curb_b.length)


def traced_corner_arc(kerb_lines: list, curb_a: LineString, curb_b: LineString) -> LineString | None:
    """One traced kerb, oriented to run from curb_a's side to curb_b's side.

    build_corner_fillets' contract is (trimmed_a, arc, trimmed_b) with the arc running
    from its tangent point on curb_a to the one on curb_b, and build_pavement_polygon's
    ring walk depends on that order. A traced kerb has whatever direction the mapper drew
    it in, so it is reversed if needed. Where several kerbs share a corner the longest is
    used - the others are usually short ramp segments rather than the return itself.
    """
    usable = [line for line in kerb_lines if line.length > 1.0]
    if not usable:
        return None
    line = max(usable, key=lambda l: l.length)
    start, end = Point(line.coords[0]), Point(line.coords[-1])
    if start.distance(curb_a) > end.distance(curb_a):
        line = LineString(list(line.coords)[::-1])
    return smooth_traced_arc(line)


def corner_radii_from_kerbs(legs: dict, kerb_lines_ft: list[LineString],
                             fallback_radius_ft: float) -> tuple[dict, list[str]]:
    """Per-corner radii derived from traced OSM kerb lines, plus notes on what happened.

    Returns ({frozenset(leg_a, leg_b): radius_ft}, notes). Corners with no usable traced
    kerb are simply absent - the caller uses `fallback_radius_ft` for those, so a site
    with one traced corner gets one real radius and keeps the placeholder elsewhere
    rather than having one corner's measurement spread over the whole junction.

    A kerb way is assigned to the corner between the two legs it sits closest to. Several
    traces at one corner are combined by median, which is what makes two independent
    tracings of the same return (13.6 and 13.4 ft at Columbia & Princeton) reinforce each
    other, and stops a single odd trace from deciding the answer alone.
    """
    by_corner: dict[frozenset, list[float]] = {}
    notes: list[str] = []

    for line in kerb_lines_ft:
        fit = fit_circle_ft(line)
        midpoint = line.interpolate(0.5, normalized=True)
        ranked = sorted(legs.items(), key=lambda kv: kv[1].centerline.distance(midpoint))
        if len(ranked) < 2:
            continue
        corner = frozenset((ranked[0][0], ranked[1][0]))
        if not kerb_radius_is_usable(fit):
            reason = ("degenerate fit" if fit is None else
                      f"sweep {fit['sweep_deg']:.0f} deg, residual {fit['max_residual_ft']:.2f} ft, "
                      f"radius {fit['radius_ft']:.1f} ft")
            notes.append(f"kerb trace at {'/'.join(sorted(corner))} not usable as a corner radius "
                          f"({reason}) - too short an arc, too poor a fit, or not a corner return.")
            continue
        by_corner.setdefault(corner, []).append(fit["radius_ft"])

    radii = {}
    for corner, values in by_corner.items():
        radius = float(np.median(values))
        radii[corner] = radius
        spread = f", {len(values)} traces spanning {min(values):.1f}-{max(values):.1f} ft" if len(values) > 1 else ""
        notes.append(f"corner {'/'.join(sorted(corner))}: radius {radius:.1f} ft from traced OSM kerb"
                      f"{spread} (placeholder was {fallback_radius_ft:.0f} ft).")

    # Untraced corners: prefer the median of THIS junction's own measured corners over the
    # site-wide placeholder. A generic 20 ft next to corners measured at 13.5 ft inflates
    # the modelled throat enough to swallow the real footway - which is what was dropping
    # tactile pads at Columbia & Princeton. Still an inference, but one drawn from the same
    # junction rather than from a typical-value assumption, and reported as such.
    if radii:
        local_default = float(np.median(list(radii.values())))
        untraced = [frozenset((a, b)) for a, b in _corner_pairs(legs)
                     if frozenset((a, b)) not in radii]
        if untraced and abs(local_default - fallback_radius_ft) > 1.0:
            for corner in untraced:
                radii[corner] = local_default
            names = ", ".join("/".join(sorted(c)) for c in untraced)
            notes.append(f"untraced corner(s) {names}: using {local_default:.1f} ft, the median of this "
                          f"junction's own traced corners, instead of the {fallback_radius_ft:.0f} ft "
                          f"site placeholder. Trace them to replace this.")
    return radii, notes


def _corner_pairs(legs: dict) -> list[tuple[str, str]]:
    """Angularly adjacent leg pairs - the same corners build_corner_fillets() forms."""
    usable = {name: leg for name, leg in legs.items() if leg.left_curb is not None}
    ordered = sorted(usable.items(), key=lambda kv: _leg_bearing(kv[1]))
    return [(ordered[i][0], ordered[(i + 1) % len(ordered)][0]) for i in range(len(ordered))]


# Building a leg's curb from the surveyor's traced kerb ways.
#
# EVERY traced kerb way is curb. The earlier version took only kerb=raised and only the
# single longest run per side, which threw away exactly the geometry that matters: the
# corner returns are tagged kerb=lowered (they're the ramps), so the SW corner of Broad &
# Greenwood - traced in full - was being dropped and redrawn as a fitted fillet off the
# NJDOT centerline. Raised vs lowered is a height, not a question of where the curb is.
#
# Each traced VERTEX is placed in the leg frame as (station along the centerline, signed
# offset from it), and assigned to the one leg side whose half-width it best matches. That
# splits a corner return between the two sides it joins, which is what a corner return is.
# The curb is then the traced points themselves, in station order - no offsetting, no
# fitting, no fillet. Nothing is invented except where nothing was traced.
CURB_POINT_MAX_WIDTH_RATIO = 2.6   # |offset| / half-width; corner returns flare to ~2.3x
CURB_POINT_MIN_WIDTH_RATIO = 0.45  # below this it's a median or a driveway, not this curb
CURB_POINT_BEHIND_TOLERANCE_FT = 3.0
# A vertex a little behind a leg's junction node is still claimable - a corner return's own
# geometry straddles station 0, and dropping those vertices loses the corner. But a leg must
# never outbid one the vertex lies IN FRONT of, and unpenalised it can: at E Broad & Princeton
# the two legs are 179.9 deg apart, and the vertex where East Broad's north kerb changes from
# the corner return to the straight run sits 0.8 ft ahead of e_broad_st_east and 0.8 ft BEHIND
# e_broad_st_west - on the far side of the intersection from it. The west leg's half-width
# happened to match a shade better (0.995 vs 1.010), so it took the vertex; the 58.3 ft way it
# was the near end of then had one point left, curb_line_from_points needs two, and the whole
# stretch was discarded. That left e_broad_st_east's north kerb "traced only from 59 ft out"
# and 58 ft of a surveyed no-stopping kerb unhatched.
#
# Larger than any ratio the window admits (2.6), so forward always beats behind and the ratio
# only ever breaks ties among legs that all have the vertex ahead of them.
CURB_POINT_BEHIND_PENALTY = 10.0
# Out along a leg, past its corner returns, a kerb that IS that leg's kerb runs along it.
# Offset alone can't tell the difference: at W Broad & Louellen a kerb swinging from 16 ft
# to 37 ft off Louellen's alignment over 60 ft - a driveway apron running away from the
# street - sits inside any offset window wide enough to admit the real south kerb at 34 ft,
# and claiming it measured the leg at 66 ft. A kerb 53 degrees off the street is not the
# street's edge. Inside the corner zone the test is suspended, because a corner return
# sweeps through 90 degrees by definition and is still curb.
CURB_POINT_MAX_SKEW_DEG = 30.0
CURB_POINT_CORNER_ZONE_FT = 40.0


def _vertex_tangents(line: LineString) -> np.ndarray:
    """Unit direction of a polyline at each of its own vertices.

    Averages the segments either side of a vertex (one-sided at the ends), so a vertex on a
    curve gets the curve's local heading rather than one arbitrary neighbouring segment's.
    """
    coords = np.asarray(line.coords, dtype=float)
    if len(coords) < 2:
        return np.zeros_like(coords)
    steps = np.diff(coords, axis=0)
    tangents = np.zeros_like(coords)
    tangents[:-1] += steps
    tangents[1:] += steps
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    return np.divide(tangents, norms, out=np.zeros_like(tangents), where=norms > 0)


def _line_direction(line: LineString) -> np.ndarray:
    coords = np.asarray(line.coords)
    vec = coords[-1] - coords[0]
    norm = np.hypot(*vec)
    return vec / norm if norm else np.array([1.0, 0.0])


@lru_cache(maxsize=512)
def _polyline_frame(centerline: LineString):
    """(vertices, unit segment directions, segment lengths, station at each vertex).

    The one description of a leg's frame. Both directions of the transform read it, so
    station_offset(_point_at(...)) round-trips exactly - it did not when the forward
    direction used segment tangents and the inverse estimated one from a +/-2 ft window.

    Cached on the centerline itself. Every point this project places goes through the frame -
    a scenario resolves it ~6,000 times per site - and a leg centerline is a 2-3 vertex line,
    so rebuilding the four arrays each time cost more than the projection it exists to serve.
    Shapely geometries hash by value and are immutable, so the key is exactly the input: a leg
    whose centerline is replaced (which is how the width fit re-centres one) gets a new entry
    rather than a stale frame.

    The returned arrays are shared, so callers must not write to them. Nothing here does -
    every consumer indexes or does arithmetic producing new arrays - and marking them
    read-only is what keeps that true rather than conventional.
    """
    verts = np.asarray(centerline.coords, dtype=float)
    seg_vec = verts[1:] - verts[:-1]
    seg_len = np.hypot(seg_vec[:, 0], seg_vec[:, 1])
    seg_dir = seg_vec / np.where(seg_len > 0, seg_len, 1.0)[:, None]
    cumulative = np.concatenate(([0.0], np.cumsum(seg_len)))
    for array in (verts, seg_dir, seg_len, cumulative):
        array.flags.writeable = False
    return verts, seg_dir, seg_len, cumulative


def _frame_at(centerline: LineString, station: float) -> tuple[np.ndarray, np.ndarray]:
    """(origin, unit tangent) of the leg frame at `station`, extrapolating past either end."""
    verts, seg_dir, _seg_len, cumulative = _polyline_frame(centerline)
    i = int(np.clip(np.searchsorted(cumulative, station, side="right") - 1, 0, len(seg_dir) - 1))
    return verts[i] + seg_dir[i] * (station - cumulative[i]), seg_dir[i]


def station_offset(centerline: LineString, xy) -> tuple[float, float]:
    """A point in the leg's frame: distance along the centerline, and signed distance from
    it - positive to the left, matching Leg.left_curb / right_curb.

    The station is signed. LineString.project() clamps to [0, length], so everything behind
    the junction comes back as station 0 with a small offset - which let a leg claim the
    curb of the leg OPPOSITE it and draw it straight back through the intersection. Behind
    the junction the station is measured against the leg's own starting tangent instead, so
    it comes out negative and those points are rejected.

    One point through the vectorized path, so there is only ever one definition of the frame.
    """
    stations, offsets = station_offset_many(centerline, np.asarray([xy], dtype=float))
    return float(stations[0]), float(offsets[0])


def _point_at(centerline: LineString, station: float, offset: float) -> tuple[float, float]:
    origin, tangent = _frame_at(centerline, station)
    return tuple(origin + np.array([-tangent[1], tangent[0]]) * offset)


def station_offset_many(centerline: LineString, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """station_offset() for many points at once: (stations, offsets) arrays.

    Same frame and the same signed-station convention as the scalar version, but the whole
    (points x centerline segments) projection is one numpy expression instead of two shapely
    calls per point. Centerlines carry a handful of vertices, so the matrix is small and
    this collapses the dominant cost of reading a junction's traced kerbs.
    """
    verts, seg_dir, seg_len, cumulative = _polyline_frame(centerline)
    seg_start = verts[:-1]

    pts = np.atleast_2d(np.asarray(points, dtype=float))
    rel = pts[:, None, :] - seg_start[None, :, :]             # (p, s, 2)
    along = np.einsum("psc,sc->ps", rel, seg_dir)
    clamped = np.clip(along, 0.0, seg_len[None, :])
    perp = rel - clamped[:, :, None] * seg_dir[None, :, :]
    nearest = np.argmin(np.hypot(perp[:, :, 0], perp[:, :, 1]), axis=1)

    rows = np.arange(len(pts))
    stations = cumulative[nearest] + clamped[rows, nearest]
    tangents = seg_dir[nearest]
    rel_nearest = rel[rows, nearest]
    offsets = tangents[:, 0] * rel_nearest[:, 1] - tangents[:, 1] * rel_nearest[:, 0]

    # Past either end, measure against that end's tangent rather than letting the projection
    # clamp. Behind the junction this is what keeps a station negative, so a leg can't claim
    # the curb of the leg opposite it (see station_offset). Past the far end it stops every
    # point beyond the leg's working length from collapsing onto the same station, and makes
    # this an exact inverse of _point_at over the whole line.
    for outside, vertex, direction, base in (
            (stations <= 0, verts[0], seg_dir[0], 0.0),
            (stations >= cumulative[-1], verts[-1], seg_dir[-1], cumulative[-1])):
        if outside.any():
            rel_end = pts[outside] - vertex
            stations[outside] = base + rel_end @ direction
            offsets[outside] = direction[0] * rel_end[:, 1] - direction[1] * rel_end[:, 0]
    return stations, offsets


def assign_curb_points_to_legs(legs: dict, kerb_lines: list[LineString],
                                ratio_bounds: tuple[float, float] | None = None) -> dict:
    """{leg_name: {"left": [(station, offset), ...], "right": [...]}} from traced kerbs.

    Every vertex of every traced kerb way is considered, and goes to the single leg side
    whose half-width it sits closest to in proportional terms. One vertex can only be one
    piece of curb, so a corner return splits between the two sides that meet there rather
    than being drawn twice.

    Vectorized over vertices: each leg scores every traced vertex in one pass, and the
    winning leg per vertex is an argmin over the resulting (legs x vertices) score matrix.

    `ratio_bounds` widens (or narrows) the window a vertex has to fall in to be claimed at
    all. It exists because judging a vertex against a width the caller is only about to
    measure FROM that vertex is circular, and the circularity bites both ways at W Broad &
    Louellen: with the window at its normal width, Louellen St's south kerb - 155 ft of it,
    at a steady 34 ft offset - sat at 3.5x the half-width then assumed and was discarded, so
    the leg measured 19 ft wide off its north kerb alone; and W Broad's near kerb, 6.5 ft off
    NJDOT's badly off-centre alignment, sat at 0.43x and was discarded as a median. Opening
    the window admits both, and the proportional scoring still hands each vertex to the leg
    it best fits. See src/geometry/intersection.py:_fit_legs_to_traced_kerbs.
    """
    if not kerb_lines:
        return {}
    pts = np.concatenate([np.asarray(line.coords, dtype=float) for line in kerb_lines])
    tangents = np.concatenate([_vertex_tangents(line) for line in kerb_lines])
    low, high = ratio_bounds or (CURB_POINT_MIN_WIDTH_RATIO, CURB_POINT_MAX_WIDTH_RATIO)
    min_cosine = np.cos(np.radians(CURB_POINT_MAX_SKEW_DEG))

    names, stations, offsets, ratios = [], [], [], []
    for name, leg in legs.items():
        if leg.curb_to_curb_ft is None:
            continue
        leg_stations, leg_offsets = station_offset_many(leg.centerline, pts)
        ratio = np.abs(leg_offsets) / (leg.curb_to_curb_ft / 2)
        # abs: a kerb traced against the leg's outward direction is still parallel to it.
        skewed = np.abs(tangents @ _line_direction(leg.centerline)) < min_cosine
        # np.inf marks "this leg can't claim this vertex", so it never wins the argmin.
        disqualified = ((leg_stations < -CURB_POINT_BEHIND_TOLERANCE_FT)
                        | (ratio < low) | (ratio > high)
                        | (skewed & (leg_stations > CURB_POINT_CORNER_ZONE_FT)))
        # Still claimable behind the node, but only if nobody has it in front - see
        # CURB_POINT_BEHIND_PENALTY.
        score = ratio + np.where(leg_stations < 0, CURB_POINT_BEHIND_PENALTY, 0.0)
        names.append(name)
        stations.append(leg_stations)
        offsets.append(leg_offsets)
        ratios.append(np.where(disqualified, np.inf, score))
    if not names:
        return {}

    ratios = np.vstack(ratios)
    winner = np.argmin(ratios, axis=0)
    claimed = np.isfinite(ratios[winner, np.arange(ratios.shape[1])])

    out: dict[str, dict[str, list]] = {}
    for leg_index, name in enumerate(names):
        mine = claimed & (winner == leg_index)
        if not mine.any():
            continue
        leg_stations, leg_offsets = stations[leg_index][mine], offsets[leg_index][mine]
        for side, on_side in (("left", leg_offsets > 0), ("right", leg_offsets <= 0)):
            if on_side.any():
                out.setdefault(name, {})[side] = list(
                    zip(leg_stations[on_side].tolist(), leg_offsets[on_side].tolist()))
    return out


# Extrapolating past the end of the tracing. A curb that leaves the corner is running down
# the street, so it can diverge from the centerline by a few degrees (NJDOT's alignment
# error) but not more. Taking the slope off the last two traced vertices instead read the
# flare of a corner return - at Columbia & Princeton the south leg is traced for only 9 ft,
# all of it return, and running that slope out 100 ft crossed the two curbs into an X.
CURB_EXTRAPOLATION_MAX_SLOPE = 0.11        # ~6 degrees
CURB_EXTRAPOLATION_MIN_BASELINE_FT = 15.0  # shorter than this is corner, not street


def _outward_slope(points: list[tuple[float, float]]) -> float:
    """d(offset)/d(station) for the outward end of a traced side, or 0 if the tracing is
    too short to establish one - in which case the curb continues at the width last seen."""
    end_station, end_offset = points[-1]
    for station, offset in reversed(points[:-1]):
        if end_station - station >= CURB_EXTRAPOLATION_MIN_BASELINE_FT:
            slope = (end_offset - offset) / (end_station - station)
            return float(np.clip(slope, -CURB_EXTRAPOLATION_MAX_SLOPE, CURB_EXTRAPOLATION_MAX_SLOPE))
    return 0.0


def _inward_slope(points: list[tuple[float, float]]) -> float:
    """d(offset)/d(station) for the JUNCTION end of a traced side. Mirror of _outward_slope."""
    start_station, start_offset = points[0]
    for station, offset in points[1:]:
        if station - start_station >= CURB_EXTRAPOLATION_MIN_BASELINE_FT:
            slope = (offset - start_offset) / (station - start_station)
            return float(np.clip(slope, -CURB_EXTRAPOLATION_MAX_SLOPE, CURB_EXTRAPOLATION_MAX_SLOPE))
    return 0.0


def through_street_sides(legs: dict) -> set:
    """{(leg name, side)} for the kerbs that run STRAIGHT THROUGH the junction.

    Two angularly-adjacent legs more than THROUGH_STREET_ANGLE_DEG apart are one street
    passing through, and the pair of kerbs facing away from the stem is one unbroken kerb with
    no corner in it. Paired the way build_corner_fillets pairs them - leg A's LEFT with leg B's
    RIGHT - so the answer is per side.

    Computed from the leg centerlines alone, which is what lets _apply_traced_curb_lines use it:
    the corner fillets are not built yet at that point, and they depend on the curb lines.
    """
    usable = {name: leg for name, leg in legs.items() if leg.left_curb is not None}
    if len(usable) < 2:
        return set()
    ordered = sorted(usable.items(), key=lambda kv: _leg_bearing(kv[1]))
    sides = set()
    for i, (name_a, leg_a) in enumerate(ordered):
        name_b, leg_b = ordered[(i + 1) % len(ordered)]
        if _through_street(leg_a, leg_b):
            sides.add((name_a, "left"))
            sides.add((name_b, "right"))
    return sides


def curb_line_from_points(points: list[tuple[float, float]], leg: "Leg",
                          working_length_ft: float,
                          extend_to_junction: bool = False) -> LineString | None:
    """One leg side's curb, straight off the traced points.

    The points are the surveyor's own vertices, kept as traced and ordered along the leg.
    The outward end is extended along the bearing of the last traced stretch to reach the
    leg's working length, when the tracing stops short of it.

    The junction end is normally left exactly where the tracing ends - the corner is built
    from the traced geometry there, not by running this line on into the intersection.
    `extend_to_junction` lifts that for a side with NO CORNER RETURN, where the kerb genuinely
    runs straight through and stopping short is the fabrication. The north side of E Broad at
    Princeton is one unbroken kerb; the OSM way covering its last 20 ft before the junction has
    only two vertices, one of which the collinear leg on the far side legitimately claims, so
    the west leg's curb began 20.7 ft out and its no-stopping hatching could not be built
    inside that. Extending it in is the same extrapolation the outward end already gets, along
    a bearing the tracing establishes over 60+ ft of straight kerb - see through_street_sides
    for what licenses it.
    """
    ordered = sorted(points)
    if len(ordered) < 2:
        return None
    # One vertex per station: two traced ways can share an endpoint, and a curb that
    # doubled back in station would fold the pavement edge over itself.
    deduped = [ordered[0]]
    for station, offset in ordered[1:]:
        if station - deduped[-1][0] > 0.25:
            deduped.append((station, offset))
    if len(deduped) < 2:
        return None

    if deduped[-1][0] < working_length_ft:
        deduped.append((working_length_ft,
                        deduped[-1][1] + _outward_slope(deduped) * (working_length_ft - deduped[-1][0])))
    if extend_to_junction and deduped[0][0] > 0.0:
        station = deduped[0][0]
        deduped.insert(0, (0.0, deduped[0][1] - _inward_slope(deduped) * station))

    return LineString([_point_at(leg.centerline, s, o) for s, o in deduped])


def trimmed_curb_lines(legs: dict, corner_fillets: dict) -> dict[str, dict[str, LineString]]:
    """Each leg side clipped at the corner tangent point, i.e. the curb as it actually
    bounds the pavement.

    A leg's raw curb line deliberately overshoots the junction so the fillet has something
    to trim into - so drawing it raw puts curb lines straight across the middle of the
    intersection, marking a curb where there is none. The pavement polygon and the 3D
    export already use the trimmed pieces; this is how the plan view says the same thing.
    Sides whose corner failed to build keep the raw line, which is honest: that corner has
    no tangent point.
    """
    out = {name: {"left": leg.left_curb, "right": leg.right_curb} for name, leg in legs.items()}
    for (name_a, name_b), pieces in corner_fillets.items():
        if "error" in pieces:
            continue
        if name_a in out:
            out[name_a]["left"] = pieces["trimmed_a"]
        if name_b in out:
            out[name_b]["right"] = pieces["trimmed_b"]
    return out
