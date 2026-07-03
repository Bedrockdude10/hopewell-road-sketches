"""Geometry operations: WGS84 buffering, radius clipping, CRS reprojection, and
curb-line / corner-fillet construction from centerlines + widths."""
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring

WGS84 = "EPSG:4326"
NJ_STATE_PLANE_FT = "EPSG:3424"  # NAD83(HARN) / New Jersey (ftUS)


def buffer_point_wgs84(point: Point, radius_m: float) -> tuple[float, float, float, float]:
    """Buffer a WGS84 point by radius_m meters (via a local UTM projection) and
    return a WGS84 bbox as (minx, miny, maxx, maxy)."""
    point_gs = gpd.GeoSeries([point], crs=WGS84)
    utm_crs = point_gs.estimate_utm_crs()
    buffered = point_gs.to_crs(utm_crs).buffer(radius_m).to_crs(WGS84)
    return tuple(buffered.total_bounds)


def clip_to_radius(gdf: gpd.GeoDataFrame, center: Point, radius_m: float) -> gpd.GeoDataFrame:
    """Clip a WGS84 GeoDataFrame to a circular radius (meters) around center,
    trimming feature geometry (not just filtering by bbox)."""
    center_gs = gpd.GeoSeries([center], crs=WGS84)
    utm_crs = center_gs.estimate_utm_crs()
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

    def __post_init__(self):
        if self.curb_to_curb_ft is not None:
            half = self.curb_to_curb_ft / 2
            self.left_curb = self.centerline.offset_curve(half)
            self.right_curb = self.centerline.offset_curve(-half)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


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
    """
    pa, da = np.array(curb_a.coords[0]), _unit(np.array(curb_a.coords[1]) - np.array(curb_a.coords[0]))
    pb, db = np.array(curb_b.coords[0]), _unit(np.array(curb_b.coords[1]) - np.array(curb_b.coords[0]))

    # true square-corner vertex: intersection of the two curb lines, extended
    a_matrix = np.array([da, -db]).T
    t, _s = np.linalg.solve(a_matrix, pb - pa)
    vertex = pa + t * da

    theta = np.arccos(np.clip(np.dot(da, db), -1, 1))
    if theta < np.radians(1) or theta > np.radians(179):
        raise ValueError(f"Curb lines meet at an implausible angle ({np.degrees(theta):.1f} deg) - check inputs.")

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


def build_corner_fillets(legs: dict, radius_ft: float) -> dict:
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
        try:
            trimmed_a, arc, trimmed_b = fillet_curb_corner(leg_a.left_curb, leg_b.right_curb, radius_ft)
            results[(name_a, name_b)] = {"trimmed_a": trimmed_a, "arc": arc, "trimmed_b": trimmed_b, "radius_ft": radius_ft}
        except ValueError as e:
            results[(name_a, name_b)] = {"error": str(e)}
    return results


def leg_clearance_ft(leg_name: str, legs: dict, corner_fillets: dict, buffer_ft: float = 3.0) -> float:
    """
    Distance from a leg's near point out past BOTH of its corner fillets'
    tangent points, plus a small buffer - the point beyond which the leg's
    curb lines run straight rather than curving through the corner. Use this
    to place crosswalks / raised crossings outside the curve, not inside it -
    a fixed small offset from the intersection center lands inside the curve
    for any leg wide enough or with a generous enough corner radius.
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
        if "error" in pieces:
            continue
        if leg_a == leg_name:
            max_along_dist = max(max_along_dist, centerline.project(Point(pieces["trimmed_a"].coords[0])))
        if leg_b == leg_name:
            max_along_dist = max(max_along_dist, centerline.project(Point(pieces["trimmed_b"].coords[0])))
    return max_along_dist + buffer_ft


def lane_narrowing_polygons_ft(leg: "Leg", stripe_width_ft: float,
                                start_left_ft: float = 0.0, start_right_ft: float = 0.0) -> list[Polygon]:
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
    intersection box where no paint actually exists."""
    half = leg.curb_to_curb_ft / 2
    inner_half = max(half - stripe_width_ft, 0.5)
    polys = []
    for curb, sign, start_ft in ((leg.left_curb, 1, start_left_ft), (leg.right_curb, -1, start_right_ft)):
        inner = leg.centerline.offset_curve(sign * inner_half)
        trimmed_curb = substring(curb, start_ft, curb.length)
        trimmed_inner = substring(inner, start_ft, inner.length)
        ring = list(trimmed_curb.coords) + list(reversed(trimmed_inner.coords))
        if len(ring) >= 3:
            polys.append(Polygon(ring))
    return polys


def lane_narrowing_edge_lines_ft(leg: "Leg", stripe_width_ft: float,
                                  start_left_ft: float = 0.0, start_right_ft: float = 0.0) -> list[LineString]:
    """The solid line marking the new, narrower travel lane's outer edge on
    each side - the same inner boundary lane_narrowing_polygons_ft's buffer
    zone starts from (11 ft from centerline for this site's proposals - see
    TARGET_LANE_WIDTH_FT in sites/broad_st_greenwood/scenarios.py) - drawn
    explicitly so the lane width actually reads on the render, rather than
    only being implied by wherever the diagonal hatching happens to start.
    start_left_ft/start_right_ft match lane_narrowing_polygons_ft's (see its
    docstring) so this line, the hatch fill, and the corner taper
    (lane_narrowing_taper_ft) all begin at the same point with no gap."""
    half = leg.curb_to_curb_ft / 2
    inner_half = max(half - stripe_width_ft, 0.5)
    lines = []
    for sign, start_ft in ((1, start_left_ft), (-1, start_right_ft)):
        inner = leg.centerline.offset_curve(sign * inner_half)
        lines.append(substring(inner, start_ft, inner.length))
    return lines


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


def lane_narrowing_taper_ft(leg: "Leg", stripe_width_ft: float, anchor_ft: float, target_ft: float,
                             n_points: int = 16) -> list[LineString]:
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
    half = leg.curb_to_curb_ft / 2
    inner_half = max(half - stripe_width_ft, 0.5)
    tapers = []
    for curb, sign, role in ((leg.left_curb, 1, "left"), (leg.right_curb, -1, "right")):
        inset = leg.centerline.offset_curve(sign * inner_half)
        p1 = np.array(inset.interpolate(anchor_ft).coords[0])
        p2 = np.array(curb.interpolate(target_ft).coords[0])
        n = _corner_bulge_normal(leg, role)
        d = p2 - p1
        denom = 2 * np.dot(d, n)
        if abs(denom) < 1e-6:
            continue  # p2 already (near enough) on the tangent line - no taper needed
        radius_ft = np.dot(d, d) / denom
        center = p1 + radius_ft * n
        a1 = np.arctan2(p1[1] - center[1], p1[0] - center[0])
        a2 = np.arctan2(p2[1] - center[1], p2[0] - center[0])
        delta = (a2 - a1 + np.pi) % (2 * np.pi) - np.pi
        angles = a1 + np.linspace(0, delta, n_points)
        tapers.append(LineString([(center[0] + radius_ft * np.cos(t), center[1] + radius_ft * np.sin(t))
                                   for t in angles]))
    return tapers


def bollard_points_ft(leg: "Leg", stripe_width_ft: float, start_ft: float,
                       spacing_ft: float = 10.0) -> list[tuple[float, float]]:
    """Points down the center of each side's paint-only lane-narrowing buffer
    (same inner_half math as lane_narrowing_polygons_ft, so a bollard line
    always sits centered in the buffer that's actually painted, not a
    separately-guessed offset) - one line per curb, starting start_ft along
    the centerline (past the corner fillet curve, same clearance convention
    as crosswalks/stop bars/trees - see leg_clearance_ft) and spaced
    spacing_ft apart to the end of the leg. Used by
    src/geometry/treatments.py:add_bollards."""
    half = leg.curb_to_curb_ft / 2
    inner_half = max(half - stripe_width_ft, 0.5)
    lateral = (half + inner_half) / 2  # centered within the buffer strip
    length = leg.centerline.length
    points = []
    for sign in (1, -1):
        offset_line = leg.centerline.offset_curve(sign * lateral)
        d = start_ft
        while d <= min(length, offset_line.length):
            pt = offset_line.interpolate(d)
            points.append((pt.x, pt.y))
            d += spacing_ft
    return points


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


def hatch_lines_ft(polygon: Polygon, spacing_ft: float = 2.0, angle_deg: float = 45.0) -> list[LineString]:
    """Diagonal hatch lines filling a polygon, clipped to its boundary - used
    to render paint-only diagonal/chevron marking (e.g. corner_hatching_polygon
    above) without any real curb/pavement geometry change."""
    minx, miny, maxx, maxy = polygon.bounds
    diag = ((maxx - minx) ** 2 + (maxy - miny) ** 2) ** 0.5
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    theta = np.radians(angle_deg)
    u = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-u[1], u[0]])
    lines = []
    offset = -diag
    while offset <= diag:
        center = np.array([cx, cy]) + n * offset
        clipped = LineString([center - u * diag, center + u * diag]).intersection(polygon)
        if not clipped.is_empty:
            lines.extend(clipped.geoms if clipped.geom_type == "MultiLineString" else [clipped])
        offset += spacing_ft
    return lines


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

    return Polygon(ring)
