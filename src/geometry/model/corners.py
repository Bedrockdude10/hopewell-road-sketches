"""Corner fillets, and the pavement polygon they close.

The junction's shape is built from one fillet per corner - a tangent arc between two legs' kerb
lines - and build_pavement_polygon rings them together. A junction whose legs meet too acutely for
their widths has no such ring, which is a real limit of this model rather than a bug, and
_acute_corner_diagnosis exists to say so in those terms."""

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring
from shapely.validation import explain_validity
from src.geometry.model.leg_frame import (_leg_bearing, line_direction, station_offset_many,
                                          unit_vector)
from src.geometry.model.traced_kerbs import traced_corner_arc, traced_corner_join



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
    pa, da = np.array(curb_a.coords[0]), unit_vector(np.array(curb_a.coords[1]) - np.array(curb_a.coords[0]))
    pb, db = np.array(curb_b.coords[0]), unit_vector(np.array(curb_b.coords[1]) - np.array(curb_b.coords[0]))

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
    bisector = unit_vector(da + db)

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


def is_through_street(leg_a, leg_b) -> bool:
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
    theta = np.arccos(np.clip(np.dot(line_direction(leg_a.centerline),
                                     line_direction(leg_b.centerline)), -1, 1))
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
        if is_through_street(leg_a, leg_b):
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
    treatments: diagonal paint hatching (src/geometry/treatments/corners.py:CornerHatching)
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
        if is_through_street(leg_a, leg_b):
            sides.add((name_a, "left"))
            sides.add((name_b, "right"))
    return sides


def corner_tangent_station_ft(leg_name: str, side: str, legs: dict, corner_fillets: dict) -> float:
    """Where this kerb stops running straight and begins turning into the corner, as a station.

    ONE geometric fact with two readers, which is why it lives here rather than with either of
    them. src/geometry/daylighting.py calls it the SIDE LINE of the intersecting street, because
    that is what R.S. 39:4-138(e) measures its 25 ft from; src/geometry/kerbs.py calls it the far
    end of the JUNCTION'S OWN MOUTH, because that is where the kerb a marking runs beside starts
    existing. Both are the same point - the corner fillet's tangent point - and the statute and
    the marking rule have no business each finding it for themselves.

    ZERO WHERE THE KERB RUNS STRAIGHT THROUGH, which is not a special case bolted on: a pair of
    legs more than THROUGH_STREET_ANGLE_DEG apart is one street, their outer kerbs are one
    unbroken kerb, and the fillet between them has no tangent point out along either leg. Reading
    that off the fillet alone would leave it to a clamp; through_street_sides is asked directly so
    the answer is stated rather than inferred. It is what makes MUTCD 11th ed. 3B.11(07) - a solid
    edge line MAY continue "through that part of an intersection with no intersecting approach
    (such as at the far side of a T-intersection)" - fall out of the geometry instead of needing a
    rule of its own.
    """
    leg = legs.get(leg_name)
    if leg is None:
        return 0.0
    if (leg_name, side) in through_street_sides(legs):
        return 0.0
    station = 0.0
    for (leg_a, leg_b), pieces in corner_fillets.items():
        if "error" in pieces:
            continue
        # build_corner_fillets pairs leg_a's LEFT curb with leg_b's RIGHT curb.
        if leg_a == leg_name and side == "left":
            tangent = pieces["trimmed_a"].coords[0]
        elif leg_b == leg_name and side == "right":
            tangent = pieces["trimmed_b"].coords[0]
        else:
            continue
        stations, _offsets = station_offset_many(leg.centerline,
                                                  np.asarray([tangent], dtype=float))
        station = max(station, float(stations[0]))
    return station


def junction_mouth_ft(leg_name: str, side: str, legs: dict, corner_fillets: dict,
                      crossing_reach_ft: float | None = None) -> tuple[float, float] | None:
    """The stations over which THIS junction opens this kerb, or None where it does not.

    The modelled junction is an intersecting approach like any other, and this is its mouth in
    the same (start, end) form src/geometry/cross_streets.py gives Blackwell Avenue's.

    IT ENDS AT THE CROSSWALK where the leg has one painted - `crossing_reach_ft`, how far that
    crossing reaches along THIS kerb - because the crosswalk is where a person reads the
    intersection as ending, and the corner outside it is the ground a painted curb extension goes
    on. Only where nothing is painted does it fall back to the corner return's tangent point,
    which is where the kerb starts and also the side line R.S. 39:4-138(e) measures from on
    exactly those legs. See paint.junction_mouths_ft, which resolves the reach; this function is
    the geometry and takes it as a number so it stays usable before any crossing is resolved.

    None, not (0, 0), where the kerb runs straight through. An opening of zero length is a fact
    that has to be filtered downstream by everyone who reads it; the absence of one is the same
    fact stated once, here.
    """
    if (leg_name, side) in through_street_sides(legs):
        return None
    end_ft = crossing_reach_ft
    if end_ft is None:
        end_ft = corner_tangent_station_ft(leg_name, side, legs, corner_fillets)
    return (0.0, end_ft) if end_ft > 0.0 else None
