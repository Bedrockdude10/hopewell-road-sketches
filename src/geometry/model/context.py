"""Measurements about the street's surroundings rather than its carriageway: where a
crossing would fall if nobody surveyed one, and how wide the sidewalk band is."""

import numpy as np
from shapely.geometry import LineString
from src.geometry.model.leg_frame import leg_bearing_deg



# A leg whose bearing is within this of the reverse of another's is that leg's continuation
# across the junction, not a street crossing it. Broad St's two legs do not make each other's
# crosswalk longer; Greenwood's do. Same threshold THROUGH_STREET_ANGLE_DEG uses, and for the
# same reason - see is_through_street.
CROSS_STREET_MAX_ANGLE_DEG = 150.0
# How far beyond the cross street's kerb line a crosswalk actually sits. MEASURED, not chosen:
# fitted against the 11 OSM-surveyed crossings at the four sites, which give a mean setback of
# 8.3 ft with a standard deviation of 2.4 ft (range 5.1-13.9). See
# tests/test_sites.py:test_the_crosswalk_estimate_reproduces_the_surveyed_crossings.
#
# NOT named CROSSWALK_SETBACK_FT: daylighting.py already uses that name for R.S. 39:4-138(e)'s
# 25 ft, and cross_streets.py reads THIS one (placing the unmarked crosswalk N.J.S.A. 39:1-1
# puts at every approach) and hands the result to daylighting.py, which measures the statutory
# 25 ft from it. Only one of the two is a legal figure; one grep must not return both.
CROSSWALK_OFFSET_FROM_KERB_FT = 8.3


def crosswalk_estimate_ft(leg_name: str, legs: dict) -> float:
    """Where a crosswalk goes on a leg with no surveyed crossing to copy.

    The controlling dimension is the CROSS street's half-width, not this leg's own kerb: a
    crosswalk sits just outside the box the intersecting roadway occupies, and the corner
    return it also has to clear scales with that same roadway. Reproduces all 11 surveyed
    crossings to a standard deviation of 2.4 ft.

    WHY NOT the two obvious alternatives, both fitted against those 11 and both rejected: the
    fillet tangent point (leg_clearance_ft) scattered -31.5 to +41.7 ft, and projecting the
    cross street's kerb lines onto this leg's centerline scattered -38.0 to -2.3 ft and
    returned 119.7 ft where a near-parallel through street meets the leg at a shallow angle.
    """
    leg = legs[leg_name]
    bearing = leg_bearing_deg(leg)
    widest_cross_half_ft = 0.0
    for other_name, other in legs.items():
        if other_name == leg_name:
            continue
        apart = abs(leg_bearing_deg(other) - bearing) % 360
        apart = min(apart, 360 - apart)
        if apart > CROSS_STREET_MAX_ANGLE_DEG:
            continue        # this leg's own continuation across the junction
        widest_cross_half_ft = max(widest_cross_half_ft, other.curb_to_curb_ft / 2)
    return widest_cross_half_ft + CROSSWALK_OFFSET_FROM_KERB_FT


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
