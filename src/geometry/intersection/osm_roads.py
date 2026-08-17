"""Tying this project's legs to somebody else's linework: NJDOT's SRI centrelines and OSM's
ways.

Both matches are by geometry rather than by name, and both can legitimately fail - a leg with no
match is a fact to report, not an error - so the thresholds that decide a match are constants here
rather than magic numbers at the call site."""


import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point

from src.render.coords import wgs84_to_state_plane
from src.sources.osm_context import fetch_roads
from src.geometry.model import (
    line_direction,
    station_offset_many,
)
from src.geometry.intersection.junction import RoadSpan



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
ROAD_CONTEXT_RADIUS_M = 130


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


def _assign_leg_pieces(pieces: list, leg_names: list[str], legs_cfg: dict, center_ft: Point,
                        sri: str = "?") -> dict[str, object]:
    """
    Match centerline pieces (all sharing one SRI, split at the intersection) to
    the configured leg names that reference that SRI, by nearest compass bearing.
    Generalizes to any number of pieces per SRI (2 for a through road, 1 for a
    dead-end/stub leg) and any intersection shape - nothing here assumes a
    4-way or perpendicular roads, only that each leg's config entry has an
    accurate `bearing_deg`.

    THE COUNTS HAVE TO MATCH, and when they do not it is a config error worth naming.
    A road network splits an SRI at the junction into as many pieces as there are
    approaches on it; the config says how many legs it has there. A disagreement means
    one of two mistakes, and both used to be silent or near-silent:

      MORE PIECES THAN LEGS - the road runs through the junction but the config only
      declares one side of it. This raised `min() iterable argument is empty` from the
      matcher below, naming neither the SRI nor the config. It is an easy mistake where
      NJDOT's name for an SRI describes only part of what it covers: at NJ 31 & W
      Delaware Ave, NJDOT carries the whole of Delaware Ave under one SRI called
      "E DELAWARE AVE", so the obvious reading puts the west leg on the neighbouring
      "PENNINGTON-TITUSVILLE RD" SRI - whose segment begins 334 ft west and never
      reaches this junction.

      MORE LEGS THAN PIECES - a leg declared on an SRI that has no approach for it here.
      That one never raised at all: the leg was simply absent from the returned dict, so
      it got no centerline, no curb line, no crossing and no mention.

    The leftover piece's bearing is reported because it IS the `bearing_deg` the missing
    leg needs, so the message contains the fix rather than just the diagnosis.
    """
    if len(pieces) != len(leg_names):
        bearings = [_bearing_deg((center_ft.x, center_ft.y), p.coords[-1]) for p in pieces]
        declared = ", ".join(f"{n} ({legs_cfg[n]['bearing_deg']:.1f} deg)" for n in leg_names)
        raise ValueError(
            f"SRI {sri} splits into {len(pieces)} piece(s) at this junction "
            f"(bearings {', '.join(f'{b:.1f}' for b in bearings)} deg) but the config declares "
            f"{len(leg_names)} leg(s) on it: {declared}. "
            + ("Add the missing leg(s) to config.yaml with the unmatched bearing(s) above - a "
               "road that runs THROUGH the junction has two approaches on one SRI, and NJDOT's "
               "name for an SRI may describe only part of what it covers."
               if len(pieces) > len(leg_names) else
               "Remove the extra leg(s), or move them to the SRI that actually carries them - "
               "a leg on an SRI with no piece here is drawn as nothing at all.")
        )

    assigned = {}
    remaining_names = list(leg_names)
    for piece in pieces:
        far_bearing = _bearing_deg((center_ft.x, center_ft.y), piece.coords[-1])
        best_name = min(remaining_names, key=lambda n: _bearing_diff(far_bearing, legs_cfg[n]["bearing_deg"]))
        assigned[best_name] = piece
        remaining_names.remove(best_name)
    return assigned


# A leg is matched to the OSM way whose geometry it lies along: within this far of the
# leg's own centerline, and pointing the same way. Both are needed - the cross street
# passes just as close to the junction, and a parallel service road points the same way.
ROAD_MATCH_MAX_OFFSET_FT = 40.0
ROAD_MATCH_MAX_ANGLE_DEG = 30.0
# ...and it has to be a CARRIAGEWAY. Geometry alone is not enough to identify one: east of
# Princeton Ave, OSM has a `highway=service, service=parking_aisle` way (772378208) running
# 0.5 ft from East Broad Street's centerline at 0.2 deg to it - indistinguishable from the
# street on distance and bearing, and it won the nearest-way tie. So the leg's operational
# tags were read off a parking aisle, which carries none, and East Broad Street's own
# `parking:both:restriction=no_stopping` (way 1546878992) was never seen: the proposal
# hatched that kerb for having 7.5 ft spare and reported it as untagged, while the
# restriction sat in the data the whole time.
#
# A driveway, a parking aisle, a footway and a cycleway all fail this; every leg at all four
# sites is one of these classes. A leg that matches nothing keeps its defaults and says so,
# which is the safe direction - it invents no restriction it cannot source.
ROAD_MATCH_HIGHWAY_CLASSES = frozenset({
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
    "living_street", "motorway_link", "trunk_link", "primary_link", "secondary_link",
    "tertiary_link",
})


def _match_legs_to_osm_roads(legs: dict, center_wgs84: Point, center_ft: Point) -> dict:
    """{leg name: (tags, aligned)} for the OSM highway way each leg runs along.

    `aligned` is True when the way is drawn in the same direction the leg points outward.
    It decides whether OSM's left/right mean the leg's left/right or the reverse.

    Matched on geometry rather than on the street name in config.yaml: names disagree
    between sources ("W Broad St" vs "West Broad Street"), and a leg is a piece of a
    specific way, not of a name.
    """
    try:
        roads = fetch_roads(center_wgs84, radius_m=ROAD_CONTEXT_RADIUS_M)
    except Exception as e:   # operational tags are an enhancement, not a dependency
        print(f"  NOTE: couldn't read OSM road tags ({type(e).__name__}); centerline styles "
              f"fall back to the site config.")
        return {}

    # Projected once, outside the leg loop. Each candidate way was re-transformed for every
    # leg, so a 4-leg junction did the same coordinate transform four times per way.
    carriageways = [
        (road, LineString(zip(*wgs84_to_state_plane.transform(
            [c[0] for c in road["coords_wgs84"]], [c[1] for c in road["coords_wgs84"]]))))
        for road in roads
        if road["tags"].get("highway") in ROAD_MATCH_HIGHWAY_CLASSES
    ]

    out: dict[str, list[RoadSpan]] = {}
    for name, leg in legs.items():
        leg_dir = line_direction(leg.centerline)
        spans = []
        for road, line in carriageways:
            along = np.dot(line_direction(line), leg_dir)
            angle = np.degrees(np.arccos(np.clip(abs(along), -1, 1)))
            if angle > ROAD_MATCH_MAX_ANGLE_DEG:
                continue
            # The stretch of THIS leg the way actually covers, in the leg's own frame. A way is
            # matched on lying along the leg over that stretch, not on being nearest the leg's
            # midpoint - see RoadSpan for why the difference discarded a real restriction.
            stations, offsets = station_offset_many(leg.centerline,
                                                    np.asarray(line.coords, dtype=float))
            lo = max(float(stations.min()), 0.0)
            hi = min(float(stations.max()), leg.centerline.length)
            if hi - lo < MIN_ROAD_SPAN_FT:
                continue
            # Measured over the part that overlaps, so a way running alongside for miles is not
            # judged by how far away its far end wanders.
            covering = (stations >= -MIN_ROAD_SPAN_FT) & (stations <= leg.centerline.length
                                                          + MIN_ROAD_SPAN_FT)
            if not covering.any():
                continue
            if float(np.abs(offsets[covering]).min()) > ROAD_MATCH_MAX_OFFSET_FT:
                continue
            spans.append(RoadSpan(start_ft=lo, end_ft=hi, tags=road["tags"],
                                   aligned=bool(along >= 0), way_id=road.get("id")))
        if spans:
            out[name] = sorted(spans, key=lambda span: span.start_ft)
    return out


# Below this a way barely touches a leg - usually the cross street clipping the junction node -
# and its tags describe a different street.
MIN_ROAD_SPAN_FT = 5.0
