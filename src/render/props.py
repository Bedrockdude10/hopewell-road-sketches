"""Street-furniture placement: streetlights, stop signs, traffic signals
(pole + pedestrian head), no-turn-on-red signs, and any site-specific extras
from config.yaml. Every function here only decides WHERE a prop goes and WHY
(a "source" string on every entry) - scripts/blender/blender_props.py is what
actually draws it. See sites/README.md for the `signals`/`props.extra`
config schema this reads from."""
import numpy as np
from shapely.geometry import LineString, Point

from src.geometry.intersection import IntersectionModel
from src.geometry.model import bollard_points_ft, build_pavement_polygon, leg_clearance_ft
from src.geometry.treatments import DesignState
from src.render.coords import wgs84_to_state_plane

STREETLIGHT_SIDEWALK_SETBACK_FT = 4
SIGN_SIDEWALK_SETBACK_FT = 3
PED_HEAD_POLE_OFFSET_FT = 3  # lateral offset (tangent to the corner, along the sidewalk) for a pedestrian
                             # signal head confirmed to be on a separate pole from the vehicle signal -
                             # placement approximation, no surveyed separate-pole location available


def control_nodes_ft(traffic_control: list[dict] | None) -> list[dict]:
    """Fetched OSM control nodes -> the same dicts with a state-plane `point_ft` added."""
    out = []
    for node in traffic_control or []:
        x, y = wgs84_to_state_plane.transform(node["lon"], node["lat"])
        out.append({**node, "point_ft": Point(x, y)})
    return out


def _osm_streetlight_props(nodes_ft: list[dict]) -> list[dict]:
    """Streetlights at their real OSM-surveyed positions.

    Returns [] when no highway=street_lamp node is mapped nearby - the case at every one
    of this project's four sites - and nothing is drawn. There is deliberately no
    fallback: a lamp at every corner was the previous behaviour and it fabricated all 14
    lamps across the four sites. See build_props and data_gaps.
    """
    props = []
    for node in nodes_ft:
        if node["tags"].get("highway") != "street_lamp":
            continue
        pos = (node["point_ft"].x, node["point_ft"].y)
        props.append({
            "type": "streetlight", "position_ft": pos, "heading_deg": 0.0,
            "surveyed_position": True,
            "source": "real (OSM highway=street_lamp node): surveyed pole position, not derived from "
                      "the corner geometry.",
        })
    return props


# How close a highway=crossing node must be to a crossing way's midpoint to be treated as
# that way's node. OSM normally places the node exactly on the way, so this only absorbs
# the odd case where a mapper split them; anything further away is a different crossing.
CROSSING_NODE_MATCH_FT = 20.0

# A detectable warning surface: 2 ft deep (into the footway, along the crossing) by 3 ft
# wide (along the kerb) - wider than it is long, as built. 24 in depth is the PROWAG
# minimum, so this is at standard rather than a generous guess; the previous 3x5 ft read
# visibly oversized at this scale.
#
# Defined here alongside placement and passed to the renderer in the prop dict, because
# the step-back that keeps a pad off the roadway is half the DEPTH - placement and
# geometry must agree on that number or pads end up in the carriageway (they have, twice).
TACTILE_PAD_DEPTH_FT = 2.0
TACTILE_PAD_WIDTH_FT = 3.0


def _merged_crossing_tags(line, crossing_tags: dict, nodes_ft: list[dict]) -> dict:
    """A crossing's tags from BOTH its way and its highway=crossing node.

    OSM splits the detail across the two: the way usually carries crossing:markings and
    button_operated, while tactile_paving, crossing:island and flashing_lights are often
    only on the node. Reading either alone under-reports - checking only the ways is what
    made an earlier data_gaps() claim Broad/Greenwood had no ADA data when all four of
    its crossings are tagged tactile_paving=yes. Node tags win on conflict, being the
    more specific of the two.
    """
    mid = line.interpolate(0.5, normalized=True)
    merged = dict(crossing_tags)
    for node in nodes_ft:
        if node["tags"].get("highway") != "crossing":
            continue
        if node["point_ft"].distance(mid) <= CROSSING_NODE_MATCH_FT:
            merged.update(node["tags"])
    return merged


def _crossing_endpoint_props(line, leg, tags: dict, leg_name: str, pavement) -> list[dict]:
    """Kerbside hardware at a crossing's two ends: pedestrian pushbuttons, RRFB beacons,
    and the tactile paving pad where the crossing meets each curb.

    All three are positioned from real surveyed geometry rather than derived: a crossing
    way runs sidewalk-centerline to sidewalk-centerline, so its ENDPOINTS are where the
    pushbutton pole and beacon stand, and its intersection with the leg's own curb lines
    is where the curb ramp and its truncated-dome pad sit.
    """
    props = []
    (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
    heading = np.degrees(np.arctan2(y1 - y0, x1 - x0))

    if tags.get("button_operated") == "yes":
        for i, (px, py) in enumerate(((x0, y0), (x1, y1))):
            # A pushbutton pole stands on the footway. The crossing way's endpoint is a
            # real surveyed position there, but our modelled pavement can be wider than
            # the real roadway and swallow it - at Broad & Greenwood that put 6 of 8
            # buttons in the carriageway. Step out along the crossing until clear, the
            # same treatment tactile pads get.
            placed = _step_clear_of_roadway((px, py), (x0, y0), (x1, y1), pavement)
            if placed is None:
                print(f"  NOTE: {leg_name} has a pushbutton-actuated crossing, but this end can't be "
                      f"placed clear of the modelled roadway - the modelled pavement covers the real "
                      f"footway here. Not drawn.")
                continue
            props.append({
                "type": "pedestrian_pushbutton", "position_ft": placed,
                # Face back along the crossing, toward the pedestrian waiting to use it.
                "heading_deg": heading + (180 if i == 0 else 0),
                "source": f"real (OSM button_operated=yes on {leg_name}'s crossing): a pushbutton-actuated "
                          "pedestrian phase really exists here, and this end of the surveyed crossing way is "
                          "where the pole stands. Approximation: stepped clear of the modelled roadway if "
                          "needed, and the button's height/offset from the pole are generic.",
            })

    if tags.get("flashing_lights") == "button":
        for i, (px, py) in enumerate(((x0, y0), (x1, y1))):
            props.append({
                "type": "rrfb", "position_ft": (px, py),
                "heading_deg": heading + (180 if i == 0 else 0),
                "surveyed_position": True,
                "source": f"real (OSM flashing_lights=button on {leg_name}'s crossing): a rectangular rapid "
                          "flashing beacon, at the surveyed end of the crossing way.",
            })

    if tags.get("tactile_paving") == "yes":
        props += _tactile_pad_props(line, pavement, leg_name, heading)
    return props


PUSHBUTTON_CLEARANCE_FT = 1.0   # a pole is a point; this much off the kerb reads as "on the footway"
PUSHBUTTON_MAX_STEP_FT = 12.0


def _step_clear_of_roadway(point, end_a, end_b, pavement):
    """Move `point` outward along the crossing until it is off the pavement, or None.

    Outward is away from the crossing's midpoint, which is always in the road. Returns
    the original point unchanged when it is already clear, so surveyed positions are only
    adjusted where our own geometry demands it.
    """
    if pavement is None:
        return point
    from shapely.geometry import Point as _P

    if not pavement.contains(_P(*point)):
        return point
    mid = ((end_a[0] + end_b[0]) / 2.0, (end_a[1] + end_b[1]) / 2.0)
    out_x, out_y = point[0] - mid[0], point[1] - mid[1]
    norm = np.hypot(out_x, out_y)
    if norm < 1e-6:
        return None
    out_x, out_y = out_x / norm, out_y / norm
    step = PUSHBUTTON_CLEARANCE_FT
    while step <= PUSHBUTTON_MAX_STEP_FT:
        candidate = (point[0] + out_x * step, point[1] + out_y * step)
        if not pavement.contains(_P(*candidate)):
            return candidate
        step += 0.5
    return None


def _pad_polygon(x: float, y: float, heading_deg: float):
    """The rectangle a tactile pad occupies - depth along the crossing, width across it.
    Shared by placement and by src/render/plan_view.py so both draw the same pad."""
    from shapely.geometry import Polygon

    angle = np.radians(heading_deg)
    ux, uy = np.cos(angle), np.sin(angle)
    nx, ny = -uy, ux
    half_d, half_w = TACTILE_PAD_DEPTH_FT / 2, TACTILE_PAD_WIDTH_FT / 2
    return Polygon([
        (x + ux * half_d + nx * half_w, y + uy * half_d + ny * half_w),
        (x - ux * half_d + nx * half_w, y - uy * half_d + ny * half_w),
        (x - ux * half_d - nx * half_w, y - uy * half_d - ny * half_w),
        (x + ux * half_d - nx * half_w, y + uy * half_d - ny * half_w),
    ])


def _kerb_tactile_pad_props(kerb_ways: list, crossings: list[dict], pavement, center_ft=None):
    """One tactile pad per ATTACH NODE - a node shared by a crossing way and a
    tactile_paving kerb way - deduplicated by node id.

    That single rule expresses both real-world cases, because the surveyor mapped the
    distinction into the topology:

      * One lowered kerb serving two crosswalks -> BOTH crossings attach at the SAME
        node -> the dedupe collapses them to ONE shared pad. (Three of the four corners
        at Broad & Greenwood.)
      * Two separate ramps on one lowered kerb -> the crossings attach at TWO DIFFERENT
        nodes on that kerb way -> TWO pads. (Every corner at Columbia & Princeton.)

    The kerb WAY is not the unit and must not be: at Columbia every corner is a single
    kerb way carrying two distinct pads, and at Broad & Greenwood a single kerb way
    carries one. Keying on the way collapsed Columbia's eight pads to four. The attach
    node is the ramp; the way is just the kerb it sits on.

    Returns (props, covered_crossing_ids) so the caller can skip crossing-inferred pads
    for crossings already served by a traced ramp.
    """
    # node id -> the kerb way it belongs to, for tactile kerbs near this junction only
    kerb_by_node: dict[int, dict] = {}
    for kerb in kerb_ways:
        coords = kerb.get("coords_wgs84")
        if not coords or kerb.get("tags", {}).get("tactile_paving") != "yes":
            continue
        kxs, kys = wgs84_to_state_plane.transform([c[0] for c in coords], [c[1] for c in coords])
        kerb_line = LineString(zip(kxs, kys))
        if center_ft is not None and kerb_line.distance(center_ft) > PAD_NEAR_JUNCTION_FT:
            continue  # a ramp at a neighbouring junction, pulled in by the context fetch radius
        for node_id in kerb.get("node_ids") or []:
            kerb_by_node[node_id] = {"kerb": kerb, "line": kerb_line}

    props = []
    covered_ways: set[int] = set()
    seen: set[int] = set()
    for crossing in crossings:
        for node_id, coord in zip(crossing.get("node_ids") or [], crossing.get("coords_wgs84") or []):
            entry = kerb_by_node.get(node_id)
            if entry is None:
                continue
            covered_ways.add(id(crossing))
            if node_id in seen:
                continue  # a second crossing on the SAME ramp - one pad, not two
            seen.add(node_id)

            x, y = wgs84_to_state_plane.transform(coord[0], coord[1])
            placed = _pad_orientation(x, y, entry["line"], pavement)
            if placed is None:
                print(f"  NOTE: ramp node {node_id} has tactile paving, but no pad can be placed clear "
                      f"of the modelled roadway - the modelled pavement covers the real footway here. "
                      f"Not drawn. Check this junction's widths and corner radii.")
                continue
            heading, pos = placed
            props.append({
                "type": "tactile_paving_pad", "position_ft": pos, "heading_deg": heading,
                "pad_depth_ft": TACTILE_PAD_DEPTH_FT, "pad_width_ft": TACTILE_PAD_WIDTH_FT,
                "source": f"real (OSM node {node_id}, where a crossing way meets barrier=kerb way "
                          f"{entry['kerb']['id']} tagged tactile_paving=yes): the surveyed curb ramp. "
                          f"One pad per attach node, so two crossings meeting at one node render as the "
                          f"single shared ramp they are. Approximation: pad size is a standard one.",
            })
    return props, covered_ways


def _pad_orientation(x: float, y: float, kerb_line, pavement):
    """(heading_deg, position) or None - where a pad goes at a ramp node on `kerb_line`.

    The pad's depth runs PERPENDICULAR TO THE KERB, into the footway, which is both
    physically right and the only direction that reliably leaves the roadway. An earlier
    version stepped outward from the crossing way's centroid instead; at nodes where that
    direction points along or into the junction it never escaped, leaving pads 100% inside
    the carriageway at 5 of 21 ramps.

    Returns None if the pad can't be cleared of the pavement within a sane distance. That
    means the modelled roadway has covered the real footway, and drawing the pad anyway
    would put it in the street - so it's dropped and reported instead.
    """
    coords = np.asarray(kerb_line.coords)
    node = np.array([x, y])
    # Local tangent from the two kerb vertices nearest this node.
    order = np.argsort(np.hypot(coords[:, 0] - x, coords[:, 1] - y))[:2]
    tangent = coords[order[1]] - coords[order[0]]
    norm = np.hypot(*tangent)
    if norm < 1e-6:
        return None
    normal = np.array([-tangent[1] / norm, tangent[0] / norm])

    for direction in (normal, -normal):
        heading = float(np.degrees(np.arctan2(direction[1], direction[0])))
        step = TACTILE_PAD_DEPTH_FT / 2
        while step <= PAD_MAX_STEP_FT:
            centre = node + direction * step
            pad = _pad_polygon(centre[0], centre[1], heading)
            if pavement is None or not pad.intersects(pavement):
                return heading, (float(centre[0]), float(centre[1]))
            step += 0.5
    return None


def _tactile_pad_props(line, pavement, leg_name: str, heading: float) -> list[dict]:
    """A detectable warning pad at each end of a crossing, wholly on the footway.

    Finding the roadway edge: take the parts of the crossing way that lie OUTSIDE the
    pavement polygon (line.difference(pavement)) - those are the footway approaches at
    either end - and put a pad at the inner end of each. Two earlier attempts got this
    wrong and are worth not repeating:

      * Intersecting with the leg's straight CURB LINE. Near a corner the curb return
        (fillet arc) bulges past where the straight curb runs, so those points were still
        inside the paved area: all eight pads at Broad/Greenwood landed 61-100% in the road.
      * Intersecting with the pavement boundary and stepping out a fixed half-depth. Better,
        but a fixed step doesn't clear a CURVED boundary, and it silently produced one pad
        per crossing instead of two wherever the boundary intersection wasn't a clean pair
        of points.

    So the pad is nudged outward until it genuinely clears the pavement, rather than by an
    amount assumed to be enough. If it can't be cleared within a sane distance the pad is
    dropped - better absent than drawn in the roadway.
    """
    if pavement is None:
        return []
    outside = line.difference(pavement)
    pieces = [outside] if outside.geom_type == "LineString" else list(getattr(outside, "geoms", []))
    mid = line.interpolate(0.5, normalized=True)

    props = []
    for piece in pieces:
        if piece.is_empty or piece.length < 0.5:
            continue
        # The end of this outside piece nearer the crossing's midpoint is the roadway edge;
        # the far end is out on the footway.
        ends = [Point(piece.coords[0]), Point(piece.coords[-1])]
        edge, far = sorted(ends, key=lambda pt: pt.distance(mid))
        out_x, out_y = far.x - edge.x, far.y - edge.y
        norm = np.hypot(out_x, out_y)
        if norm < 1e-6:
            continue
        out_x, out_y = out_x / norm, out_y / norm

        placed = None
        step = TACTILE_PAD_DEPTH_FT / 2
        while step <= TACTILE_PAD_DEPTH_FT * 3:
            cx, cy = edge.x + out_x * step, edge.y + out_y * step
            if not _pad_polygon(cx, cy, heading).intersects(pavement):
                placed = (cx, cy)
                break
            step += 0.5
        if placed is None:
            continue

        props.append({
            "type": "tactile_paving_pad", "position_ft": placed, "heading_deg": heading,
            "pad_depth_ft": TACTILE_PAD_DEPTH_FT, "pad_width_ft": TACTILE_PAD_WIDTH_FT,
            "source": f"real (OSM tactile_paving=yes on {leg_name}'s crossing): truncated-dome warning "
                      "surface at the curb ramp, placed where the surveyed crossing way leaves the paved "
                      f"area ({step:.1f} ft back onto the footway, far enough to clear the curb return). "
                      "Approximation: pad size is a standard one, not surveyed.",
        })

    if len(props) < 2:
        # A crossing runs sidewalk to sidewalk, so it should leave the paved area at BOTH
        # ends. Fewer means our modelled pavement has swallowed one or both ends - the
        # junction throat we build is wider than the real crossing is long. Same root
        # cause as the over-wide leg widths phase 2 checks against OSM's sidewalks, and
        # worth saying out loud rather than quietly drawing fewer ramps than exist.
        print(f"  NOTE: only {len(props)} tactile pad(s) placed on {leg_name} - its {line.length:.0f} ft "
              f"surveyed crossing does not clear the modelled pavement at both ends, so the modelled "
              f"junction is wider there than reality. Check this leg's curb_to_curb_ft and the corner radius.")
    return props


def _osm_crossing_hardware_props(state: DesignState, crossings: list[dict], nodes_ft: list[dict],
                                  kerb_ways: list | None = None, center_ft=None) -> list[dict]:
    """Pushbuttons, RRFBs and tactile paving pads for every crossing we can match to a leg.

    Reuses the same matcher the crosswalk geometry uses, so a crossing credited to a leg
    here is the same crossing drawn there - no second, differently-behaved association.
    """
    from src.render.crosswalks import match_crossing_lines_to_legs  # local: avoids an import cycle

    from src.geometry.model import build_pavement_polygon

    try:
        pavement = build_pavement_polygon(state.corner_fillets)
    except (ValueError, KeyError, StopIteration):
        # A junction whose pavement ring can't be built (see build_pavement_polygon's
        # acute-corner diagnosis) still gets its pushbuttons and beacons; only the pads,
        # which need the roadway edge, are skipped.
        pavement = None

    kerb_pads, covered_ways = _kerb_tactile_pad_props(kerb_ways or [], crossings, pavement, center_ft)

    # Which crossings already have their ramps placed from traced kerb geometry. Suppression
    # is PER CROSSING, not global: a junction can have some corners traced and some not, and
    # blanket-suppressing meant Columbia & Princeton showed 4 pads where all four crossings
    # are tagged tactile_paving=yes. Traced kerbs still win wherever they exist - they're
    # the more specific statement - but an untraced corner falls back to inference rather
    # than silently losing its ramp.
    props = list(kerb_pads)
    for leg_name, (line, crossing_tags) in match_crossing_lines_to_legs(state.legs, crossings).items():
        tags = _merged_crossing_tags(line, crossing_tags, nodes_ft)
        if _crossing_is_covered(crossings, covered_ways, line):
            tags = {k: v for k, v in tags.items() if k != "tactile_paving"}
        props += _crossing_endpoint_props(line, state.legs[leg_name], tags, leg_name, pavement)
    return props


def _crossing_is_covered(crossings: list[dict], covered_ways: set, line) -> bool:
    """Whether the crossing matching `line` already had pads placed from a traced kerb."""
    target = line.interpolate(0.5, normalized=True)
    best, best_d = None, None
    for crossing in crossings:
        coords = crossing.get("coords_wgs84") or []
        if len(coords) < 2:
            continue
        xs, ys = wgs84_to_state_plane.transform([c[0] for c in coords], [c[1] for c in coords])
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        d = np.hypot(cx - target.x, cy - target.y)
        if best_d is None or d < best_d:
            best, best_d = crossing, d
    return best is not None and id(best) in covered_ways


def osm_tree_points_ft(nodes_ft: list[dict]) -> list[tuple[float, float]]:
    """Positions of real OSM natural=tree nodes.

    Street trees used to be generated by walking each sidewalk piece at a fixed 25 ft
    spacing, which invented 6-24 trees per site - none of them recorded anywhere. They
    now come only from OSM, so a junction with no mapped trees renders with none.
    """
    return [(n["point_ft"].x, n["point_ft"].y) for n in nodes_ft
            if n["tags"].get("natural") == "tree"]


def _hydrant_props(nodes_ft: list[dict]) -> list[dict]:
    """Fire hydrants at their real OSM positions. Background detail, but real background
    detail - and a hydrant is one of the things that actually constrains where a curb
    extension or a parking stall can go."""
    props = []
    for node in nodes_ft:
        if node["tags"].get("emergency") != "fire_hydrant":
            continue
        props.append({
            "type": "fire_hydrant", "position_ft": (node["point_ft"].x, node["point_ft"].y),
            "heading_deg": 0.0,
            "surveyed_position": True,
            "source": "real (OSM emergency=fire_hydrant node): surveyed position.",
        })
    return props


SIGN_MAX_STEP_FT = 30.0  # past this the modelled pavement has swallowed the footway entirely


def _sign_offset_ft(state: DesignState, leg_name: str, requested_ft: float) -> float:
    """How far out a sign post on this leg has to stand: at least clear of the corner return.

    A sign placed at the crosswalk offset is level with the corner, where "sideways off this
    leg" points straight into the cross street - so stepping laterally never leaves the
    roadway, and the sign ends up standing in the middle of the junction. Beyond the return
    the only thing beside the leg is its own footway. This is where a sign post physically
    is, too: past the curve, not in it.
    """
    try:
        clearance_ft = leg_clearance_ft(leg_name, state.legs, state.corner_fillets)
    except (KeyError, ValueError):
        return requested_ft
    return max(requested_ft, clearance_ft)


def _leg_sign_position_ft(leg, offset_ft: float, side: str,
                           pavement=None) -> tuple[tuple[float, float], float]:
    """A point offset_ft along a leg's centerline from the intersection, pushed
    laterally past the curb (left or right, per `side`) onto the sidewalk.
    Returns (position, heading_deg) with heading pointing back toward the road.

    Half the NOMINAL width is only a first guess at where the curb is, and since the curb
    lines became the surveyor's traced kerbs it is frequently an underestimate - the real
    kerb flares wider than curb_to_curb_ft/2 approaching a corner, which is exactly where
    signs go. Trusting the nominal figure put NO TURN ON RED signs inside the carriageway
    at Broad & Greenwood. So the nominal offset is a starting point and the sign then steps
    outward until it is genuinely clear of the modelled roadway.
    """
    centerline = leg.centerline
    p = centerline.interpolate(min(offset_ft, centerline.length))
    p2 = centerline.interpolate(min(offset_ft + 1, centerline.length))
    u = np.array([p2.x - p.x, p2.y - p.y])
    u = u / np.linalg.norm(u)
    n = np.array([-u[1], u[0]]) if side == "left" else np.array([u[1], -u[0]])
    heading = float(np.degrees(np.arctan2(-n[1], -n[0])))  # face back toward the road

    # Measure from the CURB on this side, not from half the nominal width. Those are the
    # same number only where the curb is a centerline offset; where it is the surveyor's
    # traced kerb - which is everywhere now - the kerb flares past half-width approaching a
    # corner, and half-width put signs in the carriageway.
    base = np.array([p.x, p.y])
    curb = getattr(leg, f"{side}_curb", None)
    to_curb = curb.distance(p) if curb is not None and not curb.is_empty else leg.curb_to_curb_ft / 2
    lateral = to_curb + SIGN_SIDEWALK_SETBACK_FT
    pos = base + n * lateral
    pos = _step_outward_clear(base, n, lateral, pavement, extra_ft=SIGN_SIDEWALK_SETBACK_FT)
    return (tuple(pos) if pos is not None else None), heading


def _step_outward_clear(base, direction, lateral_ft: float, pavement, extra_ft: float = 0.0,
                         max_step_ft: float = SIGN_MAX_STEP_FT):
    """Push `base + direction * lateral_ft` further out until it clears the roadway.

    Used for anything whose position we DERIVE rather than read from a survey - signs,
    signal poles. A derived position that lands in the carriageway is our error to correct;
    a surveyed one is not ours to move (see checks.py's surveyed_furniture_in_roadway).
    """
    pos = base + direction * lateral_ft
    if pavement is None or pavement.is_empty:
        return pos
    step = 0.0
    while pavement.contains(Point(*pos)) and step < max_step_ft:
        step += 1.0
        pos = base + direction * (lateral_ft + step)
    if pavement.contains(Point(*pos)):
        # Nowhere along this ray is off the roadway. Rather than draw the thing in the
        # carriageway, return nothing and let the caller say so: an absent prop is an honest
        # "we could not place this", a prop standing in the road is a false claim.
        return None
    if step:  # clear of the kerb now, plus whatever footway setback we owed it
        pos = base + direction * (lateral_ft + step + extra_ft)
    return pos


# An approaching driver travels INWARD along a leg, i.e. opposite the centerline's own
# outward direction, so their right-hand side (where a US stop sign belongs) is the leg's
# own "left" in _leg_sign_position_ft's convention. This is the same swap
# blender_crosswalks.add_stop_bar documents for the stop bar. The original code placed
# stop signs on side="right" - the far side of the road from the driver they govern.
APPROACHING_DRIVER_RIGHT = "left"

# How far from the junction an OSM stop/give_way node may sit and still be governing THIS
# junction rather than a neighbouring one. Real nodes here sit 29.6-34.9 ft out; the next
# nearest belong to junctions 158 ft and 268 ft away.
STOP_NODE_MAX_ALONG_FT = 100.0
STOP_NODE_MAX_PERP_FT = 25.0

OSM_CONTROL_TO_PROP = {"stop": "stop_sign", "give_way": "yield_sign"}


def _osm_control_props(state: DesignState, nodes_ft: list[dict], pavement=None) -> list[dict]:
    """Stop / give-way signs at the approaches OSM actually records them on.

    A stop node sits ON the road way at the approach it governs, so matching it to a leg
    by perpendicular distance gives both WHICH approach has the sign and how far out it
    is - real surveyed facts, replacing the previous one-per-approach guess. The lateral
    side is still derived (a node on the centerline can't say which kerb the sign stands
    on), but it's derived correctly now, onto the approaching driver's right.

    Returns [] when OSM maps no control nodes at this junction, which is a genuine
    "no data" - the caller decides whether to fall back to a guess.
    """
    props = []
    for node in nodes_ft:
        prop_type = OSM_CONTROL_TO_PROP.get(node["tags"].get("highway"))
        if prop_type is None:
            continue
        point = node["point_ft"]
        best = None
        for leg_name, leg in state.legs.items():
            along = leg.centerline.project(point)
            perp = leg.centerline.interpolate(along).distance(point)
            if best is None or perp < best[0]:
                best = (perp, leg_name, along)
        perp, leg_name, along = best
        if along > STOP_NODE_MAX_ALONG_FT or perp > STOP_NODE_MAX_PERP_FT or along <= 0:
            continue  # governs a different junction, or isn't on any of our legs
        leg = state.legs[leg_name]
        pos, heading = _leg_sign_position_ft(leg, _sign_offset_ft(state, leg_name, along),
                                              side=APPROACHING_DRIVER_RIGHT, pavement=pavement)
        if pos is None:
            print(f"  NOTE: a sign for {leg_name} can't be placed clear of the modelled roadway. Not drawn.")
            continue
        props.append({
            "type": prop_type, "position_ft": pos, "heading_deg": heading,
            "source": f"real (OSM highway={node['tags'].get('highway')} node on {leg_name}, "
                      f"{along:.0f} ft from the junction): a surveyed control node, so WHICH approach "
                      f"is controlled and how far out is real data, not a guess. Approximation: the "
                      f"lateral side is derived (approaching driver's right) and the sign is set "
                      f"{SIGN_SIDEWALK_SETBACK_FT} ft back from the curb - the node itself sits on the "
                      f"road centerline and says nothing about either.",
        })
    return props


def _bollard_props(state: DesignState) -> list[dict]:
    """Plastic bollards (flex-post delineators) down the center of each
    lane-narrowing buffer for legs a scenario has explicitly added them to
    (src/geometry/treatments.py:add_bollards) - not a general per-site fact, only
    ever present when a proposal calls for this specific paint+bollard
    escalation."""
    props = []
    for leg_name, spacing_ft in state.bollard_lines.items():
        leg = state.legs[leg_name]
        stripe_width_ft = state.lane_narrowing[leg_name]
        start_ft = leg_clearance_ft(leg_name, state.legs, state.corner_fillets)
        for pos in bollard_points_ft(leg, stripe_width_ft, start_ft, spacing_ft):
            props.append({
                "type": "bollard", "position_ft": pos, "heading_deg": 0.0,
                "source": f"scenario-specified (add_bollards): flex-post delineator centered in {leg_name}'s "
                          f"painted lane-narrowing buffer (stripe_width_ft={stripe_width_ft:.1f}), spaced "
                          f"{spacing_ft:.0f} ft apart.",
            })
    return props


def _traffic_signal_props(model: IntersectionModel, state: DesignState, center_ft: Point,
                           pavement=None) -> list[dict]:
    """
    Traffic signal pole + pedestrian signal head at each corner listed in the
    site config's `signals.corners` (see sites/README.md) - confirmed via
    direct street-view photo review, NOT a field survey, but real/observed
    rather than a geometric placeholder. Pole position reuses the same real
    corner-fillet-arc-midpoint geometry the streetlights used to be placed on.

    The mast arm extends at a RIGHT ANGLE to leg_a - the one leg of this
    corner's pair whose LEFT curb feeds it (see build_corner_fillets /
    fillet_curb_corner: a corner is always (leg_a.left_curb, leg_b.right_curb),
    so the pole sits on leg_a's left side) - parallel to leg_a's own crosswalk
    and perpendicular to leg_a's direction of travel, reaching from the pole
    across to roughly mid-roadway. Confirmed against a real example: at the
    broad_st_west/greenwood_ave_north (NW) corner, leg_a is broad_st_west, and
    the arm reaches out over West Broad St's lanes, near-directly above its
    crosswalk - visible to a driver heading west on Broad St looking across
    the intersection - NOT diagonally toward the intersection center (an
    earlier version of this function had it reaching for the bisector of both
    adjacent legs, which is a ~45 degree angle relative to either road, not a
    real mast-arm layout). The vehicle head at the arm's end faces back down
    leg_a (toward oncoming leg_a-direction traffic) so it's actually visible
    to approaching drivers; exactly which approach's signal phase it
    represents isn't modeled, only the physical mast/head geometry.
    """
    signals_cfg = model.config.get("signals")
    if not signals_cfg:
        return []
    corner_cfg = {frozenset(c["legs"]): c for c in signals_cfg.get("corners", [])}
    confirmation = signals_cfg.get("source", "confirmed in site config.yaml (signals block)")

    props = []
    for (leg_a, leg_b), pieces in state.corner_fillets.items():
        if "error" in pieces:
            continue
        cfg = corner_cfg.get(frozenset((leg_a, leg_b)))
        if cfg is None:
            continue
        mid = pieces["arc"].interpolate(0.5, normalized=True)
        outward = np.array([mid.x - center_ft.x, mid.y - center_ft.y])
        norm = np.linalg.norm(outward)
        outward = outward / norm if norm > 1e-6 else np.array([1.0, 0.0])
        # A signal pole stands on the corner footway. The fillet arc midpoint is on the
        # kerb only when the corner is right; where a junction falls back to a fitted
        # fillet (W Broad & Louellen's acute Y), the arc cuts inside the real corner and a
        # fixed 4 ft setback leaves the pole standing in the carriageway.
        placed = _step_outward_clear(np.array([mid.x, mid.y]), outward,
                                      STREETLIGHT_SIDEWALK_SETBACK_FT, pavement)
        if placed is None:
            print(f"  NOTE: the signal pole for corner {leg_a}/{leg_b} can't be placed clear of the "
                  f"modelled roadway - the modelled pavement covers the real corner footway here. "
                  f"Not drawn.")
            continue
        pole_pos = tuple(placed)
        pole_heading = np.degrees(np.arctan2(outward[1], outward[0]))

        # leg_a's own outward direction - the axis the arm/crosswalk are actually
        # built around, not the corner's outward-from-center bisector above.
        leg_a_line = state.legs[leg_a].centerline
        c0, c1 = np.array(leg_a_line.coords[0]), np.array(leg_a_line.coords[1])
        u_a = (c1 - c0) / np.linalg.norm(c1 - c0)
        arm_dir = np.array([u_a[1], -u_a[0]])  # perpendicular to leg_a: from its left curb (the pole) across to its right
        arm_heading = np.degrees(np.arctan2(arm_dir[1], arm_dir[0]))
        head_facing = np.degrees(np.arctan2(-u_a[1], -u_a[0]))  # back down leg_a, toward the intersection
        arm_length_ft = state.legs[leg_a].curb_to_curb_ft / 2

        props.append({
            "type": "traffic_signal_pole", "position_ft": pole_pos, "heading_deg": head_facing,
            "arm_heading_deg": arm_heading, "arm_length_ft": arm_length_ft,
            "source": f"confirmed ({leg_a}/{leg_b} corner - {confirmation}): full-width mast-arm signal, "
                      "pole at the real corner-fillet arc midpoint; the arm extends at a right angle to "
                      f"{leg_a} (parallel to its crosswalk, perpendicular to its travel direction), reaching "
                      f"roughly to mid-roadway (arm_length_ft={arm_length_ft:.1f}, half of {leg_a}'s real "
                      "curb-to-curb width) - confirmed via street-view against a real example (NW corner's "
                      "arm over West Broad St), not a diagonal reach toward the intersection center. Exactly "
                      "which lane the head hangs over isn't surveyed.",
        })

        same_pole = cfg.get("pedestrian_head") == "same_pole"
        if same_pole:
            ped_pos, ped_heading = pole_pos, pole_heading
        else:
            tangent = np.array([-outward[1], outward[0]])
            ped_pos = (pole_pos[0] + tangent[0] * PED_HEAD_POLE_OFFSET_FT,
                       pole_pos[1] + tangent[1] * PED_HEAD_POLE_OFFSET_FT)
            ped_heading = pole_heading
        props.append({
            "type": "pedestrian_signal_head", "position_ft": ped_pos, "heading_deg": ped_heading,
            "own_post": not same_pole,
            "source": f"confirmed ({leg_a}/{leg_b} corner - {confirmation}): " + (
                "pedestrian head mounted on the same pole as the vehicle signal."
                if same_pole else
                "pedestrian head is on a SEPARATE pole from the vehicle signal; approximation: offset "
                f"{PED_HEAD_POLE_OFFSET_FT} ft along the sidewalk from the vehicle signal pole (no "
                "surveyed separate-pole location available)."
            ),
        })
    return props


def _no_turn_on_red_props(model: IntersectionModel, state: DesignState, offsets_ft: dict,
                           pavement=None) -> list[dict]:
    """NO TURN ON RED restriction signs for the legs listed in the site config's
    `signals.no_turn_on_red_legs` (confirmed via street-view photo review, not
    a signage-inventory survey). Positioned the same way as the automatic
    per-approach stop sign (_stop_sign_props) - same placement approximation."""
    signals_cfg = model.config.get("signals")
    if not signals_cfg:
        return []
    props = []
    for leg_name in signals_cfg.get("no_turn_on_red_legs", []):
        leg = state.legs.get(leg_name)
        if leg is None:
            continue
        offset_ft = offsets_ft[leg_name][0]
        pos, heading = _leg_sign_position_ft(leg, _sign_offset_ft(state, leg_name, offset_ft),
                                              side=APPROACHING_DRIVER_RIGHT, pavement=pavement)
        if pos is None:
            print(f"  NOTE: a sign for {leg_name} can't be placed clear of the modelled roadway. Not drawn.")
            continue
        props.append({
            "type": "no_turn_on_red_sign", "position_ft": pos, "heading_deg": heading,
            "source": "confirmed (street-view photo review, site config.yaml signals.no_turn_on_red_legs) "
                      "that no-turn-on-red signage exists on this approach; placement approximation: same "
                      "near-corner curb-line pattern as _stop_sign_props (not a real traffic-engineering "
                      "placement study).",
        })
    return props


def _extra_props_from_config(model: IntersectionModel, state: DesignState, offsets_ft: dict,
                              pavement=None) -> list[dict]:
    """User-specified extra signage (e.g. a school zone sign) from the site's
    config.yaml `props.extra` list - explicitly site-specific knowledge that
    doesn't belong in the general pipeline. See sites/README.md."""
    props = []
    for entry in model.config.get("props", {}).get("extra", []):
        leg = state.legs.get(entry["leg"])
        if leg is None:
            continue
        offset_ft = entry.get("offset_ft", offsets_ft.get(entry["leg"], (10, ""))[0])
        pos, heading = _leg_sign_position_ft(leg, _sign_offset_ft(state, entry["leg"], offset_ft),
                                              side=entry.get("side", "right"), pavement=pavement)
        if pos is None:
            print(f"  NOTE: the configured {entry['type']} on {entry['leg']} can't be placed clear of the "
                  f"modelled roadway. Not drawn.")
            continue
        props.append({
            "type": entry["type"], "position_ft": pos, "heading_deg": heading,
            "source": f"user-specified in site config.yaml (props.extra): {entry.get('note', 'no note given')}",
        })
    return props


def _extra_props_from_state(state: DesignState, offsets_ft: dict, pavement=None) -> list[dict]:
    """Scenario-specific extra signage added by a treatment
    (src/geometry/treatments.py:add_extra_prop) - e.g. an RRFB or a relocated
    school-zone sign that only exists in one particular proposal, not the
    site's baseline config (see _extra_props_from_config for the site-wide
    equivalent)."""
    props = []
    for entry in state.extra_props:
        leg = state.legs.get(entry["leg"])
        if leg is None:
            continue
        # offset_ft may be explicitly None (see add_extra_prop) - `or` (not .get's
        # default) is required to fall through to the real crosswalk offset in that case.
        offset_ft = entry.get("offset_ft") or offsets_ft.get(entry["leg"], (10, ""))[0]
        pos, heading = _leg_sign_position_ft(leg, _sign_offset_ft(state, entry["leg"], offset_ft),
                                              side=entry.get("side", "right"), pavement=pavement)
        if pos is None:
            print(f"  NOTE: the configured {entry['type']} on {entry['leg']} can't be placed clear of the "
                  f"modelled roadway. Not drawn.")
            continue
        props.append({
            "type": entry["type"], "position_ft": pos, "heading_deg": heading,
            "source": f"scenario-specified (treatment-level prop, not site config): {entry.get('note') or 'no note given'}",
        })
    return props


# Bollards the TREATMENT layer already draws for itself. The plan view builds its markings
# from src/geometry/paint.py, which emits a bollard piece for every leg in
# state.bollard_lines, so drawing these props on top would just thicken the same markers.
# Tagged rather than inferred: the plan view used to skip EVERY bollard prop on that
# reasoning, which silently dropped the daylight-zone bollards - they come only from props,
# so the 2D view of the bollard proposals showed no bollards at all while the 3D render
# showed thirteen. Exactly the 2D/3D disagreement this project cannot ship.
DRAWN_BY_PAINT = "drawn_by_paint"


def _parking_buffer_bollard_props(state: DesignState) -> list[dict]:
    """Plastic bollards centered in the striped no-parking buffer between a
    marked-parking lane and the curb (src/geometry/treatments.py:
    add_parking_buffer_bollards) - the same bollard prop type _bollard_props
    uses (just on the curb side of a parking lane instead of the travel-lane
    side of a lane-narrowing buffer), so both render identically in 3D via
    the one add_bollard() builder."""
    props = []
    for (leg_name, side), spacing_ft in state.parking_buffer_bollards.items():
        leg = state.legs[leg_name]
        curb_offset_ft = state.parking_zones[(leg_name, side)]["curb_offset_ft"]
        start_ft = leg_clearance_ft(leg_name, state.legs, state.corner_fillets)
        for pos in bollard_points_ft(leg, curb_offset_ft, start_ft, spacing_ft, sides=(side,)):
            props.append({
                "type": "bollard", "position_ft": pos, "heading_deg": 0.0,
                DRAWN_BY_PAINT: True,
                "source": f"scenario-specified (add_parking_buffer_bollards): flex-post delineator centered in "
                          f"{leg_name}'s {side} striped buffer between its marked-parking lane and the curb "
                          f"(curb_offset_ft={curb_offset_ft:.1f}), spaced {spacing_ft:.0f} ft apart.",
            })
    return props


def _daylight_device_props(state: DesignState, offsets_ft: dict, so_far: list[dict]) -> list[dict]:
    """Bollards standing in a daylight zone (treatments.protect_daylight_zone).

    Placed along the zone's LANE edge, which is the side that needs protecting - an object
    against the kerb protects nothing. Spaced along the statutory no-parking span itself, so
    the devices end exactly where parking is allowed to begin and the two never overlap.

    `so_far` is the props already built, because the span depends on them: a hydrant or a
    stop sign carries its own setback (R.S. 39:4-138(h),(i)) and lengthens the zone.
    """
    import numpy as np

    from src.geometry.daylighting import merged_no_parking_spans_ft, no_parking_zones_ft
    from src.geometry.model import _point_at, leg_clearance_ft
    from src.geometry.paint import LANE_EDGE_LINE_WIDTH_FT
    from src.geometry.treatments import TARGET_LANE_WIDTH_FT

    MIN_DAYLIGHT_DEVICE_SPAN_FT = 3.0
    props = []
    for (leg_name, side), device in sorted(state.daylight_devices.items()):
        # A zone shorter than this cannot hold even one device clear of the crossing.
        leg = state.legs.get(leg_name)
        if leg is None or leg_name not in offsets_ft:
            continue
        spacing_ft = device["spacing_ft"]
        sign = 1 if side == "left" else -1
        # Just outside the lane edge line, i.e. the first thing a driver would clip.
        offset_ft = sign * (TARGET_LANE_WIDTH_FT + LANE_EDGE_LINE_WIDTH_FT * 1.5)
        clearance_ft = leg_clearance_ft(leg_name, state.legs, state.corner_fillets)
        spans = merged_no_parking_spans_ft(
            no_parking_zones_ft(state, leg_name, side, offsets_ft, so_far))
        for start_ft, end_ft in spans:
            start_ft = max(start_ft, clearance_ft)
            span_ft = end_ft - start_ft
            if span_ft < MIN_DAYLIGHT_DEVICE_SPAN_FT:
                continue
            # Distributed across the span rather than stepped from its start, so the row
            # ends where the zone does, and a zone shorter than one spacing still gets one
            # device rather than none.
            count = max(int(span_ft // spacing_ft), 1)
            for station in np.linspace(start_ft, end_ft, count + 1)[:-1] + (span_ft / count) / 2:
                props.append({
                    "type": "bollard",
                    "position_ft": tuple(_point_at(leg.centerline, float(station), offset_ft)),
                    "heading_deg": 0.0,
                    "source": f"scenario-specified (protect_daylight_zone): {device['kind']} "
                              f"in {leg_name}'s {side} daylight zone, {spacing_ft:.0f} ft apart.",
                })
    return props


def build_props(model: IntersectionModel, state: DesignState, offsets_ft: dict, center_ft: Point,
                 traffic_control: list[dict] | None = None, street_furniture: list[dict] | None = None,
                 crossings: list[dict] | None = None, kerb_ways: list | None = None) -> list[dict]:
    """All street-furniture props for one scenario export: a streetlight at every corner
    (always), the junction's traffic control, and any site- or scenario-specific extras.

    NOTHING IS DRAWN THAT ISN'T ATTESTED. A prop appears only if its EXISTENCE is
    recorded either in OSM or in the site config's own observations. There is no
    "plausible default" tier any more: a junction with no mapped street lamps renders
    with no street lamps, and a junction with no mapped stop nodes and no configured
    signals renders with no traffic control at all. An empty corner is an honest
    statement that nobody has recorded what is there; a fabricated lamp is not, and it
    is indistinguishable in the render from a surveyed one.

    Two sources, in precedence order:

    1. OBSERVED - the site config's `signals` block. Direct street-view/field
       observation, and the only source that says where each pole and pedestrian head
       actually is. Supersedes OSM (src/provenance.py: if OSM disagrees with something
       we looked at, OSM is wrong).
    2. OSM - surveyed nodes and ways: highway=stop / give_way, highway=street_lamp,
       emergency=fire_hydrant, and the crossing tags below.

    Positions may still be DERIVED where the source records existence but not placement
    (a stop node sits on the road centerline and can't say which kerb its sign is on).
    That is different from inventing the thing itself, and every such case says so in its
    `source` string. What was removed: a streetlight conjured at every corner, and an
    all-way stop assumed whenever no signals were configured.

    Everything else OSM knows about the junction is used too, all from surveyed positions:
    pedestrian pushbuttons and RRFB beacons at the ends of the crossing ways that carry
    button_operated / flashing_lights, tactile paving pads where a tactile_paving crossing
    meets each curb line, and fire hydrants. Nothing OSM records here is discarded.

    Streetlights follow the same idea: real highway=street_lamp nodes when OSM has them,
    otherwise one derived per corner. None of this project's four sites has any mapped,
    so all four currently use the derived fallback - see data_gaps().

    Signals and stop signs are not mutually exclusive any more: a signalized junction can
    still have a stop sign on a minor approach, and OSM will say so if it does. What the
    code no longer does is invent an all-way stop just because no signals were configured.
    """
    signalized = bool(model.config.get("signals"))
    furniture_ft = control_nodes_ft(street_furniture)  # same lon/lat -> point_ft conversion
    control_ft = control_nodes_ft(traffic_control)
    # The real modelled roadway, so every sign can be placed clear of IT rather than clear
    # of a nominal half-width that the traced kerb often exceeds near a corner.
    try:
        pavement = build_pavement_polygon(state.corner_fillets)
    except ValueError:
        pavement = None
    props = (
        _osm_streetlight_props(furniture_ft)
        + _osm_control_props(state, control_ft, pavement)
        + _osm_crossing_hardware_props(state, crossings or [], control_ft, kerb_ways, center_ft)
        + _hydrant_props(furniture_ft)
        + _traffic_signal_props(model, state, center_ft, pavement)
        + _no_turn_on_red_props(model, state, offsets_ft, pavement)
        + _extra_props_from_config(model, state, offsets_ft, pavement)
        + _extra_props_from_state(state, offsets_ft, pavement)
        + _bollard_props(state)
        + _parking_buffer_bollard_props(state)
    )
    # Last, and passed everything above: a daylight zone's length depends on the hydrants and
    # stop signs already placed, which carry setbacks of their own under 39:4-138(h),(i).
    return props + _daylight_device_props(state, offsets_ft, props)


# The pad/furniture-in-the-roadway invariant moved to src/checks.py, where it runs
# alongside every other scene invariant and reports all violations at once instead of
# stopping at the first. Re-exported so existing imports keep working.
from src.checks import (  # noqa: E402
    MAX_PAD_ROADWAY_OVERLAP,
    PedestrianFurnitureInRoadwayError,
    TactilePadInRoadwayError,
    check_furniture_off_roadway,
)

PAD_MAX_STEP_FT = 12.0   # past this, the modelled pavement has swallowed the footway
PAD_NEAR_JUNCTION_FT = 90.0  # a ramp belonging to THIS junction; crossings/kerbs are fetched
                              # over a much wider radius for context, and three of Columbia &
                              # Princeton's 'ramps' were 350+ ft away at a different junction


def assert_pads_off_roadway(props: list[dict], pavement) -> None:
    """Back-compat shim for the narrower original check. Prefer checks.assert_scene_valid."""
    violations = check_furniture_off_roadway(props, pavement)
    if violations:
        raise PedestrianFurnitureInRoadwayError(
            "\n  ".join(["pedestrian furniture in the modelled roadway:"] + [str(v) for v in violations]))


def data_gaps(traffic_control: list[dict] | None, street_furniture: list[dict] | None,
               signalized: bool = False) -> list[str]:
    """Describe what this junction is being DERIVED rather than sourced, so a gap in OSM
    is visible in the phase output instead of silently becoming a confident-looking prop.

    Every item here is a concrete invitation to improve the render by improving OSM.
    """
    gaps = []
    if not any(n["tags"].get("highway") == "street_lamp" for n in (street_furniture or [])):
        gaps.append("no highway=street_lamp nodes mapped - NO streetlights are drawn. Map them in OSM "
                    "to have them appear.")
    has_signs = any(n["tags"].get("highway") in ("stop", "give_way") for n in (traffic_control or []))
    if not has_signs and not signalized:
        # Only a gap where it changes what gets drawn. At a signalized junction with no
        # mapped signs, nothing is guessed - build_props draws no signs at all.
        gaps.append("no highway=stop/give_way nodes mapped and no `signals` block configured - "
                    "NO traffic control is drawn at all. Map the control in OSM, or record it in "
                    "the site config.")
    # tactile_paving / button_operated live on the highway=crossing NODES, which
    # fetch_traffic_control returns - NOT on the crossing ways from fetch_crossings.
    crossing_nodes = [n for n in (traffic_control or []) if n["tags"].get("highway") == "crossing"]
    if not crossing_nodes:
        gaps.append("no highway=crossing nodes mapped - no pedestrian-facing crossing detail "
                    "(ADA tactile paving, pushbuttons, refuge islands) is available.")
    elif not any(n["tags"].get("tactile_paving") for n in crossing_nodes):
        gaps.append(f"none of the {len(crossing_nodes)} crossing nodes is tagged tactile_paving - ADA ramp "
                    "presence is unknown here (the crosswalk inventory's LIMITATIONS.md section 6 flags "
                    "the same gap).")
    return gaps


# fire_hydrant:position values that assert the hydrant is NOT in the carriageway. `lane`
# says it genuinely is, so a hydrant tagged that way is left alone.
HYDRANT_OFF_ROADWAY_POSITIONS = ("sidewalk", "green", "parking_lot", "roof")


def hydrant_position_conflicts(street_furniture: list[dict] | None, pavement) -> list[str]:
    """Cross-check each hydrant's fire_hydrant:position tag against our pavement polygon.

    A hydrant tagged sidewalk/green/parking_lot that nonetheless falls inside the modelled
    roadway is a real contradiction between two sources, and it has exactly two possible
    causes - both worth knowing:

      * OSM's node is in the wrong place. That was the case at Broad & Greenwood, where the
        hydrant was tagged fire_hydrant:position=green while its coordinates put it in the
        street; it rendered in the carriageway because we place it where OSM says it is.
      * Our modelled roadway is too wide and has swallowed the footway it stands on - the
        same failure the sidewalk width check and the tactile pad placement both surface.

    Nothing is moved either way. A hydrant is drawn at its surveyed position, full stop;
    this only reports the disagreement. A hydrant with no position tag can't be checked, so
    it's reported separately rather than being assumed correct.
    """
    if pavement is None:
        return []
    notes = []
    for node in control_nodes_ft(street_furniture):
        if node["tags"].get("emergency") != "fire_hydrant":
            continue
        if not pavement.contains(node["point_ft"]):
            continue
        position = node["tags"].get("fire_hydrant:position")
        if position in HYDRANT_OFF_ROADWAY_POSITIONS:
            notes.append(f"hydrant tagged fire_hydrant:position={position} sits INSIDE the modelled "
                          f"roadway. Either the OSM node needs moving onto the footway, or this "
                          f"junction's modelled width has swallowed it.")
        elif position == "lane":
            pass  # genuinely in the carriageway; OSM and the model agree
        else:
            notes.append("hydrant with no fire_hydrant:position tag sits inside the modelled roadway - "
                          "can't tell whether the node is misplaced or our road is too wide. Tag it "
                          "(sidewalk / green / lane) to make this checkable.")
    return notes


def signalization_conflicts(model: IntersectionModel, traffic_control: list[dict] | None) -> list[str]:
    """Cross-check the config's observed signal state against OSM's, and describe any
    disagreement. Purely advisory - the observation wins either way (see build_props) -
    but a silent disagreement between two sources is worth surfacing, the same way
    phase2 reports a field-measured width that OSM's sidewalks contradict.
    """
    configured = bool(model.config.get("signals"))
    osm_signalled = any(n["tags"].get("highway") == "traffic_signals" for n in (traffic_control or []))
    if configured and not osm_signalled:
        return ["config declares a `signals` block but OSM maps no traffic_signals node here. "
                "The observation stands; consider adding the signal to OSM."]
    if osm_signalled and not configured:
        return ["OSM maps a traffic_signals node here but the site config has no `signals` block, "
                "so no signal hardware will be drawn. Confirm by street view and add the block."]
    return []
