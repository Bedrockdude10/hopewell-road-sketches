"""WHERE THE KERBSIDE MARKINGS OPEN FOR A VEHICLE, and what a marking does across the gap.

A driveway, a cross-street mouth and this junction's own throat are all holes in the kerbside
paint, but they are NOT the same hole and the difference is visible on the sheet: a driveway has an
apron, so the hatching ends on the apron's fillet; a street mouth has none, so it ends square. Kept
apart by kind for exactly that reason - see KerbSideOpenings.

THE DOTTED EXTENSION LIVES HERE because it is what a marking becomes at an opening: MUTCD dotted
marks carry the lane across a mouth where a solid line would tell a driver the wrong thing.
"""
import math
from dataclasses import dataclass, field
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union
from src.geometry.model import (band_from_offsets, corner_apron_annulus, corner_overlay_polygon,
                                curb_offsets_at_stations, paint_stations, point_at,
                                station_offset_many)
from src.render.crosswalks import crosswalk_reach_on_leg_side_ft
from src.geometry.paint.pieces import LANE_EDGE_LINE_WIDTH_FT
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: these types are layered above this module,
    # so importing them for real would close a cycle.
    from shapely.geometry import Point
    from src.geometry.treatments.state import DesignState

# How far an opening's ends are trimmed back, with a rounded corner, past the dropped kerb's own
# extent. A driveway apron flares at the kerb in reality and a car turning in cuts the corner.
# Kept small on purpose: every foot of trim is a foot of bike lane or hatched buffer given up.
# Not a swept-path figure - see kerb_opening_bands.
OPENING_TRIM_FT = 1.5

# HOW FAR PAST THE KERB AN OPENING'S CUT REACHES. Not a design figure at all - it is the margin
# that makes the cut's outer edge stop BEING a description of the kerb, and it is here because
# every description of the kerb disagrees with every other in the third decimal place. See
# _kerbside_cut for the two constructions that were tried instead and what each left standing.
#
# BOUNDED FROM BOTH SIDES, which is why this value and not a rounder one. Above: it has to clear
# the largest disagreement measured between two honest descriptions of one kerb, 0.071 ft, with
# room over. Below: the cut is clipped per station to the kerb plus this, so this is also how far
# the trim's round buffer may bulge past the kerb, and checks.PaintOverTheCurb calls 0.25 ft past
# the tracing a fatal violation. 0.15 is twice the first and three fifths of the second. There is
# no paint out there for the margin itself to remove - PaintInsideTheTracedKerb is fatal too.
OPENING_PAST_THE_KERB_FT = 0.15

# A HATCHED zone ends at an opening on an arc that LEAVES ITS OWN EDGE LINE TANGENTIALLY and
# curves out to the kerb - a fillet, not a chamfer and not a bulge. That tangency is the whole
# difference between a line and a cut: the white line beside the travel lane runs straight, peels
# away in one sweep around the driveway apron, and comes back as one continuous stroke. Do not
# make the arc tangent to the TRANSVERSE direction instead; that is flat where the eye follows the
# edge line and is the blunt end this exists to fix.
#
# The radius is the depth of the strip being closed, expressed per unit depth so a shallow strip
# gets a short sweep and a deep one a long one - the run and the depth are the same measurement
# seen twice, so there is nothing to tune per site. It costs HATCHING and nothing else.
OPENING_FILLET_PER_DEPTH = 1.0

# The dotted extension a lane line becomes where it crosses an opening, in feet. MUTCD's dotted
# lane extension is a 2 ft segment with a 2-6 ft gap; the TIGHT end of that range is used because
# a driveway mouth is short - E Broad's openings run 4-37 ft, and a 2+6 pattern would put a single
# dash in a 10 ft one, which reads as a stray mark rather than as a line continuing.
DOTTED_MARK_FT = 2.0
DOTTED_GAP_FT = 2.0


# How far past the nominal half-width a dash's station band reaches before the opening's own
# polygon bounds it laterally. Generous on purpose - the band is a STATION filter, and one that
# stopped at the nominal kerb would clip the outer end off every mark on a leg whose traced kerb
# flares, which approaching a corner is every one of them.
DASH_BAND_REACH = 3.0
DASH_BAND_MARGIN_FT = 20.0


def _station_band(leg, start_ft: float, end_ft: float):
    """The band right across a leg between two stations - one mark's worth of ground.

    Deliberately NOT offset_band_polygon, which clamps its offsets to the traced kerb: this is a
    station filter and not a lateral one. The clip against the opening's polygon is what bounds it
    laterally.

    Sampled along the centreline rather than taken as one rectangle, so a dash laid on a bending
    leg sits on the road rather than cutting the corner of it.
    """
    length_ft = leg.centerline.length
    lo, hi = max(min(start_ft, length_ft), 0.0), max(min(end_ft, length_ft), 0.0)
    if hi - lo < 1e-6:
        return None
    reach_ft = abs(leg.curb_to_curb_ft or 0.0) * DASH_BAND_REACH + DASH_BAND_MARGIN_FT
    stations = np.linspace(lo, hi, max(int((hi - lo) / DASH_BAND_STEP_FT) + 2, 2))
    left = [point_at(leg.centerline, float(s), reach_ft) for s in stations]
    right = [point_at(leg.centerline, float(s), -reach_ft) for s in stations]
    band = Polygon([*left, *reversed(right)])
    if not band.is_valid:
        band = band.buffer(0)
    return None if band.is_empty else band


# How finely the station band is sampled along the centreline. Well under a dash's own length, so
# a bend inside one mark is followed rather than chorded.
DASH_BAND_STEP_FT = 1.0

# How far past an opening a marking has to go on before it counts as CROSSING the opening rather
# than ending at it - see PaintContext._dash_spans_along. A dotted extension is only for the
# first. Half a mark: below that there is nothing on the far side to continue into.
DASH_CROSSING_SLACK_FT = 1.0


def _dash_spans(lo_ft: float, hi_ft: float) -> list[tuple[float, float]]:
    """`lo_ft`..`hi_ft` cut into MUTCD dotted-extension marks, centred in it.

    STATIONS, not distance along one line, because everything the dashes carry across an opening
    has to break at the SAME places: the lane's two edge lines and the green between them are one
    marking seen three ways, and dashing each along its own arc length puts them out of phase - by
    little on a straight leg and visibly on a curved one, where the inner and outer stripes have
    different lengths through the same mouth.

    Centred rather than started at one end so the pattern reads as deliberate: an opening is only a
    few marks long, and one clipped to a stub at the far end looks like a striping error. The count
    comes out of the length, so a wide entrance gets more marks rather than longer ones.
    """
    length_ft = hi_ft - lo_ft
    if length_ft < DOTTED_MARK_FT:
        return []
    period = DOTTED_MARK_FT + DOTTED_GAP_FT
    n = max(1, int(round((length_ft + DOTTED_GAP_FT) / period)))
    span = n * DOTTED_MARK_FT + (n - 1) * DOTTED_GAP_FT
    while n > 1 and span > length_ft:
        n -= 1
        span = n * DOTTED_MARK_FT + (n - 1) * DOTTED_GAP_FT
    start_ft = lo_ft + max((length_ft - span) / 2, 0.0)
    return [(start_ft + i * period, start_ft + i * period + DOTTED_MARK_FT) for i in range(n)]


# How finely the kerb is sampled when holding a rim inside it. Well under STRIP_SAMPLE_FT,
# because a corner return curves through most of its bearing inside two feet and a chord across
# that bulges OUTSIDE the kerb it is supposed to bound - which is the one direction that matters
# here (checks.PaintInsideTheCurb allows 0.25 ft and the chord let 0.46 ft through at W Broad &
# Louellen's north corner). Only the rim pays for the finer grid; nothing else is held this way.
KERB_HOLD_SAMPLE_FT = 0.5


def _held_inside_the_kerb(leg, side: str, line: LineString):
    """`line` with every vertex pulled back to the traced kerb, measured as the CHECK measures it.

    The band intersection above holds a rim inside the kerb as a REGION, and a region has to pick
    a representation: it follows the kerb's own coordinates, straight from vertex to vertex in
    world space. checks.PaintInsideTheTracedKerb instead reads each drawn vertex's own station and
    interpolates the kerb's OFFSET there. The two are the same curve only where the centreline is
    straight, and on louellen_st_west's bend they differ by 0.34 ft - enough to fail a 0.25 ft
    tolerance on paint the region clamp thought it had already held.

    So the last word goes to the frame the invariant is stated in. Vertex by vertex, no region: a
    marking may meet the kerb, never cross it, and "cross it" means what the check means by it.
    """
    from src.geometry.model import place_in_measured_frame

    if leg is None or side is None or line.is_empty or line.geom_type != "LineString":
        return line
    points = np.asarray(line.coords, dtype=float)
    stations, offsets = station_offset_many(leg.centerline, points)
    curb = curb_offsets_at_stations(leg, side, stations)
    if curb is None:
        return line
    sign = 1.0 if str(side) == "left" else -1.0
    room = np.maximum(np.abs(curb) - LANE_EDGE_LINE_WIDTH_FT / 2, 0.0)
    over = np.abs(offsets) > room
    if not over.any():
        return line
    offsets[over] = sign * room[over]
    return LineString(place_in_measured_frame(leg.centerline, stations, offsets))


def _inside_the_traced_kerb(leg, side: str, near):
    """The strip from the centreline out to the TRACED kerb, over `near`'s own extent.

    Not curbside_strip_polygon: that function builds its grid from model.paint_stations at
    STRIP_SAMPLE_FT, which is right for a marking running the length of a leg but wrong for holding
    a line against a kerb bending through a corner return. Sampled here at KERB_HOLD_SAMPLE_FT
    over just the span being held, so the chord error is a sixteenth of what it was.

    THE OUTER EDGE IS THE KERB'S OWN COORDINATES, not that sampling of them - the same rule
    model.curbside_strip_polygon states for the same reason. Resampled, the chord between two grid
    stations lies OUTSIDE a kerb that curves inward between them, so the band leaked exactly where
    it is needed most: at a driveway opening's fillet, where the kerb turns hardest. That let an
    opening rim stand 0.4 ft past the traced kerb on louellen_st_west at 2.5x and
    checks.PaintInsideTheTracedKerb refuse the export.
    """
    from src.geometry.model import curb_edge_by_station, point_at

    coords = [xy for part in getattr(near, "geoms", [near])
              if not part.is_empty and part.geom_type in ("LineString", "Polygon")
              for xy in (part.exterior.coords if part.geom_type == "Polygon" else part.coords)]
    if not coords:
        return None
    stations, _offsets = station_offset_many(leg.centerline, np.asarray(coords, dtype=float))
    lo = max(float(stations.min()) - KERB_HOLD_SAMPLE_FT, 0.0)
    hi = min(float(stations.max()) + KERB_HOLD_SAMPLE_FT, leg.centerline.length)
    if hi - lo < KERB_HOLD_SAMPLE_FT:
        return None
    grid = np.linspace(lo, hi, max(int((hi - lo) / KERB_HOLD_SAMPLE_FT) + 1, 2))
    outer = curb_edge_by_station(leg, side, lo, hi)
    if outer is None:
        return None
    inner = [point_at(leg.centerline, float(station), 0.0) for station in grid]
    band = Polygon(list(outer) + list(reversed(inner)))
    if not band.is_valid:
        band = band.buffer(0)
    return None if band.is_empty else band


def junction_mouths_ft(state: "DesignState", crosswalk_bands: dict | None = None) -> dict[tuple[str, str], tuple[float, float]]:
    """{(leg, side): (0.0, end_ft)} - where THIS junction opens each kerb.

    THE INTERSECTION ENDS AT THE CROSSWALK, and that is the whole rule. A person reads a junction
    by its crosswalks: the box between them is the intersection, and the corner OUTSIDE a crosswalk
    is approach - the ground a painted curb extension is put on. So the mouth ends at that
    crossing's reach along this kerb, and only a leg with no crossing resolved at all falls back
    to the corner return's tangent point.

    PAINTED OR NOT, and that distinction has no business here. Whether a crossing is MARKED is a
    fact about paint - it decides what other paint has to keep clear of, which is `keep_clear`'s
    job. Where the junction ends is a fact about the street, and an unmarked approach still has a
    crosswalk on it: N.J.S.A. 39:1-1's, the one daylighting.py already measures R.S. 39:4-138(e)
    from by name on exactly these legs. Its position is context.crosswalk_estimate_ft, which
    reproduces all 11 surveyed crossings here to a standard deviation of 2.4 ft - and which
    rejected the fillet tangent point for this very job, at -31.5 to +41.7 ft of scatter. So the
    estimate is not the weaker answer being used for want of a survey; on this question it is the
    stronger of the two, and the corner return is the fallback.

    WHY NOT THE CORNER RETURN EVERYWHERE. The tangent point is where the KERB starts, and a
    hatched no-parking zone held back to it stops short of the crossing - which undoes the
    treatment, because the bare stretch beside a crossing is the parking space daylighting exists
    to remove, and because filling that corner IS the painted curb extension. It costs 15.3 ft of
    hatching on W Broad & Louellen's south kerb and 13.2 ft on Greenwood Ave north's.

    AND THE TANGENT POINT DIVERGES AS A CORNER SHARPENS, which is why holding paint back to it is
    not merely conservative. A fillet of radius R meets its kerbs R/tan(theta/2) back from the
    corner, so the same corner radius reads as 1.0 R at a square junction and 2.5 R at W Broad &
    Louellen's 44 deg Y: 63.7 ft along W Broad's northwest kerb, against a crossing that reaches
    32.1 ft. That left 31.7 ft of the statutory zone bare on the sharpest corner at any of these
    sites - the one where a parked car blocks the most sight line.

    IT CAN ALSO MOVE THE MOUTH OUTWARD, which is the same rule and not a separate one: at
    W Broad & Louellen the crossing is surveyed 43.7 deg off square, so on Louellen's NORTH kerb
    it reaches station 25.4 against a tangent point at 7.9. The 17.5 ft between them is on the
    junction side of the crosswalk however short the corner is, and paint there is paint in the
    intersection.

    Empty for a kerb that runs straight through, via junction_mouth_ft - MUTCD 3B.11(07)'s
    T-intersection exception, falling out of the geometry rather than written as a rule.
    """
    from src.geometry.model import junction_mouth_ft
    from src.geometry.treatments import travel_lane_edge_ft

    bands = crosswalk_bands or {}
    out = {}
    for leg_name, leg in state.legs.items():
        band = bands.get(leg_name)
        for side in ("left", "right"):
            reach_ft = None
            if band is not None and not band.is_empty and leg.curb_to_curb_ft is not None:
                # THE STRIP THIS KERB'S PAINT OCCUPIES, which is the same restriction
                # leg_anchors makes and for the same reason: a skewed band reaches further along
                # the leg near the centreline than it does at the kerb, and no kerbside marking
                # goes near the centreline.
                inner_ft = travel_lane_edge_ft(state, leg_name, side)
                reach_ft = crosswalk_reach_on_leg_side_ft(leg, side, band, inner_ft,
                                                           beyond_the_tracing=True) or None
            mouth = junction_mouth_ft(leg_name, side, state.legs, state.corner_fillets,
                                       crossing_reach_ft=reach_ft)
            if mouth is not None:
                out[(leg_name, side)] = mouth
    return out


def _union(shapes) -> object:
    """The union of whatever is not None, or None. Used to compose an opening's shapes per rule."""
    parts = [s for s in shapes if s is not None and not s.is_empty]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else unary_union(parts)


@dataclass(frozen=True)
class KerbSideOpenings:
    """Where ONE KERB opens for a vehicle, kept apart BY WHAT KIND OF OPENING IT IS.

    Three shapes, and the split is MUTCD 1C.02's: an intersecting approach and a driveway are
    different things and the markings do different things at them. What each marking does is
    declared once in markings.AT_AN_OPENING; this class holds the ground, and `against` composes
    the two.

      * `driveway_mouths` - entrances that are NOT intersections, trimmed back and rounded.
      * `driveway_tapered` - the same, plus the rounded run-out at the travel lane's edge.
      * `intersection_mouths` - approaches that ARE intersections: no trim and no run-out,
        because a street mouth has no apron.

    The fields are the ground and markings.AT_AN_OPENING is the rule, so adding a marking cannot
    silently inherit a branch it happens to fall through.
    """
    driveway_mouths: object = None
    driveway_tapered: object = None
    intersection_mouths: object = None

    @property
    def driven(self) -> object:
        """Every entrance, of both kinds, at its real width. What an OBJECT is kept out of and
        what a dotted extension is laid inside - neither question cares which kind it is."""
        return _union((self.driveway_mouths, self.intersection_mouths))

    @property
    def tapered(self) -> object:
        """`driven` plus every driveway run-out - the widest of the three, and the ground a
        hatched zone gives up. Equal in area to `driven` on a kerb whose only openings are
        intersections, which is the shape of "a street mouth has no apron"."""
        return _union((self.driveway_tapered, self.driveway_mouths, self.intersection_mouths))

    def against(self, kind) -> object:
        """The ground `kind` is cut out of, composed from its row in markings.AT_AN_OPENING.

        One rule per column: CARRIED subtracts nothing, FILLETED subtracts the run-out (and at an
        intersection there is none, so it subtracts the mouth), DOTTED and STOPPED subtract the
        mouth. What differs between the two columns is which SHAPES they apply to, which is the
        whole reason the shapes are kept apart above.

        3B.11(07)'s exception - solid edge lines MAY continue "through that part of an
        intersection with no intersecting approach (such as at the far side of a T-intersection)"
        - needs no code here, and that is worth stating because it looks like it should. An
        opening is only ever made on the kerb the approach actually leaves on: cross_streets.py
        reads that off the street's own vertices, and model.junction_mouth_ft returns None where
        the kerb runs straight through. A T's far kerb never enters `intersection_mouths` in the
        first place and its line is never cut. A crossroads opens both.
        """
        from src.geometry.markings import AtAnOpening, opening_rule

        rule = opening_rule(kind)
        driveway = {AtAnOpening.CARRIED: None,
                    AtAnOpening.FILLETED: self.driveway_tapered}.get(rule.at_a_driveway,
                                                                      self.driveway_mouths)
        intersection = (None if rule.at_an_intersection is AtAnOpening.CARRIED
                         else self.intersection_mouths)
        return _union((driveway, intersection))

    def dotted(self, kind) -> object:
        """The ground `kind` lays a dotted extension across, or None. The complement of `against`
        restricted to the columns whose rule is DOTTED, so the two never overlap: `add` keeps
        what is outside and this is where the dashes go back."""
        from src.geometry.markings import AtAnOpening, opening_rule

        rule = opening_rule(kind)
        return _union((
            self.driveway_mouths if rule.at_a_driveway is AtAnOpening.DOTTED else None,
            self.intersection_mouths if rule.at_an_intersection is AtAnOpening.DOTTED else None))

    def __bool__(self) -> bool:
        driven = self.driven
        return driven is not None and not driven.is_empty


@dataclass(frozen=True)
class KerbOpenings:
    """Every kerb's openings, KEPT PER KERB, plus the union for the things that stand in one.

    AN OPENING CUTS ONLY THE KERB IT OPENS. Every leg's junction mouth reaches into the SAME
    throat, so a single union asks an opening about a kerb that is not the one it opens: at
    W Broad & Louellen, whose two streets meet at 43.6 deg, Louellen's south mouth would swallow
    part of W Broad's two-way bike lane and break the corridor at the junction it runs through.
    So `against` and `dotted` take the marking's own (leg, side), falling back to the union only
    where a marking belongs to no single kerb - the corner treatments.

    THE OBJECTS STILL READ THE UNION, deliberately: `driven` is ground a vehicle crosses, and a
    flex post standing on it is in the way whichever kerb's entrance put it there.
    """
    by_kerb: dict = field(default_factory=dict)     # (leg, side) -> KerbSideOpenings

    @property
    def everywhere(self) -> KerbSideOpenings:
        """Every kerb's openings unioned - the answer for a marking that belongs to no one kerb."""
        return KerbSideOpenings(
            driveway_mouths=_union([o.driveway_mouths for o in self.by_kerb.values()]),
            driveway_tapered=_union([o.driveway_tapered for o in self.by_kerb.values()]),
            intersection_mouths=_union([o.intersection_mouths for o in self.by_kerb.values()]))

    def on(self, leg, side: str) -> KerbSideOpenings:
        if leg is None or side is None:
            return self.everywhere
        return self.by_kerb.get((leg, str(side)), KerbSideOpenings())

    @property
    def driven(self) -> object:
        """Every entrance on every kerb. What an OBJECT is kept out of - see the class docstring
        for why that one question is not asked per kerb."""
        return self.everywhere.driven

    @property
    def tapered(self) -> object:
        return self.everywhere.tapered

    def against(self, kind, leg=None, side: str | None = None) -> object:
        return self.on(leg, side).against(kind)

    def dotted(self, kind, leg=None, side: str | None = None) -> object:
        return self.on(leg, side).dotted(kind)

    def __bool__(self) -> bool:
        driven = self.driven
        return driven is not None and not driven.is_empty


def _stands_in_a_crossing(keep_clear, geometry) -> bool:
    """Whether an OBJECT is standing on a painted crossing, so it must not be placed.

    Measured against `keep_clear`, which is the crossings already buffered by
    PAINT_TO_CROSSWALK_GAP_FT. The gap is deliberately included: a post a foot from a crosswalk
    is a post in the crosswalk as far as anyone walking into it is concerned, and the same
    striper's gap that keeps paint off it keeps a bollard off it.
    """
    return (keep_clear is not None and not keep_clear.is_empty
            and keep_clear.intersects(geometry))


def stands_in_an_opening(openings, geometry) -> bool:
    """Whether an OBJECT belongs to ground a vehicle drives over, so it must not be placed.

    Shared by PaintContext.emit and the prop builders in src/render/props.py, which compute their
    own post positions and would otherwise disagree with the paint about where a post stands -
    the 2D/3D split this project keeps finding. Takes the openings rather than the state so a
    caller that already has them does not rebuild them per post.

    Measured against `driven`, not against the taper: a post beside a driveway is in the way only
    if it stands in the entrance. The extra few feet the hatching gives up is paint ending
    gracefully, not roadway a car uses.
    """
    if openings is None or geometry is None:
        return False
    driven = openings.driven if isinstance(openings, KerbOpenings) else openings
    return driven is not None and driven.intersects(geometry)


# How many points the fillet's arc is sampled at. A curve, not a staircase - see _opening_run_out.
FILLET_ARC_POINTS = 28


def _opening_run_out(leg, side: str, inner_ft, outer_ft, start_ft, end_ft):
    """The fillet a hatched zone ends on at an opening: one polygon per end, or [].

    An arc of radius = the strip's own depth, TANGENT TO THE ZONE'S EDGE LINE at the travel lane
    and arriving at the mouth at the kerb. So the zone's outline runs straight beside the lane,
    peels away in one sweep, and meets the entrance - which is what the white line does around a
    driveway apron on a real street, and the reason this is a fillet rather than a chamfer or a
    bulge. `run(u) = R - sqrt(R^2 - (R-u)^2)` for a strip depth R, u measured out from the lane
    edge: R at the lane edge, 0 at the kerb, vertical tangent at u=0.

    SAMPLED AS THE ARC ITSELF, in the leg's own frame, and NEVER BUFFERED. A round buffer grows
    the fillet in every direction including along its own tangent, so the curve would leave the
    edge line OPENING_TRIM_FT wide instead of at a point - a bulge where the sweep begins. The
    trim belongs to the mouth, where a turning vehicle needs the room; the fillet joins the
    trimmed mouth at both ends because it is built from the trimmed stations.

    `outer_ft` HAS TO BE THE REAL KERB, measured off the band, not the nominal width. A radius
    taken from the request rather than the clamp puts every arc step within a few percent of the
    full run, i.e. a square end. The result is intersected with the kerbside strip by the caller,
    which is what holds the arc to the traced kerb.
    """
    from src.geometry.model import inset_point_at_station
    from src.geometry.targets import Side

    depth_ft = outer_ft - inner_ft
    if depth_ft <= 0:
        return []
    radius_ft = depth_ft * OPENING_FILLET_PER_DEPTH
    sign = Side(side).sign

    def at(station_ft, offset_ft):
        return tuple(inset_point_at_station(leg, station_ft, sign * offset_ft))

    out = []
    for mouth_ft, direction in ((start_ft, -1), (end_ft, +1)):
        ring = []
        for i in range(FILLET_ARC_POINTS + 1):
            u_ft = depth_ft * i / FILLET_ARC_POINTS
            run_ft = radius_ft - math.sqrt(max(0.0, radius_ft ** 2 - (radius_ft - u_ft) ** 2))
            ring.append(at(max(mouth_ft + direction * run_ft, 0.0), inner_ft + u_ft))
        ring.append(at(mouth_ft, inner_ft))     # back along the mouth, then the lane edge closes it
        fillet = Polygon(ring)
        if not fillet.is_valid:
            fillet = fillet.buffer(0)
        if not fillet.is_empty:
            out.append(fillet)
    return out


def _traced_kerb_depth_ft(leg, side: str, start_ft: float, end_ft: float | None) -> float | None:
    """How far out the kerb actually runs over one span - the depth an opening's fillet sweeps.

    Measured, so it is the kerb's own figure and not the width the cut asked for: see
    _opening_run_out for the flat 4 ft gap that using the requested width gave instead.
    """
    stations = paint_stations(leg, side, start_ft, end_ft, beyond_the_tracing=True)
    if stations is None:
        return None
    offsets = curb_offsets_at_stations(leg, side, stations)
    if offsets is None:
        return abs(leg.curb_to_curb_ft) / 2
    return float(np.minimum(np.abs(offsets), abs(leg.curb_to_curb_ft)).max())


def _kerbside_cut(leg, side: str, inner_ft: float,
                  start_ft: float, end_ft: float | None):
    """The ground an opening removes: from the travel lane's edge to PAST the kerb, whatever
    describes the kerb.

    Deliberately NOT a kerbside strip. Every kerbside zone this cuts runs from `inner_ft` out to
    the kerb, so the obvious cut is the same strip over the opening's span - and that is wrong,
    because there is no single "the kerb" to share. Three constructions here answer that question
    and no two of them agree past the third decimal:

    - `curbside_strip_polygon` takes the kerb's OWN traced vertices (curb_edge_by_station). The
      fills use it, so over a traced stretch the difference is exact - but BEYOND the tracing its
      vertex list is just the two interpolated ends, a chord across a leg that curves. On
      louellen_st_west's left kerb, traced only from station 58.2 against a junction mouth of
      0-65.9, that chord cut 605.6 sq ft where the mouth is 893.3, and left 85 sq ft of daylight
      hatching standing in the intersection.
    - `offset_band_polygon` clamps a lateral offset to the kerb RESAMPLED onto paint_stations. It
      follows a curve out front and sags inside the traced kerb between samples, leaving a hairline
      of zone along the kerb - 0.005 to 0.071 ft of it at four driveways, a parking aisle and a side
      street across three sites.
    - EVEN THE SAME CONSTRUCTION OVER A DIFFERENT SPAN DISAGREES WITH ITSELF, which is what makes
      this margin rather than a shared construction the answer. A cut that starts mid-way along one
      traced kerb segment interpolates its end vertex in station-offset space, while the fill it
      cuts draws the world chord between that segment's two real vertices. On
      greenwood_ave_south's right kerb the two cross by 0.003 ft over a 12 ft cross street: 0.66
      sq ft of hatching survived, invisible as hatching, and PaintContext.rim then outlined it as a
      22.2 ft white line straight back across the opening - the exact defect this module exists to
      prevent, and the one a reader reported.

    So the cut is a plain band between two lateral offsets, and its outer offset is past every one
    of those answers. Overshooting was tried once before and appeared to make things worse; it was
    being clawed straight back, because the shapes were then clipped to a kerbside strip built with
    the clamping - so the overshoot bought nothing and left the cut's corners degenerate, and GEOS's
    overlay against the resulting doubled-back sliver is unreliable (a stroke measured 100% inside a
    fill whose own bounding box does not contain it, flipping in and out with the overshoot size).
    The clip is built here too, so that does not recur.
    """
    stations = paint_stations(leg, side, start_ft, end_ft, beyond_the_tracing=True)
    if stations is None:
        return None
    # PER STATION, and that is not a detail. Bounded instead by the kerb's widest offset anywhere
    # on the leg, this permitted 2.1 ft of slack where louellen_st_west's right kerb comes in at
    # 18.94 against a 20.92 ft maximum further out - enough for the trim's 1.5 ft round buffer to
    # bulge out there, and checks.PaintOverTheCurb reported the fill 0.45 ft past its kerb. A
    # maximum over a leg is also a figure that MOVES WITH THE SHEET (.claude/SKILLS.md 0b): the
    # leg is longer at --frame-scale 2.5, so it reaches a wider kerb and every cut on the leg
    # loosens. The bound has to be local.
    offsets = curb_offsets_at_stations(leg, side, stations)
    kerb_ft = (np.abs(offsets) if offsets is not None
               else np.full(stations.shape, abs(leg.curb_to_curb_ft) / 2))
    # Arrays, which is what band_from_offsets takes: a scalar reaches place_in_measured_frame as
    # a 0-d array and it cannot iterate one.
    return band_from_offsets(leg, side, stations,
                             np.full(stations.shape, inner_ft),
                             kerb_ft + OPENING_PAST_THE_KERB_FT)


def kerb_opening_bands(state: "DesignState", junction_mouths: dict | None = None) -> KerbOpenings:
    """Where the kerbside markings open for a vehicle, in the two shapes KerbOpenings holds.

    WHERE A VEHICLE CROSSES THE KERB, the markings it drives over open for it. A driveway is not
    a place to paint a bike lane's green surface, a parking stall or a hatched buffer across:
    those markings describe how the kerbside is used, and at a driveway it is used as an
    entrance. The spans come from the traced kerbs' own kerb=lowered / kerb=flush tags - see
    src/geometry/kerbs.py for why a dropped kerb rather than a driveway way is the signal.

    HOW DEEP. From the travel lane's edge out to the real kerb, and no further in. A driveway
    breaks what a car drives over on its way in; it does not break the line that marks the edge
    of the running lane, which carries straight past. TARGET_LANE_WIDTH_FT is the inner bound
    because that is the lane every treatment here holds - TravelLanesKeepTheirWidth is the
    invariant that makes it true, so no kerbside marking on a passing leg starts inside it.

    THE ENDS ARE TRIMMED BACK AND ROUNDED by OPENING_TRIM_FT, so a vehicle turning in or out has
    a little room and the gap reads as an entrance rather than as a rectangle punched through the
    markings. Deliberately small: this is cohesion, not a swept-path design, and every foot of it
    is a foot of bike lane or hatching given up. The trim is clipped back inside the kerbside
    strip so it can never reach into the travel lane, whose edge line runs straight past.

    AND A HATCHED ZONE GETS MORE THAN A TRIM. The trim alone is a foot and a half at 2D scale -
    correct for an entrance, but it left every no-travel zone ending on a blunt transverse edge,
    which is not how one ends anywhere else in this project: at a crossing it ends on the
    crossing's own diagonal. So the fills are cut against `tapered`, the same band plus
    _opening_run_out, and the lines and the green against `driven`.

    NEITHER THE TRIM NOR THE FILLET IS APPLIED AT AN INTERSECTING APPROACH, and both omissions
    are the same point: they model a DRIVEWAY APRON, which is a thing a street mouth does not
    have. A street mouth's flare is its CORNER RETURN, already in the geometry - so adding an
    apron's trim counts the same flare twice, and sweeping a fillet onto it draws a driveway
    apron across the mouth of Blackwell Avenue.

    A zone that ends at a street therefore ends SQUARE, which is not a shrug - it is the same end
    zone_end_line_ft already draws for a zone with nothing to end against, and past it the
    statutory setback (R.S. 39:4-138(e), src/geometry/daylighting.py) has usually stopped the
    parking well before the mouth anyway.
    """
    from src.geometry.kerbs import OpeningSource
    from src.geometry.treatments import travel_lane_edge_ft

    # WHERE THIS JUNCTION'S OWN MOUTH ENDS, resolved against the crossings by junction_mouths_ft
    # and passed in rather than re-derived: the span seeded onto the state by kerbs.py is the
    # corner return's, which is the right answer only where no crossing is painted. Absent (a
    # design built with no scene behind it) the seeded span stands.
    junction_mouths = junction_mouths or {}
    by_kerb: dict = {}
    for (leg_name, side), openings in getattr(state, "kerb_openings", {}).items():
        leg = state.legs.get(leg_name)
        if leg is None or leg.curb_to_curb_ft is None:
            continue
        # WHERE THE KERBSIDE ZONE BEGINS, which is where the travel lane ENDS on this side - not
        # a fixed TARGET_LANE_WIDTH_FT from the alignment. Those coincide only while the two travel
        # lanes straddle the alignment. Under a two-way bike lane the section starts far closer in
        # (4.22 ft from the alignment on e_broad_st_east against 11), so a region beginning at 11
        # covered only the OUTER part of the lane: the driveway break was drawn across some of the
        # bike lane and not the rest of it, which is visible in the render as striping that stops
        # part way across.
        #
        # THROUGH travel_lane_edge_ft AND NOT REBUILT AS `divider_shift + TARGET_LANE_WIDTH_FT`,
        # which is what stood here and is wrong by 0.92 ft on a leg that splits its travel way
        # rather than holding two target-width lanes. Rebuilt that way the cutter sat OUTBOARD of
        # the zone it had to cut, so instead of a swept fillet it left a 0.10 ft ribbon of hatching
        # along the zone's inner face and the rim outlined it across the driveway.
        inner_ft = travel_lane_edge_ft(state, leg_name, side)
        # BEYOND THE TRACING, because an opening is a FACT about the street and not a marking
        # proposed on it - the same short list model.paint_stations lets past that bound, and for
        # the same reason a daylight zone is on it. Where a vehicle crosses the kerb does not stop
        # being true where nobody traced the kerb, and a cut built only over the traced stretch is
        # narrower than the markings it has to cut. W Broad & Louellen's south kerb is traced from
        # station 60.3 against a junction mouth of 0-68.0, so without this the mouth came out 7.7 ft
        # long and left the daylight hatching it exists to remove standing in the intersection in
        # pieces - MORE pieces than before it was cut, which is the worst of both.
        # The whole kerbside band on this side, as the bound the trim is clipped to - and built by
        # _kerbside_cut too, which is the half of the overshoot that was missing the first time it
        # was tried: clipping to a kerb-clamped strip claws the overshoot straight back off.
        kerbside = _kerbside_cut(leg, side, inner_ft, 0.0, None)
        for opening in openings:
            start_ft, end_ft = opening.start_ft, opening.end_ft
            if opening.source is OpeningSource.JUNCTION:
                start_ft, end_ft = junction_mouths.get((leg_name, side), (start_ft, end_ft))
                if end_ft <= start_ft:
                    continue
            band = _kerbside_cut(leg, side, inner_ft, start_ft, end_ft)
            if band is None or band.is_empty:
                continue
            opening_start_ft, opening_end_ft = start_ft, end_ft
            # THE KERB AS TRACED HERE, and measured rather than read off the cut's own exterior:
            # the cut deliberately reaches OPENING_PAST_THE_KERB_FT further out than the kerb, so
            # its widest offset is no longer a fact about the street.
            depth_ft = _traced_kerb_depth_ft(leg, side, start_ft, end_ft)
            # THE TRIM IS THE MOUTH'S, and only the mouth's. JOIN_STYLE 1 is round, so the corners
            # where the entrance meets the travel lane edge and the kerb come off as arcs rather
            # than right angles. Buffering the fillet along with it - which is what this did - grew
            # the sweep by 1.5 ft in every direction including along its own tangent, so the curve
            # left the edge line 1.5 ft wide: the bulge where the sweep begins.
            #
            # AND ONLY AN APRON HAS ONE. An intersecting approach keeps the mouth the tracing
            # gave it, square - see this function's docstring for why the trim and the fillet are
            # one decision and both belong to a driveway.
            if opening.is_an_intersection:
                mouth, run_out = band, []
            else:
                mouth = band.buffer(OPENING_TRIM_FT, join_style=1, cap_style=1)
                # Grown from the TRIMMED mouth, so the arc's square end lands exactly on the
                # entrance's edge and the two join without a step.
                run_out = _opening_run_out(leg, side, inner_ft, depth_ft,
                                           opening_start_ft - OPENING_TRIM_FT,
                                           opening_end_ft + OPENING_TRIM_FT)
            shapes = by_kerb.setdefault((leg_name, side), {"driveway_mouths": [],
                                                            "driveway_tapered": [],
                                                            "intersection_mouths": []})
            if opening.is_an_intersection:
                targets = [(mouth, "intersection_mouths")]
            else:
                targets = [(mouth, "driveway_mouths"),
                           (unary_union([mouth, *run_out]), "driveway_tapered")]
            for shape, target in targets:
                if kerbside is not None and not kerbside.is_empty:
                    shape = shape.intersection(kerbside)
                if not shape.is_empty:
                    shapes[target].append(shape)
    return KerbOpenings(by_kerb={
        key: KerbSideOpenings(**{name: (unary_union(parts) if parts else None)
                                  for name, parts in shapes.items()})
        for key, shapes in by_kerb.items()})


def apron_polygon(state: "DesignState", corner: tuple[str, str], apron, center_ft: "Point"):
    """The ground one CornerApron covers - a fixed-depth kite, or the swept-path annulus.

    Two shapes because there are two reasons for an apron; see
    src/geometry/treatments/base.py:CornerApron. The annulus is built from the corner's two real
    curb lines rather than offset off the drawn arc, so the outer edge genuinely reaches the
    radius a bus needs.
    """
    if apron.swept_radius_ft is None:
        return corner_overlay_polygon(state.corner_fillets[corner], center_ft, apron.depth_ft)
    leg_a, leg_b = corner
    return corner_apron_annulus(state.legs[leg_a].left_curb, state.legs[leg_b].right_curb,
                                 apron.face_radius_ft, apron.swept_radius_ft)
