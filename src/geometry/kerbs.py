"""What OSM says about a kerb: whether it is raised, and where it is dropped for a vehicle.

Stations are feet along a leg's centreline; geometry is NJ state-plane feet.

TWO SIGNALS, BOTH READ. A driveway is mapped twice over: as a `highway=service` +
`service=driveway` way running up to the road, and as the stretch of kerb it crosses being tagged
`kerb=lowered`. Reading only the kerb leaves the markings running straight across a driveway whose
kerb nobody tagged; reading only the driveway loses the many dropped kerbs no service way reaches.
So both open the markings, and each opening records WHICH source put it there
(`KerbOpening.source`).

DATUM: a dropped kerb's extent is SURVEYED; a service way is a centreline with no width, so its
mouth is ASSUMED (DRIVEWAY_WIDTH_FT). Where both describe the same mouth the surveyed span wins.
`describe_kerb_openings` says which is which, and flags a driveway with no dropped kerb tagged at
its mouth as the survey gap it is.

A PEDESTRIAN RAMP IS ALSO A LOWERED KERB, and it must not open the paint: a crosswalk's kerb ramp
is dropped for a wheelchair, not for a car, and the markings there are cut by the crossing band
instead. This borough's surveyors separate the two EXPLICITLY and this module reads that convention
rather than inferring one - a driveway mouth is `wheelchair=no` AND `tactile_paving=no`, a ramp
`wheelchair=yes` and `tactile_paving=yes`.

The test is POSITIVE (see opens_the_kerb) rather than "lowered and not tagged as a ramp", which
decides two cases the looser rule got wrong. A bare `kerb=lowered` with neither tag stays CLOSED -
breaking a bike lane over it would be inventing a driveway. And a contradictory kerb - tactile
paving present but `wheelchair=no` - keeps its paint, because the safe reading of a disagreement
between two tags is the one that does not put a gap in a marking.

`wheelchair=no` alone is NOT the signal: every raised kerb carries it too. It means "not a
pedestrian crossing point" only once the kerb is already known to be dropped.
"""
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from src.geometry.intersection.junction import PavedKind
from src.geometry.model import (curb_offsets_at_stations, junction_mouth_ft,
                                station_offset_many)


class KerbType(StrEnum):
    """OSM's `kerb=*` value, reduced to what this project draws differently.

    A StrEnum so it compares and hashes as the OSM value it came from, the same reason
    targets.Side is one. UNKNOWN is a real answer and not an error: an untagged kerb way is one
    somebody traced without saying what kind, which is not the same as a raised one.
    """
    RAISED = "raised"
    LOWERED = "lowered"
    FLUSH = "flush"
    UNKNOWN = "unknown"

    @classmethod
    def from_tags(cls, tags: dict) -> "KerbType":
        value = (tags or {}).get("kerb")
        try:
            return cls(value)
        except ValueError:
            return cls.UNKNOWN

    @property
    def is_crossable_by_a_vehicle(self) -> bool:
        """Whether a car can drive over this kerb - dropped or flush, not stood up."""
        return self in (KerbType.LOWERED, KerbType.FLUSH)


def opens_the_kerb(tags: dict) -> bool:
    """Whether this kerb way is a VEHICLE opening, so the kerbside markings break over it.

    Three things must all hold, and the last two are the surveyor's own convention read back
    rather than a rule this project invented - see the module docstring:

      * the kerb is DROPPED (lowered or flush), so a vehicle can cross it at all;
      * the mapper has said it is NOT a pedestrian crossing point (`wheelchair=no`), which is
        what distinguishes a driveway from the kerb ramp at a crosswalk;
      * no tactile paving, because a detectable warning surface means a pedestrian facility
        whatever else the way says, and a disagreement between two tags should not put a gap in
        a marking.
    """
    tags = tags or {}
    if not KerbType.from_tags(tags).is_crossable_by_a_vehicle:
        return False
    if tags.get("tactile_paving") == "yes":
        return False
    return tags.get("wheelchair") == "no"


class OpeningSource(StrEnum):
    """Which OSM object said a vehicle crosses the kerb here.

    Recorded per opening rather than collapsed, for two independent reasons: it is the datum
    behind the citation (surveyed extent vs assumed mouth), and it decides what kind of junction
    the gap is, which decides the markings - see is_an_intersection.
    """
    DROPPED_KERB = "dropped_kerb"
    DRIVEWAY = "driveway"                # highway=service + service=driveway
    PARKING_AISLE = "parking_aisle"      # highway=service + service=parking_aisle
    ALLEY = "alley"                      # highway=service + service=alley
    CROSS_STREET = "cross_street"
    JUNCTION = "junction"                # THIS junction's own mouth - see junction_mouth_ft

    @property
    def is_an_intersection(self) -> bool:
        """Whether a vehicle crossing the kerb here is leaving an INTERSECTION or a DRIVEWAY.

        NOT a distinction this project invented, and not one it is free to decide. Both
        authorities define it and they agree (STANDARDS.md section 2):

          * MUTCD 11th ed. 1C.02(113)(b) - "The junction of an alley, driveway, or site roadway
            with a public roadway or highway shall not constitute an intersection, unless the
            public roadway or highway at said junction is controlled by a traffic control
            device." Its 1C.02(113)(a) requires "two highways" for the affirmative case.
          * N.J.S.A. 39:1-1 - an "intersection" joins "two or more highways"; a "private road or
            driveway" is one "not open to the use of the public for vehicular travel".

        THE VALUES ABOVE ARE OSM'S, AND THAT IS NOT A COINCIDENCE. 1C.02(113)(b)'s own list -
        "an alley, driveway, or site roadway" - is the same taxonomy as OSM's `service=*`, so the
        negative arm of the rule is read straight off the tag, one enum value per tag value so the
        citation names what the mapper recorded. Same discipline as intersection.PavedKind.

        Everything downstream that treats a gap in the kerb differently must read THIS rather than
        testing `.source`, so a new source answers the question once, here, and every marking rule
        follows.

        The two affirmative cases are the two kinds of "junction of two highways": CROSS_STREET,
        another street our leg runs across, and JUNCTION, the one the drawing is CENTRED on.

        DROPPED_KERB is a kerb tagged `wheelchair=no` with no tactile paving - this borough's
        convention for a driveway mouth. A dropped kerb tagged across a STREET's mouth would be a
        driveway by this rule and an intersection in fact, so kerb_openings_from_model resolves that
        overlap in favour of the cross street rather than leaving it here.

        The "unless … controlled by a traffic control device" arm of 1C.02(113)(b) is NOT
        implemented, and STANDARDS.md says so rather than this pretending to be the whole rule. No
        driveway at any of these sites is signalised, so the branch would never fire, and a rule
        that has never fired pins nothing.
        """
        return self in (OpeningSource.CROSS_STREET, OpeningSource.JUNCTION)


@dataclass(frozen=True)
class KerbOpening:
    """A stretch of ONE KERB of one leg that vehicles cross.

    A span and not a point, for the same reason ParkingRestriction is one: a driveway mouth is
    10-30 ft of kerb, and the paint has to break over its width rather than at a station.

    `citation` names the way it came from, so a gap in the markings on a drawing can be traced
    back to the OSM object that put it there - the same accounting ParkingRestriction.citation
    gives a hatched kerb.
    """
    start_ft: float
    end_ft: float
    source: OpeningSource
    kerb: KerbType | None = None
    way_id: int | None = None

    @property
    def length_ft(self) -> float:
        return max(self.end_ft - self.start_ft, 0.0)

    @property
    def is_surveyed_width(self) -> bool:
        """Whether the SPAN is surveyed or assumed. A dropped kerb's own extent is the width of
        the opening; a service way's centreline has none, so DRIVEWAY_WIDTH_FT stands in for it,
        and this junction's mouth is measured off the modelled corner rather than surveyed."""
        return self.source is OpeningSource.DROPPED_KERB

    @property
    def is_an_intersection(self) -> bool:
        """Whether this gap is an intersecting approach or a driveway - OpeningSource's rule.

        Exposed on the opening rather than making every consumer reach for `.source`, because the
        marking rules are about the GAP: MUTCD 3B.11(08) discontinues an edge line "across
        intersecting approaches" and 3B.11(09) maintains it across a driveway.
        """
        return self.source.is_an_intersection

    @property
    def citation(self) -> str:
        where = f" on way {self.way_id}" if self.way_id is not None else ""
        if self.source in SERVICE_WAY_SOURCES:
            return (f"OSM service={self.source.value}{where} (mouth assumed "
                    f"{DRIVEWAY_WIDTH_FT:.0f} ft - a centreline carries no width)")
        if self.source is OpeningSource.CROSS_STREET:
            return (f"OSM intersecting street{where} (mouth is its own carriageway width, "
                    f"assumed from its highway class unless OSM records one)")
        if self.source is OpeningSource.JUNCTION:
            # NO OSM OBJECT: OSM maps no intersection AREA, so this is the one opening in the
            # project that cannot come from a survey. It comes from the corner return, the same
            # point R.S. 39:4-138(e) is read from (src/geometry/daylighting.py:sideline_station_ft).
            return "this junction's own corner return (src/geometry/model/corners.py)"
        return f"OSM kerb={self.kerb.value if self.kerb else 'lowered'}{where}"


# The three OSM `service=*` values that open a kerb without being an intersection - MUTCD
# 1C.02(113)(b)'s "an alley, driveway, or site roadway", read off the tag. Declared as a set so
# the citation and the collection loop cannot disagree about which values are in it.
SERVICE_WAY_SOURCES: frozenset = frozenset({
    OpeningSource.DRIVEWAY, OpeningSource.PARKING_AISLE, OpeningSource.ALLEY,
})


# How far from the junction centre openings are collected. Matches the kerb radius the 3D export
# draws with (src/render/export.py:KERB_RADIUS_M, 120 m), so the paint breaks everywhere the
# picture shows a driveway rather than only within a leg's working length.
OPENING_COLLECTION_RADIUS_FT = 120.0 / 0.3048


# How far outside a leg's nominal half-width a kerb vertex may sit and still be that leg's kerb.
# The same allowance intersection.py's own kerb-to-leg test uses, and for the same reason: a
# traced kerb flares through the corner returns well past the mid-block cross-section.
OPENING_OFFSET_TOLERANCE_FT = 8.0
# A dropped kerb shorter than this is not a driveway mouth - it is the last vertex of a ramp or
# a tracing artifact, and breaking the paint for it would put a nick in a marking.
MIN_OPENING_LENGTH_FT = 4.0

# How wide a driveway mouth is taken to be. AN ASSUMPTION, and the reason a dropped kerb is the
# better of the two signals: OSM maps a driveway as a centreline with no width at all, so this
# stands in for a survey. About a residential driveway, and single-sourced here - the 3D render
# draws the driveway strip at this width too (src/render/export.py writes it into the JSON), so
# the strip and the gap it explains cannot end up different sizes.
DRIVEWAY_WIDTH_FT = 10.0
# How close a driveway way has to come to a modelled kerb to be treated as meeting it. Sits between
# the driveway that touches its kerb at 0.0 ft and the next nearest at 21.7 ft, which belongs to a
# kerb further down the block, so it cannot drag in a neighbour's driveway.
DRIVEWAY_REACH_FT = 5.0

# How far past its assumed width a cross street's mouth may be moved to reach where the traced kerb
# actually stops - see _mouth_from_the_tracing. Bounded because "no raised kerb is traced between
# here and there" has two causes and only one is a mouth: a real corner return runs a few feet past
# the assumption, while on an untraced leg the nearest kerb may be a hundred feet away and the
# assumed width is the better answer.
MAX_MOUTH_SNAP_FT = 20.0


def kerb_openings_from_model(model) -> dict:
    """{(leg, side): [KerbOpening]} for every vehicle-crossable kerb along this junction's legs.

    Seeded onto the design by DesignState.from_model, exactly as parking_restrictions and
    existing_centerline_styles are, and for the same reason: this is an observed fact about the
    street that no treatment chose, so it belongs beside those two rather than being re-derived
    by each renderer or reached back for out of the model.
    """
    from src.geometry.intersection import kerb_lines_with_tags_ft

    # Guarded the way _parking_restrictions_from_model guards its own model access: a design can be
    # built from a stand-in that carries legs and config and no OSM at all, and a seeded observed
    # fact has to be absent then rather than raising.
    if not all(hasattr(model, attr) for attr in ("center_wgs84", "center_ft", "legs")):
        return {}
    openings: dict[tuple[str, str], list[KerbOpening]] = {}
    # THE DRAWING RADIUS, NOT THE LEG TEST. An opening is a fact about the KERB, so gating
    # collection on leg membership asks an unrelated question first - and that test measures
    # against the NOMINAL half-width, which on Broad St is 12-13 ft from the traced kerb a driveway
    # actually sits on, so real driveways were judged "not this leg's kerb" and dropped.
    # _place_on_a_leg_side still assigns a leg afterwards, because a STATION needs one, but that is
    # stationing rather than eligibility: an opening it cannot place has no paint to break.
    for line, tags, way_id in kerb_lines_with_tags_ft(model.center_wgs84, model.center_ft,
                                                       radius_ft=OPENING_COLLECTION_RADIUS_FT):
        if not opens_the_kerb(tags):
            continue
        placed = _place_on_a_leg_side(line, model.legs)
        if placed is None:
            continue
        leg_name, side, start_ft, end_ft = placed
        if end_ft - start_ft < MIN_OPENING_LENGTH_FT:
            continue
        openings.setdefault((leg_name, side), []).append(
            KerbOpening(start_ft=start_ft, end_ft=end_ft, source=OpeningSource.DROPPED_KERB,
                        kerb=KerbType.from_tags(tags), way_id=way_id))
    for leg_name, side, station_ft, way_id, source in _service_way_meetings(model):
        # THE SURVEYED WIDTH WINS. Where a dropped kerb is already tagged across this mouth its own
        # extent is the opening; adding a second assumed-width opening inside it would put a
        # narrower guess on top of a measurement and double-count it in the report.
        if any(o.source is OpeningSource.DROPPED_KERB and o.start_ft <= station_ft <= o.end_ft
               for o in openings.get((leg_name, side), ())):
            continue
        openings.setdefault((leg_name, side), []).append(
            KerbOpening(start_ft=max(station_ft - DRIVEWAY_WIDTH_FT / 2, 0.0),
                        end_ft=station_ft + DRIVEWAY_WIDTH_FT / 2,
                        source=source, way_id=way_id))
    # A CROSS STREET opens the kerb the same way a driveway does, over its own width. See
    # src/geometry/cross_streets.py; the statutory 25 ft either side of the same meeting is added in
    # src/geometry/daylighting.py, a separate rule about parking rather than about where the kerb
    # physically stops.
    from src.geometry.cross_streets import cross_streets_from_model

    # The model's own resolution, not a second derivation of it - see cross_streets_from_model.
    traced = _kerb_coverage_outside_openings(model)
    for leg_name, crossings in cross_streets_from_model(model).items():
        for cross in crossings:
            for side in cross.sides:
                # THE SURVEYED WIDTH WINS HERE TOO. The cross street's own carriageway width is an
                # assumption about that street, not a measurement of THIS kerb: at E Broad &
                # Hamilton the tracing gaps over 143.4-173.4 against an assumed mouth of
                # 141.7-167.7, so kerbside hatching ran on over ground with no kerb beside it.
                # Where the tracing shows where the kerb really stops, that is the mouth.
                near_ft, far_ft = _mouth_from_the_tracing(
                    traced.get((leg_name, side), ()), cross.station_ft, cross.mouth_ft)
                start_ft, end_ft = max(near_ft, 0.0), far_ft
                # A WHOLE STREET OUTRANKS A DRIVEWAY TAG AT THE SAME PLACE. A dropped kerb or
                # service way inside a street's mouth describes that mouth twice, and the two carry
                # different rules: an intersecting approach breaks the edge line (MUTCD 3B.11(08))
                # where a driveway carries it across (3B.11(09)). Left in, the same gap would be
                # both at once, each cutting a different set of markings.
                #
                # The mirror of the DROPPED_KERB-beats-DRIVEWAY rule above, resolved the opposite
                # way on purpose: that one is about which SPAN is measured, this one about what the
                # gap IS. Nothing at these sites hits it today.
                inside = [o for o in openings.get((leg_name, side), ())
                          if not o.is_an_intersection
                          and start_ft <= o.start_ft and o.end_ft <= end_ft]
                for swallowed in inside:
                    openings[(leg_name, side)].remove(swallowed)
                openings.setdefault((leg_name, side), []).append(
                    KerbOpening(start_ft=start_ft, end_ft=end_ft,
                                source=OpeningSource.CROSS_STREET, way_id=cross.way_id))
    # ...AND THIS JUNCTION, an intersecting approach like every one above rather than a special
    # case. See checks.NoPaintInsideTheJunction, and src/geometry/model/corners.py:junction_mouth_ft
    # for why a kerb that runs straight through has no mouth and needs no exception written for it.
    #
    # LAST, so the cross-street rule above has already run. A cross street cannot then be swallowed
    # by the junction: it is 130 ft or more out, and a junction mouth ends at the corner return,
    # 50 ft in at the widest of these sites.
    for leg_name in model.legs:
        for side in ("left", "right"):
            mouth = junction_mouth_ft(leg_name, side, model.legs, model.corner_fillets)
            if mouth is None:
                continue
            openings.setdefault((leg_name, side), []).append(
                KerbOpening(start_ft=mouth[0], end_ft=mouth[1], source=OpeningSource.JUNCTION))
    for key in openings:
        openings[key].sort(key=lambda o: o.start_ft)
    return openings


def _service_way_meetings(model) -> list[tuple[str, str, float, int | None, OpeningSource]]:
    """(leg, side, station, way id, source) wherever a mapped SERVICE WAY reaches a modelled kerb.

    The SECOND signal (module docstring): a driveway drawn without its kerb tagged is still a
    driveway.

    ALL THREE OF MUTCD 1C.02(113)(b)'s NON-INTERSECTIONS, not just the driveways - a parking aisle
    that meets the street is a site roadway and opens the kerb exactly as a driveway does. Which
    value it was is carried through rather than flattened to DRIVEWAY, so the citation names the tag
    the mapper wrote.

    Which kerb it meets is decided by distance to the kerb LINE rather than by the way's own
    direction, because a service way is drawn from the property to the road and its last segment is
    not reliably square to anything. The station is taken from the point on the KERB nearest the
    way, not the way's own endpoint, which a mapper may have stopped short of or run past.
    """
    from shapely.ops import nearest_points

    meetings = []
    for drive in getattr(model, "site_roadways", ()):
        line = drive.line
        if line is None or line.is_empty:
            continue
        source = _SERVICE_SOURCE_BY_KIND.get(drive.kind)
        if source is None:
            continue
        best = None
        for leg_name, leg in model.legs.items():
            for side in ("left", "right"):
                curb = getattr(leg, f"{side}_curb", None)
                if curb is None or curb.is_empty:
                    continue
                gap_ft = line.distance(curb)
                if gap_ft > DRIVEWAY_REACH_FT:
                    continue
                _on_drive, on_curb = nearest_points(line, curb)
                station_ft = leg.centerline.project(on_curb)
                if not 0 <= station_ft <= leg.centerline.length:
                    continue
                if best is None or gap_ft < best[0]:
                    best = (gap_ft, leg_name, side, station_ft)
        if best is not None:
            _gap, leg_name, side, station_ft = best
            meetings.append((leg_name, side, station_ft, drive.way_id, source))
    return meetings


# OSM's `service=*` value, kept as the PavedKind it was fetched as
# (src/geometry/intersection/junction.py:PavedKind). A PARKING_LOT is an area behind a building and
# crosses no kerb of ours; a ROADWAY is a street, and a street that meets a leg is a CROSS_STREET
# resolved by cross_streets.py, not a service way.
_SERVICE_SOURCE_BY_KIND = {
    PavedKind.DRIVEWAY: OpeningSource.DRIVEWAY,
    PavedKind.PARKING_AISLE: OpeningSource.PARKING_AISLE,
}


def _kerb_coverage_outside_openings(model) -> dict:
    """{(leg, side): [(start_ft, end_ft)]} for the stations a traced kerb covers, openings aside.

    Everything opens_the_kerb rejects counts as coverage, which is a wider set than "raised" and
    deliberately so. A dropped kerb is only an opening when the surveyor also said it is not a
    pedestrian crossing point; the lowered stubs at E Broad & Hamilton's corners are kerb RAMPS,
    tagged for pedestrians, and they are traced kerb a marking can perfectly well run beside.
    Excluding every lowered stretch instead would push each mouth out to the last raised kerb and
    swallow its ramps - 2.7 ft further out at Hamilton, on ground that is drawn as kerb.
    """
    from src.geometry.intersection import kerb_lines_with_tags_ft

    covered: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for line, tags, _way_id in kerb_lines_with_tags_ft(model.center_wgs84, model.center_ft,
                                                        model.legs):
        if opens_the_kerb(tags):
            continue
        placed = _place_on_a_leg_side(line, model.legs)
        if placed is None:
            continue
        leg_name, side, start_ft, end_ft = placed
        covered.setdefault((leg_name, side), []).append((start_ft, end_ft))
    for spans in covered.values():
        spans.sort()
    return covered


def _mouth_from_the_tracing(spans, station_ft: float, assumed: tuple) -> tuple:
    """Where the traced kerb really stops either side of a cross street, or `assumed`.

    Three ways this declines to move the mouth, each for its own reason:

      * A kerb traced STRAIGHT THROUGH the station. Then the survey says there is no gap
        in the kerb here, and widening a mouth to fit a gap that is not there would open the
        markings over a kerb somebody traced.
      * Nothing traced on that side at all - no bound to snap to, so the assumption stands.
      * A bound further than MAX_MOUTH_SNAP_FT out. "The nearest traced kerb is a long way off"
        means the kerb is unmapped, not that the mouth is enormous.

    Each end is decided on its own: a mouth is routinely traced tight on one corner and short on
    the other, and taking the assumption for both because one end failed would throw away the end
    that was measured.
    """
    if any(start < station_ft < end for start, end in spans):
        return assumed
    assumed_near, assumed_far = assumed
    before = [end for _start, end in spans if end <= station_ft]
    after = [start for start, _end in spans if start >= station_ft]
    near = max(before) if before else assumed_near
    far = min(after) if after else assumed_far
    if abs(near - assumed_near) > MAX_MOUTH_SNAP_FT:
        near = assumed_near
    if abs(far - assumed_far) > MAX_MOUTH_SNAP_FT:
        far = assumed_far
    return (near, far) if far > near else assumed


def _place_on_a_leg_side(line, legs: dict):
    """(leg, side, start_ft, end_ft) for the kerb this line lies along, or None.

    Which leg AND which side, decided by the same measurement: a kerb belongs to whichever leg
    it runs closest to the half-width of, and to the side the sign of that offset gives. Nearest
    wins rather than first match, because at a junction the two streets' kerbs meet and a corner
    return is plausibly either street's until the offsets are compared.

    THE WAY IS PLACED AS A WHOLE, then its whole extent is measured. Filtering to the vertices
    that individually sit within the tolerance and taking their span instead collapsed four of
    the six real openings to ZERO length: a dropped kerb across a driveway mouth is often two or
    three vertices drawn ACROSS the opening rather than along the leg, so only one of them lands
    near the half-width and min == max. Which leg it belongs to is a question about the way; how
    long it is, is a question about all of its vertices.
    """
    points = np.asarray(line.coords, dtype=float)
    best = None
    for leg_name, leg in legs.items():
        if leg.curb_to_curb_ft is None:
            continue
        stations, offsets = station_offset_many(leg.centerline, points)
        # The median rather than the mean, so one stray vertex - a way that turns up a driveway,
        # or runs on past the leg - cannot decide which street the kerb is on.
        typical_offset = float(np.median(offsets))
        typical_station = float(np.median(stations))
        # AGAINST THE TRACED KERB AT THIS STATION, not the nominal half-width. They are the same
        # only on a leg whose config width matches its tracing, and on Broad St they differ by
        # 12-13 ft: broad_st_east is 68 ft nominal (34 per side) against traced kerbs 21.97-34.55
        # ft out. A driveway's dropped kerb sits on the TRACED kerb, so measured against nominal
        # it looked 12 ft adrift - past the 8 ft tolerance - and was judged not to be this leg's
        # kerb at all. Nine driveways produced one opening, and the lane's paint ran unbroken
        # across the rest while the render drew their paving.
        side = "left" if typical_offset > 0 else "right"
        traced = curb_offsets_at_stations(leg, side, np.array([typical_station]))
        half_ft = (float(abs(traced[0])) if traced is not None and np.isfinite(traced[0])
                   else leg.curb_to_curb_ft / 2)
        error = abs(abs(typical_offset) - half_ft)
        if error > OPENING_OFFSET_TOLERANCE_FT:
            continue
        if not 0 <= typical_station <= leg.centerline.length:
            continue
        span = (max(float(stations.min()), 0.0),
                min(float(stations.max()), leg.centerline.length))
        side = "left" if typical_offset > 0 else "right"
        if best is None or error < best[0]:
            best = (error, leg_name, side, span)
    if best is None:
        return None
    _error, leg_name, side, (start_ft, end_ft) = best
    return leg_name, side, start_ft, end_ft


def describe_kerb_openings(state) -> list[str]:
    """One line per opening this design will break its markings for, for the phase output.

    Provenance, not diagnostics: a gap in a drawing's markings is a claim about the street, and
    the reviewer needs to check it against OSM. Every line names the way, and says whether the
    span is surveyed (a dropped kerb's own extent) or assumed (a driveway centreline has no
    width). Follows src/render/props.py:data_gaps in shape - sentences a phase script prints.

    Where the two sources CORROBORATE each other the line says so, and where a driveway stands
    alone it is called out as a survey gap: the driveway is drawn and believed, but a dropped kerb
    at its mouth would replace an assumed 10 ft width with a surveyed one.
    """
    lines = []
    for (leg_name, side), openings in sorted(state.kerb_openings.items()):
        kerbs = [o for o in openings if o.source is OpeningSource.DROPPED_KERB]
        for opening in openings:
            overlapping = [o for o in kerbs
                           if o is not opening
                           and o.start_ft < opening.end_ft and opening.start_ft < o.end_ft]
            if opening.source is OpeningSource.DRIVEWAY and overlapping:
                # Should not arise - a driveway inside a tagged dropped kerb is skipped above, in
                # favour of the surveyed extent. Kept as a statement rather than removed, so a
                # future change that lets both through says so instead of silently doubling up.
                agreement = (" Overlaps the dropped kerb tagged over "
                             f"{overlapping[0].start_ft:.0f}-{overlapping[0].end_ft:.0f} ft, whose "
                             f"surveyed extent should have governed this gap.")
            elif opening.source is OpeningSource.DRIVEWAY:
                agreement = (" NO dropped kerb is tagged at this mouth, so the width is assumed "
                             "rather than surveyed - tagging the kerb here would settle it.")
            else:
                agreement = ""
            lines.append(
                f"{leg_name} {side}: kerbside markings break over {opening.start_ft:.0f}-"
                f"{opening.end_ft:.0f} ft ({opening.length_ft:.0f} ft, "
                f"{'surveyed' if opening.is_surveyed_width else 'assumed'}) - "
                f"{opening.citation}.{agreement}")
    return lines
