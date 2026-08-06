"""What OSM says about a kerb: whether it is raised, and where it is dropped for a vehicle.

The tags were already reaching this pipeline and being thrown away at the last step. Every one
of the 95 `barrier=kerb` ways mapped in Hopewell carries `kerb=raised` or `kerb=lowered` - 67
and 28 of them, none untagged - and `fetch_kerbs` keeps every tag, and
`kerb_lines_with_tags_ft` hands back `(line, tags)` pairs. But nothing read `tags["kerb"]`: the
plan view drew every traced kerb as the same black line, the 3D render drew no kerb at all, and
the kerbside paint ran unbroken along the whole leg. So a surveyor's careful raised/lowered
distinction reached the geometry and stopped there, which is this project's signature failure -
ground truth present, never reaching the render.

TWO SIGNALS, BOTH READ. A driveway is mapped twice over: as a `highway=service` +
`service=driveway` way running up to the road, and as the stretch of kerb it crosses being tagged
`kerb=lowered`. An earlier version of this module read only the kerb, on the reasoning that it is
the better signal - it is on the kerb, so it already carries the station span the paint needs,
where a driveway way only asserts that a driveway exists near here; and it is tagged in places no
driveway way is drawn, since only ONE of the 43 driveways mapped in the borough reaches a kerb
this project models.

All true, and none of it a reason to read one source and discard the other. A driveway drawn
without its kerb tagged then produced no opening at all and the markings ran straight across it -
which is this repo's own failure mode, committed deliberately while arguing that keeping the two
independent would make the mismatch visible. It would not have: nothing was comparing them.

So both open the markings, and each opening records WHICH source put it there
(`KerbOpening.source`). A driveway way is the weaker of the two for one specific reason, stated
where it matters rather than used to dismiss it: a centreline carries no width, so the mouth has
to be assumed (DRIVEWAY_WIDTH_FT), where a dropped kerb's own extent is surveyed.
`describe_kerb_openings` says which is which, and flags a driveway with no dropped kerb tagged at
its mouth as the survey gap it is.

A PEDESTRIAN RAMP IS ALSO A LOWERED KERB, and it must not open the paint - a crosswalk's kerb
ramp is dropped for a wheelchair, not for a car, and the markings there are cut by the crossing
band instead. The surveyor's convention here separates them EXPLICITLY, and this module reads
that convention rather than inferring one: a driveway is tagged `wheelchair=no` AND
`tactile_paving=no`, a ramp `wheelchair=yes` and `tactile_paving=yes`. Borough-wide that is
12 dropped kerbs against 14 ramps, with no overlap:

    kerb=lowered  tactile_paving=no   wheelchair=no    12   driveways
    kerb=lowered  tactile_paving=yes  wheelchair=yes   14   pedestrian ramps
    kerb=lowered  (neither tag)                         1   unspecified - does NOT open
    kerb=lowered  tactile_paving=yes  wheelchair=no     1   contradictory - does NOT open
    kerb=raised   tactile_paving=no   wheelchair=no    67   not an opening at all

`wheelchair=no` is the load-bearing half, and it is why the rule is a POSITIVE test rather than
"lowered and not tagged as a ramp". Two things follow that the looser rule got wrong. A bare
`kerb=lowered` with neither tag is a kerb somebody recorded as dropped without saying what for,
and breaking a bike lane over it would be inventing a driveway; it now stays closed. And the
contradictory case - tactile paving present but wheelchair=no - keeps its paint, because tactile
paving means a pedestrian facility whatever else is on the way, and the safe reading of a
disagreement between two tags is the one that does not put a gap in a marking.

Note `wheelchair=no` alone is NOT the signal: all 67 raised kerbs carry it too. It only means
"not a pedestrian crossing point" once the kerb is already known to be dropped.

`describe_kerb_openings` still lists every opening with the way that produced it, so a gap in a
marking is reviewable against the survey rather than being a gap nobody can account for.
"""
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from src.geometry.model import station_offset_many


class KerbType(StrEnum):
    """OSM's `kerb=*` value, reduced to what this project draws differently.

    A StrEnum so it compares and hashes as the OSM value it came from, the same reason
    targets.Side is one. UNKNOWN is a real answer and not an error: a kerb way with no `kerb`
    tag is a kerb somebody traced without saying what kind, which is different from a raised one
    and must not be drawn as though the question had been settled.
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

    Recorded per opening rather than collapsed, because the two are not equally good evidence and
    the difference belongs in the citation: a dropped kerb's extent is SURVEYED, while a driveway
    is a centreline with no width, so its mouth is assumed. A reader auditing a gap in a marking
    needs to know which of those they are looking at.
    """
    DROPPED_KERB = "dropped_kerb"
    DRIVEWAY = "driveway"
    CROSS_STREET = "cross_street"


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
        the opening; a driveway centreline has none, so DRIVEWAY_WIDTH_FT stands in for it."""
        return self.source is OpeningSource.DROPPED_KERB

    @property
    def citation(self) -> str:
        where = f" on way {self.way_id}" if self.way_id is not None else ""
        if self.source is OpeningSource.DRIVEWAY:
            return (f"OSM service=driveway{where} (mouth assumed {DRIVEWAY_WIDTH_FT:.0f} ft - a "
                    f"centreline carries no width)")
        if self.source is OpeningSource.CROSS_STREET:
            return (f"OSM intersecting street{where} (mouth is its own carriageway width, "
                    f"assumed from its highway class unless OSM records one)")
        return f"OSM kerb={self.kerb.value if self.kerb else 'lowered'}{where}"


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
# How close a driveway way has to come to a modelled kerb to be treated as meeting it. The one
# driveway that reaches a kerb here touches it at 0.0 ft; the next nearest are 21.7 and 29.8 ft
# away and belong to kerbs further down the block than these legs reach, so this sits well clear
# of both and cannot drag in a neighbour's driveway.
DRIVEWAY_REACH_FT = 5.0


def kerb_openings_from_model(model) -> dict:
    """{(leg, side): [KerbOpening]} for every vehicle-crossable kerb along this junction's legs.

    Seeded onto the design by DesignState.from_model, exactly as parking_restrictions and
    existing_centerline_styles are, and for the same reason: this is an observed fact about the
    street that no treatment chose, so it belongs beside those two rather than being re-derived
    by each renderer or reached back for out of the model.
    """
    from src.geometry.intersection import kerb_lines_with_tags_ft

    # Guarded the way _parking_restrictions_from_model guards its own model access: a design can
    # be built from a stand-in that carries legs and config and no OSM at all (the centerline
    # precedence tests do exactly that), and a seeded observed fact has to be absent then rather
    # than raising. No kerbs mapped and no kerbs fetchable are the same answer here - no openings.
    if not all(hasattr(model, attr) for attr in ("center_wgs84", "center_ft", "legs")):
        return {}
    openings: dict[tuple[str, str], list[KerbOpening]] = {}
    for line, tags, way_id in kerb_lines_with_tags_ft(model.center_wgs84, model.center_ft,
                                                       model.legs):
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
    for leg_name, side, station_ft, way_id in _driveway_meetings(model):
        # THE SURVEYED WIDTH WINS. Where a dropped kerb is already tagged across this mouth, its
        # own extent is the width of the opening - measured, where a driveway centreline carries
        # none and has to be assumed. Adding a second assumed-width opening inside a surveyed one
        # would put a narrower guess on top of a measurement and double-count it in the report.
        if any(o.source is OpeningSource.DROPPED_KERB and o.start_ft <= station_ft <= o.end_ft
               for o in openings.get((leg_name, side), ())):
            continue
        openings.setdefault((leg_name, side), []).append(
            KerbOpening(start_ft=max(station_ft - DRIVEWAY_WIDTH_FT / 2, 0.0),
                        end_ft=station_ft + DRIVEWAY_WIDTH_FT / 2,
                        source=OpeningSource.DRIVEWAY, way_id=way_id))
    # A CROSS STREET opens the kerb the same way a driveway does, over its own width - it is a
    # driveway that a whole street drives out of. Legs are drawn as far as the frame asks now, so
    # Broad St runs across Blackwell and Model Avenue, and the markings were being painted
    # straight over their mouths. See src/geometry/cross_streets.py; the statutory 25 ft either
    # side of the same meeting is added in src/geometry/daylighting.py, which is a separate rule
    # about parking rather than about where the kerb physically stops.
    from src.geometry.cross_streets import cross_streets_from_model

    for leg_name, crossings in cross_streets_from_model(model).items():
        for cross in crossings:
            near_ft, far_ft = cross.mouth_ft
            for side in cross.sides:
                openings.setdefault((leg_name, side), []).append(
                    KerbOpening(start_ft=max(near_ft, 0.0), end_ft=far_ft,
                                source=OpeningSource.CROSS_STREET, way_id=cross.way_id))
    for key in openings:
        openings[key].sort(key=lambda o: o.start_ft)
    return openings


def _driveway_meetings(model) -> list[tuple[str, str, float, int | None]]:
    """(leg, side, station, way id) wherever a mapped driveway reaches a modelled kerb.

    The SECOND signal, read for the reason the module docstring gives: a driveway drawn without
    its kerb tagged is still a driveway, and reading only the kerb left the markings running
    straight across it.

    Which kerb it meets is decided by distance to the kerb LINE rather than by the driveway's own
    direction, because a driveway is drawn from the property to the road and its last segment is
    not reliably square to anything. The station is where it lands on the leg's centreline, taken
    from the point on the KERB nearest the driveway - not the driveway's own endpoint, which a
    mapper may have stopped short of or run past the kerb.
    """
    from shapely.ops import nearest_points

    meetings = []
    for drive in getattr(model, "driveways", ()):
        line = drive.line
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
            meetings.append((leg_name, side, station_ft, drive.way_id))
    return meetings


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
        half_ft = leg.curb_to_curb_ft / 2
        # The median rather than the mean, so one stray vertex - a way that turns up a driveway,
        # or runs on past the leg - cannot decide which street the kerb is on.
        typical_offset = float(np.median(offsets))
        typical_station = float(np.median(stations))
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
