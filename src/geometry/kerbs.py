"""What OSM says about a kerb: whether it is raised, and where it is dropped for a vehicle.

The tags were already reaching this pipeline and being thrown away at the last step. Every one
of the 95 `barrier=kerb` ways mapped in Hopewell carries `kerb=raised` or `kerb=lowered` - 67
and 28 of them, none untagged - and `fetch_kerbs` keeps every tag, and
`kerb_lines_with_tags_ft` hands back `(line, tags)` pairs. But nothing read `tags["kerb"]`: the
plan view drew every traced kerb as the same black line, the 3D render drew no kerb at all, and
the kerbside paint ran unbroken along the whole leg. So a surveyor's careful raised/lowered
distinction reached the geometry and stopped there, which is this project's signature failure -
ground truth present, never reaching the render.

WHY A LOWERED KERB IS THE OPENING. A driveway is mapped two ways at once: as a
`highway=service` + `service=driveway` way running up to the road, and as the stretch of kerb it
crosses being tagged `kerb=lowered`. The second is the better signal for paint, and not just
because it is already fetched:

  * it is ON the kerb, so it already has the geometry and the station span the paint needs,
    where a driveway way only asserts that a driveway exists somewhere near here;
  * it is what is physically true - the kerb is dropped, which is why a car can cross it;
  * it is present where the driveway way is not. Only ONE of the 43 driveways mapped in the
    borough reaches a kerb this project models (way 772378207, on e_broad_st_east's left kerb at
    station 88); the rest are further down the block, outside a 130 ft leg. The lowered kerb at
    that driveway is tagged, and so are lowered stretches with no driveway way drawn to them.

A PEDESTRIAN RAMP IS ALSO A LOWERED KERB, and it must not open the paint - a crosswalk's kerb
ramp is dropped for a wheelchair, not for a car, and the markings there are cut by the crossing
band instead. `tactile_paving=yes` separates the two, and at these four junctions it separates
them cleanly: of the nine lowered kerbs along E Broad's legs, three are tagged
`tactile_paving=yes` and sit exactly on a crossing, and the one at the driveway is tagged
`tactile_paving=no`.

That leaves a residue this module deliberately does NOT guess about: four lowered kerbs at
E Broad are 9-59 ft from the nearest crossing and nowhere near a driveway. They open the paint,
because the tags say the kerb is dropped and not a pedestrian ramp and that is the best
available reading - but `describe_kerb_openings` lists every opening with the way that produced
it, so a gap in a marking is reviewable against the survey rather than being a gap nobody can
account for.
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

    Dropped or flush, and not a pedestrian ramp. See the module docstring for why
    tactile_paving is the discriminator and what it leaves unresolved.
    """
    if not KerbType.from_tags(tags).is_crossable_by_a_vehicle:
        return False
    return (tags or {}).get("tactile_paving") != "yes"


@dataclass(frozen=True)
class KerbOpening:
    """A stretch of ONE KERB of one leg that is dropped for vehicles to cross.

    A span and not a point, for the same reason ParkingRestriction is one: a driveway mouth is
    12-30 ft of kerb, and the paint has to break over its width rather than at a station.

    `citation` names the way it came from, so a gap in the markings on a drawing can be traced
    back to the OSM object that put it there - the same accounting ParkingRestriction.citation
    gives a hatched kerb.
    """
    start_ft: float
    end_ft: float
    kerb: KerbType
    way_id: int | None = None

    @property
    def length_ft(self) -> float:
        return max(self.end_ft - self.start_ft, 0.0)

    @property
    def citation(self) -> str:
        return (f"OSM kerb={self.kerb.value}"
                + (f" on way {self.way_id}" if self.way_id is not None else ""))


# How far outside a leg's nominal half-width a kerb vertex may sit and still be that leg's kerb.
# The same allowance intersection.py's own kerb-to-leg test uses, and for the same reason: a
# traced kerb flares through the corner returns well past the mid-block cross-section.
OPENING_OFFSET_TOLERANCE_FT = 8.0
# A dropped kerb shorter than this is not a driveway mouth - it is the last vertex of a ramp or
# a tracing artifact, and breaking the paint for it would put a nick in a marking.
MIN_OPENING_LENGTH_FT = 4.0


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
            KerbOpening(start_ft=start_ft, end_ft=end_ft, kerb=KerbType.from_tags(tags),
                        way_id=way_id))
    for key in openings:
        openings[key].sort(key=lambda o: o.start_ft)
    return openings


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

    Provenance, not diagnostics: a driveway gap in a drawing is a claim about the street, and the
    reviewer needs to be able to check it against OSM. Every line names the kerb way, so an
    opening in the wrong place is traceable to a tag rather than to this code. Follows
    src/render/props.py:data_gaps in shape - a list of sentences the phase script prints.
    """
    lines = []
    for (leg_name, side), openings in sorted(state.kerb_openings.items()):
        for opening in openings:
            lines.append(
                f"{leg_name} {side}: kerbside markings break over {opening.start_ft:.0f}-"
                f"{opening.end_ft:.0f} ft ({opening.length_ft:.0f} ft of dropped kerb) - "
                f"{opening.citation}. A vehicle crosses here, so the paint opens for it.")
    return lines
