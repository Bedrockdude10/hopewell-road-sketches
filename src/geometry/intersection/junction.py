"""What a junction IS, once it has been built: the model every phase reads, and the value
types it holds.

Nothing here loads anything. Splitting the built thing from the building of it is the point -
IntersectionModel is imported by 20 modules that have no business pulling in OSM fetching, road
matching or kerb fitting to get at it."""
from dataclasses import dataclass, field
from pathlib import Path

from enum import StrEnum

import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon

from src.geometry.model import (
    Leg,
)


ROOT_DIR = Path(__file__).resolve().parents[3]  # src/geometry/intersection/junction.py -> repo root


@dataclass
class IntersectionModel:
    config: dict
    center_wgs84: Point
    center_ft: Point
    legs: dict[str, Leg]
    corner_fillets: dict
    parcels: gpd.GeoDataFrame
    corner_parcels: gpd.GeoDataFrame
    # {leg name: [RoadSpan]} - every OSM highway way lying along the leg, with the stretch of
    # it each one covers. A LIST because what a way says varies along a street and OSM
    # expresses that by splitting the way; see RoadSpan for the restriction that was being
    # dropped when this was one way per leg.
    leg_road_spans: dict = field(default_factory=dict)
    # {leg name: OSM tags of the way covering MOST of the leg}. For whole-leg facts only -
    # overtaking=no is what a double-yellow centerline means, and reading it beats defaulting
    # every leg to a dashed line. Anything that varies along the leg has to read
    # leg_road_spans instead, which is exactly the distinction kerbside parking needs.
    leg_osm_tags: dict = field(default_factory=dict)
    # {leg name: True if that way runs the same way the leg points outward}. OSM's left/right
    # are relative to the WAY's direction; a leg's are relative to its own outward direction.
    # Where they disagree the sides swap. Per span in leg_road_spans, since two ways covering
    # one leg can be drawn in opposite directions.
    leg_osm_aligned: dict = field(default_factory=dict)
    # [PavedSurface] - every mapped driveway, parking aisle and parking lot near this junction,
    # projected once. Paved ground that is not carriageway: drawn as asphalt in both views, and in
    # the driveways' case read as an opening signal too (src/geometry/kerbs.py). See PavedSurface
    # for why they live on the model rather than being fetched per consumer, and why all three are
    # one type.
    paved_surfaces: tuple = ()
    # {leg name: the working length the SITE configured}, before the frame scale carried it out.
    # The surveyed answer, kept because two things must not move when the picture zooms: the
    # frame (src/render/frame.py:leg_reach_ft measures against this, so widening does not
    # compound) and the metrics (src/metrics.py reports anything past it as projected).
    surveyed_leg_lengths: dict = field(default_factory=dict)
    # {leg name: [CrossStreet]} - every OTHER street these legs run across, resolved ONCE here.
    # It was derived twice, for the kerb openings and for the DesignState, which is the failure
    # PavedSurface's docstring is about: two consumers assembling the same geometry and free to
    # diverge. R.S. 39:4-138(e) applies at every one of them - see src/geometry/cross_streets.py.
    cross_streets: dict = field(default_factory=dict)

    @property
    def site_roadways(self) -> tuple:
        """The LINEAR minor carriageways - what opens a kerb where one of them meets it.

        MUTCD 11th ed. 1C.02(113)(b)'s own list is "an alley, driveway, or site roadway", which is
        the same taxonomy OSM writes as `service=*`, so this is that list read off the tag: a
        driveway and a parking aisle both open a kerb and neither is an intersection
        (src/geometry/kerbs.py:OpeningSource.is_an_intersection). A PARKING_LOT is an area behind
        a building and crosses no kerb of ours; a ROADWAY is a street, and a street meeting a leg
        is resolved as a CrossStreet, which is the affirmative arm of the same clause.

        This was `driveways`, aisles excluded, on the grounds that "an aisle inside [a mapped lot]
        reaches the street through a driveway that is mapped separately". True of the aisles
        inside a lot - 6 of the borough's 20 - and it says nothing about one that meets the street
        itself, whose mouth had no opening and whose markings ran straight across it.
        """
        return tuple(s for s in self.paved_surfaces
                     if s.kind in (PavedKind.DRIVEWAY, PavedKind.PARKING_AISLE))

    def parking_restriction_spans(self, leg_name: str) -> list[tuple]:
        """[(start_ft, end_ft, {"left": value, "right": value}, way_id)] in the LEG's frame.

        Every way along this leg, its stretch, and what it says about each kerb - already
        flipped into the leg's own left/right where the way runs against it. The spans are
        contiguous where OSM split a way and may be absent where nothing is mapped; a value of
        None means that way says nothing about that side, which is NOT the same as "none".
        """
        return [(span.start_ft, span.end_ft,
                 parking_restriction_by_side(span.tags, span.aligned), span.way_id)
                for span in self.leg_road_spans.get(leg_name, [])]


class OSMDataUnavailableError(RuntimeError):
    """Overpass could not be reached, so OSM-derived geometry can't be built.

    Distinct from "OSM has no data here", which is a legitimate finding this project
    reports and renders honestly. An unreachable server is not evidence of absence, and
    treating it as such silently downgrades every OSM-derived value to a placeholder.
    """


class PavedKind(StrEnum):
    """What a piece of paved ground beside the carriageway IS.

    A StrEnum so it travels to the 3D render and into the exported JSON as the OSM value it came
    from, the same reason KerbType is one - a reader of the geometry file sees `parking_aisle`,
    not an integer they have to look up.
    """
    DRIVEWAY = "driveway"                # highway=service + service=driveway
    PARKING_AISLE = "parking_aisle"      # highway=service + service=parking_aisle
    PARKING_LOT = "parking_lot"          # amenity=parking, mapped as an AREA
    ROADWAY = "roadway"                  # highway=* around the junction, past the modelled legs
# How wide each LINEAR kind is DRAWN. Assumptions, flagged as such wherever they surface - see
# PavedSurface.width_ft - because OSM maps these as centrelines and none of them carries a width
# here. A residential driveway; a two-way parking aisle at the low end of the 20-24 ft ITE range,
# and a one-way at 12, which is the one place a real tag (`oneway`) picks between them.
DRIVEWAY_DRAWN_WIDTH_FT = 10.0
PARKING_AISLE_WIDTH_FT = 20.0
PARKING_AISLE_ONEWAY_WIDTH_FT = 12.0
DRAWN_WIDTH_FT = {PavedKind.DRIVEWAY: DRIVEWAY_DRAWN_WIDTH_FT,
                  PavedKind.PARKING_AISLE: PARKING_AISLE_WIDTH_FT}


@dataclass(frozen=True)
class PavedSurface:
    """One piece of paved ground beside the carriageway - a driveway, a parking aisle, a lot.

    PART OF THE MODELLED STREET, and it took a correction to put it here. A driveway was added as
    render dressing, fetched and projected independently by the plan view and by the export; then
    it became a signal for where the kerbside markings open and got a THIRD independent fetch in
    src/geometry/kerbs.py, each with its own radius constant. That is the exact shape of the bug
    src/render/scene.py exists to prevent - three consumers each assembling the same geometry, free
    to diverge - committed again one layer down.

    So it is resolved once, at load, beside corner_parcels and leg_road_spans: a surveyed fact
    about this junction's street network, not something each renderer looks up for itself. A
    driveway IS street geometry - it is where vehicles cross the kerb, and it is the reason a
    marking stops.

    ONE TYPE FOR ALL THREE, because they differ in exactly two ways and are otherwise the same
    thing: paved ground that is not carriageway, drawn as asphalt in both views. What differs is
    whether the extent was surveyed (a lot is mapped as an area; a driveway and an aisle are
    centrelines this project widens) and whether it opens the kerb (a driveway does, and
    src/geometry/kerbs.py reads only those - a lot behind a building crosses no kerb of ours).
    Adding parking as its own parallel field, with its own fetch, its own export key and its own
    branch in each renderer, is the same mistake the docstring above is about.
    """
    kind: str = PavedKind.DRIVEWAY
    #: The centreline, for the kinds OSM maps as a way. None for a lot, which is mapped as an area.
    line: LineString | None = None
    way_id: int | None = None
    tags: dict = field(default_factory=dict)
    #: The paved ground itself, built once so the plan view and the 3D render draw the SAME
    #: polygon rather than each widening the line their own way.
    surface: Polygon | None = None
    #: Which sides of a ROADWAY had a traced kerb to measure the edge from - {"left", "right"},
    #: one of them, or empty. Empty for every other kind, which has no two edges to speak of.
    traced_sides: frozenset = frozenset()
    #: The width this project ASSUMED, where it had to assume one. Carried per surface rather
    #: than looked up per kind because a roadway's assumption is per highway class and a
    #: one-way aisle is 12 ft where a two-way is 20 - the table cannot express either.
    drawn_width_ft: float | None = None

    @property
    def extent_is_surveyed(self) -> bool:
        """Whether somebody traced this outline, or this project widened a line into it.

        A parking lot is mapped as an area and its extent is as surveyed as a building footprint.
        A driveway and an aisle are centrelines with no width on them - 0 of the borough's 43
        driveways and 0 of its 20 aisles carry a `width` tag - so their strips are as wide as
        DRAWN_WIDTH_FT says, which is an assumption and is labelled as one in the legend.

        A ROADWAY is the one kind that can be either, which is the whole point of it: both of
        Broad Street's kerbs are traced for the length of the corridor, so that surface is
        measured on both edges and as surveyed as the lot. With one side traced or none, it is
        not - half a measured outline is still a guess about where the street ends.
        """
        if self.kind == PavedKind.ROADWAY:
            return self.traced_sides == frozenset({"left", "right"})
        return self.kind == PavedKind.PARKING_LOT

    @property
    def width_ft(self) -> float | None:
        """How wide the strip is drawn, for the kinds that needed a width. ASSUMED.

        For a driveway, the number that is NOT this: the width of the OPENING it makes in the kerb,
        which IS surveyed (the extent of the `kerb=lowered` section) and is what the gap in the
        markings uses (src/geometry/kerbs.py). The two must not be swapped - at E Broad the dropped
        kerb runs 37 ft while the driveway centreline enters near one end of it, so that section is
        a frontage the driveway opens onto, not the driveway's own width.

        `drawn_width_ft` where the surface recorded its own, because the kind no longer determines
        it: a one-way parking aisle is built at 12 ft and used to REPORT 20, the table's two-way
        figure, which is the quiet over-claim the rest of this class is arranged against.
        """
        if self.extent_is_surveyed:
            return None
        return self.drawn_width_ft if self.drawn_width_ft is not None else DRAWN_WIDTH_FT[self.kind]


@dataclass(frozen=True)
class RoadSpan:
    """One OSM highway way, and the stretch of one leg it covers.

    A LIST of these per leg, because the thing they carry varies along a street and OSM says so
    by SPLITTING THE WAY. That is how a kerbside parking restriction covering only the approach
    to a junction is expressed, and it is what this project was throwing away: the matcher kept
    the single way nearest the leg's MIDPOINT and dropped the rest, so at Broad & Greenwood a
    `parking:both:restriction=no_parking` tagged over East Broad's first 79.5 ft (way 1547092834)
    lost to the unrestricted way beyond it (11647647) by 1.9 ft against 5.8 - the leg's midpoint
    sits at station 85, past the split. The render then marked parking exactly where the mapper
    had just said there is none, and nothing anywhere reported a problem: a pipeline reading one
    way and finding no restriction looks identical to one that read the restriction and dropped it.

    `aligned` is per SPAN, not per leg: OSM's left/right are relative to the way's own direction,
    and two ways covering one leg can be drawn in opposite directions.
    """
    start_ft: float
    end_ft: float
    tags: dict
    aligned: bool
    way_id: int | None = None

    @property
    def length_ft(self) -> float:
        return max(self.end_ft - self.start_ft, 0.0)

    def covers(self, station_ft: float) -> bool:
        return self.start_ft <= station_ft <= self.end_ft


# OSM records kerbside parking per side of the way: parking:left:restriction,
# parking:right:restriction, or parking:both:restriction. Any value other than "none" is a
# prohibition of some kind (no_parking, no_standing, no_stopping); "none" is an explicit
# statement that parking IS allowed, which is different from the tag being absent.
PARKING_RESTRICTION_KEYS = {"left": "parking:left:restriction",
                            "right": "parking:right:restriction",
                            "both": "parking:both:restriction"}


def parking_restriction_by_side(tags: dict, aligned: bool) -> dict:
    """{"left": value|None, "right": value|None} in the LEG's frame.

    OSM's left and right are relative to the direction the way was drawn; a leg's are
    relative to the direction it points outward from the junction. Half this project's legs
    run against their way - Columbia Ave's west leg, Princeton Ave's north leg - so reading
    parking:left straight through would put the restriction on the wrong kerb for them, and
    it would look entirely plausible in the render.

    A value of None means OSM says nothing about that side. That is NOT the same as "none",
    which is a positive statement that parking is permitted.
    """
    both = tags.get(PARKING_RESTRICTION_KEYS["both"])
    if both is not None:
        return {"left": both, "right": both}
    osm = {side: tags.get(key) for side, key in PARKING_RESTRICTION_KEYS.items() if side != "both"}
    if aligned:
        return {"left": osm["left"], "right": osm["right"]}
    return {"left": osm["right"], "right": osm["left"]}


def parking_is_restricted(restriction: str | None) -> bool:
    """True where OSM prohibits kerbside parking. Absent or "none" is not a prohibition."""
    return restriction is not None and restriction != "none"
