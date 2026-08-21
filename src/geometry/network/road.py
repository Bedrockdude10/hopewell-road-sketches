"""A ROAD: one street with continuous stationing, through one junction.

STEP 1 OF docs/network-model.md. Nothing renders from a Road; its job is the checkpoint question
that document asks - does the traced kerb, read as one continuous road, reproduce the widths the
per-leg model already produces?

WHY A ROAD AND NOT TWO LEGS. A leg starts at the junction and runs outward, so a street through a
junction is two legs pointing away from each other, each with its own station 0 and its own frame.
A marking measured from one leg cannot be continued onto the other. A corridor question has no
object to ask. And a surveyed crossing at a junction this site does not model cannot be drawn at
all, because a crossing needs a station, an orientation and a reach to both kerbs, and all three
come from a leg.

A Road here is the two through legs' centrelines joined head-to-head. They are ALREADY joined
tangentially at the node by intersection/fitting.py:_join_through_legs, so this is a re-reading of
geometry the model has, not a second construction of it - which is what makes the comparison
meaningful rather than circular.

An `Approach` is the other direction: a leg's own frame, recovered from the road, so a per-leg
caller can keep asking per-leg questions of a corridor-scale object.
"""
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString
from shapely.ops import substring

from src.geometry.model import (Alignment, curb_edge_by_station, curb_offsets_at_stations, curb_station_span,
                                is_through_street, leg_bearing_deg)
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: this type is layered above this package, so importing it
    # for real would close a cycle.
    from src.geometry.intersection.junction import IntersectionModel


@dataclass(frozen=True)
class Road:
    """One street through a junction: a continuous centreline and a kerb line per side.

    `near`/`far` name the two legs it was built from - `near` runs backwards along this road
    (station 0 is at its far end), `far` runs forwards. `node_ft` is the station of the junction.

    Sides are the ROAD's, not either leg's, and that is the point of the object: leg A's left kerb
    and leg B's right kerb are one physical kerb (see model.through_street_sides), so on the road
    they are one line with one name.
    """
    name: str
    centerline: LineString
    node_ft: float
    near_leg: str
    far_leg: str
    #: The frame's own attribute names (src/geometry/model/leg_frame.py:Alignment), not
    #: `left_kerb`/`right_kerb`. A Road IS an alignment - one centreline, one traced kerb per
    #: side - so under these names every frame function takes a Road unmodified, and moving the
    #: datum off the leg becomes a change of caller rather than a rewrite of the frame.
    left_curb: LineString | None = None
    right_curb: LineString | None = None
    #: Where the FAR leg's own station 0 sits on this road, which is `node_ft` plus whatever of the
    #: joint _joined_centerline could not close. Both legs start at the node, so in principle the
    #: two are equal; at W Broad & Louellen 2.79 ft of the gap between the two NJDOT alignments is
    #: longitudinal and survives the lateral blend, and the joined line carries it as real length.
    #: Translating a far-leg station arithmetically then landed 2.79 ft up the street and read the
    #: road 2.9 ft wider than the leg it was built from.
    far_node_ft: float | None = None

    @property
    def length_ft(self) -> float:
        return self.centerline.length

    def width_at_ft(self, station_ft: float) -> float | None:
        """Kerb to kerb at one station, or None where either side is untraced there.

        The road's own answer to the question `Leg.curb_to_curb_ft` answers per leg - and the
        comparison between the two is what tests/test_network.py checks.
        """
        offsets = [_kerb_offset_at(self.centerline, self.left_curb, "left", station_ft),
                   _kerb_offset_at(self.centerline, self.right_curb, "right", station_ft)]
        return None if any(o is None for o in offsets) else sum(offsets)


def _kerb_offset_at(centerline: LineString, kerb: LineString | None, side: str,
                     station_ft: float) -> float | None:
    """How far out one kerb sits at one station, unsigned, or None where it is not traced there.

    The single place a kerb is read in a road's frame. Refusing OUTSIDE THE TRACED SPAN is the
    load-bearing part: np.interp is happy to extend the first and last offset flat forever, so
    without the span test a corridor with 1,126 ft of untraced kerb would report a width across it.
    """
    if kerb is None:
        return None
    one = Alignment.one_sided(centerline, side, kerb)
    span = curb_station_span(one, side)
    if span is None or not (span[0] <= station_ft <= span[1]):
        return None
    at = curb_offsets_at_stations(one, side, np.array([station_ft]))
    if at is None or at[0] is None or not np.isfinite(at[0]):
        return None
    return abs(float(at[0]))


#: Two vertices this close are one point. A hair, in feet: the shared junction node is written by
#: the same code into both legs, so a real duplicate is exact and anything larger is a gap.
SAME_POINT_FT = 1e-6


def _same_point(a, b) -> bool:
    return bool(np.hypot(*(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))) <= SAME_POINT_FT)


def _joined_centerline(near, far) -> tuple[LineString, float]:
    """`near`'s centreline reversed, then `far`'s - one line running through the junction.

    Both legs start AT the node and run outward, so the near one is reversed to run inward. The
    shared node appears once: _join_through_legs has already given both legs the same first point,
    so the duplicate is dropped rather than left as a zero-length segment for `project` to trip on.

    EXCEPT WHERE IT HAS NOT. _blend_onto applies the shared junction point as a lateral offset
    profile, taking station_offset_many's offset and discarding its station, so it can slide a
    leg's end sideways onto the joint but never along the street. At W Broad & Louellen, 2.74 ft
    of the gap between the two NJDOT alignments is longitudinal and survives. Closing it here
    would be a second opinion about where the street is; one road built once has no joint to
    disagree about, which is the actual fix (task: retire _join_through_legs).
    """
    back = list(near.centerline.coords)[::-1]
    ahead = list(far.centerline.coords)
    joint_ft = 0.0
    # BY DISTANCE, NOT np.allclose. These are state-plane feet around 420,000, and allclose's
    # default rtol of 1e-5 is 4.2 ft there - so it called the two starts "the same point" across a
    # 2.79 ft gap and dropped one, and the joined line quietly lost that length. Every far-leg
    # station past the node was then 2.79 ft out, which read the road 2.9 ft wider than the leg it
    # was built from at W Broad & Louellen.
    if back and ahead and _same_point(back[-1], ahead[0]):
        ahead = ahead[1:]
    elif back and ahead:
        # What the blend could not close, carried as real length by the line below. Returned so a
        # caller can put the far leg's station 0 where it actually lands - see Road.far_node_ft.
        joint_ft = float(np.hypot(*(np.asarray(ahead[0]) - np.asarray(back[-1]))))
    return LineString(back + ahead), near.centerline.length + joint_ft


def _joined_kerb(near, near_side: str, far, far_side: str) -> LineString | None:
    """One physical kerb, from the two legs' halves of it.

    near_side/far_side are opposite by construction - leg A's left is leg B's right - which is
    model.through_street_sides' pairing and the reason those two are one unbroken kerb with no
    corner in it.
    """
    a = getattr(near, f"{near_side}_curb", None)
    b = getattr(far, f"{far_side}_curb", None)
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return LineString(list(a.coords)[::-1])
    coords = list(a.coords)[::-1] + list(b.coords)
    return LineString(coords)


def roads_from_model(model: "IntersectionModel") -> list[Road]:
    """Every street that runs THROUGH this junction, as a Road.

    Only through pairs: a stem (Louellen at W Broad, Princeton at E Broad) is one leg and already
    has a leg frame that covers it. Reuses the same pairing
    intersection/fitting.py:_through_leg_pairs found when it joined the centrelines, rather than
    re-deriving it.
    """
    from src.geometry.intersection.fitting import _through_leg_pairs

    roads = []
    for name_a, name_b in _through_leg_pairs(model.legs):
        leg_a, leg_b = model.legs[name_a], model.legs[name_b]
        if not is_through_street(leg_a, leg_b):
            continue
        # `near` is whichever runs backwards along the finished road, chosen by bearing so the
        # station axis is stable.
        if leg_bearing_deg(leg_a) <= leg_bearing_deg(leg_b):
            near, near_name, far, far_name = leg_a, name_a, leg_b, name_b
        else:
            near, near_name, far, far_name = leg_b, name_b, leg_a, name_a
        street = (model.config["legs"].get(far_name, {}).get("street_name")
                  or model.config["legs"].get(near_name, {}).get("street_name")
                  or f"{near_name}/{far_name}")
        joined, far_node_ft = _joined_centerline(near, far)
        roads.append(Road(
            name=street,
            centerline=joined,
            # The NEAR leg's own station 0, which is exact for it and out by the whole unclosed
            # gap for the far one - see _joined_centerline (task: retire _join_through_legs).
            node_ft=near.centerline.length,
            near_leg=near_name,
            far_leg=far_name,
            # The road's LEFT is the near leg's RIGHT: reversing the near leg's direction swaps
            # its sides.
            left_curb=_joined_kerb(near, "right", far, "left"),
            right_curb=_joined_kerb(near, "left", far, "right"),
            # From _joined_centerline, which is the only thing that knows whether the shared point
            # was dropped: where the two legs really do start together this is exactly node_ft, and
            # only an open joint makes it larger. Projecting the far leg's start onto the joined
            # line instead was right in principle and 4e-4 ft noisy in the closed case.
            far_node_ft=far_node_ft,
        ))
    return roads


@dataclass(frozen=True)
class Approach:
    """One direction of one road, at one node. What a Leg NAMED, holding nothing a Leg HELD.

    docs/network-model.md says "there is no Leg. Not 'a leg becomes a view' - the object goes
    away". What goes away is a leg OWNING geometry. This owns none: a name, a road, a node and a
    direction, and every line, kerb, station and width it can be asked for is derived from the road
    on the spot.

    `forward` is whether this approach runs the way the road's stations increase. A road is built
    head-to-head from two legs, so exactly one of its two approaches runs against it, and that
    one's left kerb is the road's right (see roads_from_model, which pairs the sides that way).
    """
    name: str
    road: Road
    node_ft: float
    forward: bool

    def station_of(self, approach_ft: float) -> float:
        """Where a distance measured OUTWARD from the node falls on the road's own axis.

        The translation the migration turns on: everything that says "42 ft along broad_st_east"
        has to keep meaning the same place once the datum is the road.
        """
        return self.node_ft + (approach_ft if self.forward else -approach_ft)

    def outward_ft(self, road_station_ft: float) -> float:
        """The inverse: how far out from the node a road station is, along this approach.

        Negative behind the node, which is a real place - the far side of the junction - and not
        an error.
        """
        return (road_station_ft - self.node_ft) * (1.0 if self.forward else -1.0)

    def side_on_road(self, side: str) -> str:
        """This approach's left/right, named as the ROAD's left/right."""
        if self.forward:
            return str(side)
        return "right" if str(side) == "left" else "left"

    @property
    def span_ft(self) -> tuple[float, float]:
        """The road stations this approach covers, low to high."""
        return (self.node_ft, self.road.length_ft) if self.forward else (0.0, self.node_ft)

    @property
    def length_ft(self) -> float:
        lo, hi = self.span_ft
        return hi - lo

    @property
    def centerline(self) -> LineString:
        """The road's own centreline over this approach's span, running OUTWARD from the node.

        A view, cut on demand. Nothing caches it, and nothing may edit it: the moment an approach
        keeps its own copy of the line, the copy can disagree with the road.
        """
        lo, hi = self.span_ft
        cut = substring(self.road.centerline, lo, hi)
        return cut if self.forward else LineString(list(cut.coords)[::-1])

    def curb(self, side: str) -> LineString | None:
        """This approach's kerb on one side - the ROAD's kerb, cut at the node.

        Through curb_edge_by_station, so the kerb's own traced vertices are what comes back and
        only the two ends are interpolated. Cutting it any other way would resample the
        surveyor's line onto a grid and hand back this project's redrawing of it.
        """
        lo, hi = self.span_ft
        edge = curb_edge_by_station(self.road, self.side_on_road(side), lo, hi)
        if edge is None or len(edge) < 2:
            return None
        line = LineString(edge)
        return line if self.forward else LineString(list(line.coords)[::-1])

    @property
    def alignment(self) -> Alignment:
        """This approach as the frame reads it: a centreline and a kerb per side, outward."""
        return Alignment(self.centerline, left_curb=self.curb("left"),
                         right_curb=self.curb("right"))


def approaches_of(road: Road) -> tuple[Approach, ...]:
    """A road's two approaches at its node, named by the legs it was built from."""
    return (Approach(road.far_leg, road, road.far_node_ft if road.far_node_ft is not None
                     else road.node_ft, forward=True),
            Approach(road.near_leg, road, road.node_ft, forward=False))


def road_station_of_leg_station(road: Road, leg_name: str, leg_station_ft: float) -> float:
    """Where a station measured along one LEG falls on the road's own axis.

    Kept as the name 20 call sites will reach for during the migration; the arithmetic lives on
    Approach.
    """
    for approach in approaches_of(road):
        if approach.name == leg_name:
            return approach.station_of(leg_station_ft)
    raise KeyError(f"{leg_name} is not one of {road.name}'s two legs "
                   f"({road.near_leg}, {road.far_leg})")
