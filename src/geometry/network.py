"""A ROAD: one street running through a junction, with continuous stationing.

STEP 1 OF docs/network-model.md, and deliberately nothing more. This module builds Roads
ALONGSIDE the existing legs and renders nothing from them. Its whole job is to answer the
checkpoint question that document asks before any of the migration is committed to:

    does the traced kerb, read as one continuous road, reproduce the widths the per-leg model
    already produces?

If it does, the frame can be moved onto roads with confidence. If it does not, the difference is
the finding, and it is much cheaper to learn here than after 14k lines have been rewritten.

WHY A ROAD AND NOT TWO LEGS. A leg starts at the junction and runs outward, so a street through a
junction is two legs pointing away from each other, each with its own station 0 and its own frame.
Every consequence of that has cost this project something:

  * a marking measured from one leg cannot be continued onto the other. The two-way bike lane's
    halves ended 1.28 ft apart at W Broad & Louellen for exactly this reason (fixed in f71a7a1, by
    letting each half reach behind its own node - a workaround for the decomposition, not a fix).
  * a corridor question has no object to ask. `narrowest_half_width_ft` takes a leg.
  * A SURVEYED CROSSING AT A JUNCTION THIS SITE DOES NOT MODEL CANNOT BE DRAWN AT ALL, because a
    crossing needs a station, an orientation and a reach to both kerbs, and all three come from a
    leg. At Broad & Greenwood framed at 2.5x, 6 of the 10 OSM crossings inside the picture are
    discarded, three of them tagged as zebra markings. That is the reason this is being done now.

A Road here is the two through legs' centrelines joined head-to-head. They are ALREADY joined
tangentially at the node by intersection/fitting.py:_join_through_legs, so this is a re-reading of
geometry the model has, not a second construction of it - which is what makes the comparison below
meaningful rather than circular.

NOT YET A ROAD IN THE FULL SENSE. It stops at the two modelled legs' far ends rather than running
the length of the borough, and it carries no station-ranged facts. Both are step 2 and step 3. What
it does have is one centreline, one kerb line per side, and one station axis.
"""
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString

from src.geometry.model import (curb_offsets_at_stations, curb_station_span, is_through_street,
                                leg_bearing_deg)


@dataclass(frozen=True)
class Road:
    """One street through a junction: a continuous centreline and a kerb line per side.

    `near`/`far` name the two legs it was built from - `near` is the one whose centreline runs
    BACKWARDS along this road (station 0 is at its far end) and `far` the one that runs forwards.
    `node_ft` is the station of the junction itself, which is where the two legs' own station 0s
    both are.

    Sides are the ROAD's, not either leg's, and that is the point of the object: leg A's left kerb
    and leg B's right kerb are one physical kerb (see model.through_street_sides, which pairs them
    that way for corner fillets), so on the road they are one line with one name.
    """
    name: str
    centerline: LineString
    node_ft: float
    near_leg: str
    far_leg: str
    left_kerb: LineString | None = None
    right_kerb: LineString | None = None

    @property
    def length_ft(self) -> float:
        return self.centerline.length

    def width_at_ft(self, station_ft: float) -> float | None:
        """Kerb to kerb at one station, or None where either side is untraced there.

        The road's own answer to the question `Leg.curb_to_curb_ft` answers per leg - and the
        comparison between the two is what tests/test_network.py checks.
        """
        offsets = []
        for side in ("left", "right"):
            kerb = self.left_kerb if side == "left" else self.right_kerb
            if kerb is None:
                return None
            span = curb_station_span(_AsLeg(self.centerline, kerb, side), side)
            if span is None or not (span[0] <= station_ft <= span[1]):
                return None
            at = curb_offsets_at_stations(_AsLeg(self.centerline, kerb, side), side,
                                          np.array([station_ft]))
            if at is None or at[0] is None or not np.isfinite(at[0]):
                return None
            offsets.append(abs(float(at[0])))
        return sum(offsets)


@dataclass
class _AsLeg:
    """The two attributes curb_offsets_at_stations reads, so a Road can borrow the leg frame.

    A shim rather than a refactor, on purpose: step 1 must not touch the frame (that is step 4),
    and the frame functions only ever look at `centerline` and one of the two curb attributes.
    Written down as a class so what is being borrowed is explicit instead of duck-typed by
    accident.
    """
    centerline: LineString
    kerb: LineString
    side: str

    @property
    def left_curb(self):
        return self.kerb if self.side == "left" else None

    @property
    def right_curb(self):
        return self.kerb if self.side == "right" else None


def _joined_centerline(near, far) -> LineString:
    """`near`'s centreline reversed, then `far`'s - one line running through the junction.

    Both legs start AT the node and run outward, so the near one is reversed to run inward. The
    shared node appears once: _join_through_legs has already given both legs the same first point,
    so the duplicate is dropped rather than left as a zero-length segment for `project` to trip on.
    """
    back = list(near.centerline.coords)[::-1]
    ahead = list(far.centerline.coords)
    if back and ahead and np.allclose(back[-1], ahead[0]):
        ahead = ahead[1:]
    return LineString(back + ahead)


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


def roads_from_model(model) -> list[Road]:
    """Every street that runs THROUGH this junction, as a Road.

    Only through pairs: a stem (Louellen at W Broad, Princeton at E Broad) is one leg and already
    has a leg frame that covers it, so there is nothing a Road would add until step 3 gives it
    station-ranged facts. Reuses the same pairing intersection/fitting.py:_through_leg_pairs found
    when it joined the centrelines, rather than re-deriving it - two answers to "which legs are one
    street" is the kind of second definition this whole document is about.
    """
    from src.geometry.intersection.fitting import _through_leg_pairs

    roads = []
    for name_a, name_b in _through_leg_pairs(model.legs):
        leg_a, leg_b = model.legs[name_a], model.legs[name_b]
        if not is_through_street(leg_a, leg_b):
            continue
        # `near` is whichever runs backwards along the finished road. Chosen by bearing so the
        # station axis is stable rather than depending on dict order.
        if leg_bearing_deg(leg_a) <= leg_bearing_deg(leg_b):
            near, near_name, far, far_name = leg_a, name_a, leg_b, name_b
        else:
            near, near_name, far, far_name = leg_b, name_b, leg_a, name_a
        street = (model.config["legs"].get(far_name, {}).get("street_name")
                  or model.config["legs"].get(near_name, {}).get("street_name")
                  or f"{near_name}/{far_name}")
        roads.append(Road(
            name=street,
            centerline=_joined_centerline(near, far),
            node_ft=near.centerline.length,
            near_leg=near_name,
            far_leg=far_name,
            # The road's LEFT is the near leg's RIGHT: reversing the near leg's direction of travel
            # swaps its sides. Getting this backwards would compare one kerb against itself and
            # report a plausible-looking width that is not the road's.
            left_kerb=_joined_kerb(near, "right", far, "left"),
            right_kerb=_joined_kerb(near, "left", far, "right"),
        ))
    return roads


def road_station_of_leg_station(road: Road, leg_name: str, leg_station_ft: float) -> float:
    """Where a station measured along one LEG falls on the road's own axis.

    The translation the migration turns on, and the reason it is one function: everything that
    currently says "42 ft along broad_st_east" has to keep meaning the same place once the datum
    is the road.
    """
    if leg_name == road.far_leg:
        return road.node_ft + leg_station_ft
    if leg_name == road.near_leg:
        return road.node_ft - leg_station_ft
    raise KeyError(f"{leg_name} is not one of {road.name}'s two legs "
                   f"({road.near_leg}, {road.far_leg})")
