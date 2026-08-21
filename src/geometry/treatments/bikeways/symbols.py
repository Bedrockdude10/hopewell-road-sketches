"""THE BIKE SYMBOL AND THE CONTRAFLOW DASH: how a lane says what it is.

Independent of everything else here - a symbol needs a line to sit on and a width, not a section
and not a state - so it is the other leaf, and a caller wanting the stencil pattern does not have
to read the treatments to find it.
"""
import numpy as np

from src.geometry.markings import BIKE_CONTRAFLOW_DIVIDER

# The contraflow stripe's cadence. Shorter than the roadway's dashed centreline, because it is
# read at bicycle speed over a 12 ft lane rather than at 25 mph over a 40 ft one, and a stripe
# scaled to the road reads as two or three marks over a whole block.
# WHERE THE BIKE LANE SYMBOL GOES. NACTO asks for it after every driveway and intersection AND at
# least every 500 ft along the lane - both rules, not either, because on a corridor with 19
# junctions the interval alone leaves long unmarked stretches while the mouths alone cluster them
# where nobody needs reminding. MUTCD Fig 9E-1 is the marking. STANDARDS.md, verified 2026-08-18.
SYMBOL_INTERVAL_FT = 500.0
# How far past a mouth the reminder sits: clear of the opening itself, near enough to read as
# belonging to it. It is what keeps the symbol out of any opening, which is why BIKE_LANE_SYMBOL's
# AT_AN_OPENING row can be CARRIED - there is never one inside a mouth to cut.
SYMBOL_CLEAR_OF_OPENING_FT = 15.0
# The painted footprint. A schematic arrow rather than a drawn bicycle - see markings.py:
# BIKE_LANE_SYMBOL_POLYGONS for why, and the legend says which marking it represents.
SYMBOL_LENGTH_FT = 5.5
SYMBOL_WIDTH_FT = 2.4
# HOW FAR A PAIR OF STENCILS SITS CLEAR OF THE STRIPE BETWEEN THEM, in a two-way lane. The divider
# is not a hairline - its channel lays real paint about the axis - so half that stroke is ground the
# stencil may not have, and the width is READ OFF THE MARKING rather than restated here.
# The 0.25 ft on top is for CHORDING: the stripe is a run of 3 ft dashes between points on a curve
# while the stencil is placed at its own stations, so two markings at one nominal offset are not
# that far apart everywhere. Measured on the 2.5x sheet before this constant existed, the flat 0.4
# ft it replaces left 0.08 ft of clearance at Broad & Greenwood and none at all at W Broad &
# Louellen, where 0.12 sq ft of stencil went under the stripe and MarkingsDoNotCollide said so.
# NOT through paint.stroke_width_ft, which is the one home for this conversion and is unreachable
# from here: anything under src.geometry.paint runs that package's __init__, which reaches
# anchors -> render.crosswalks -> treatments, and a treatment importing it is a cycle.
_stroke_m = BIKE_CONTRAFLOW_DIVIDER.channel.stroke_width_m if BIKE_CONTRAFLOW_DIVIDER.channel else 0
assert _stroke_m, "the contraflow divider is a LINE in a stroked channel, so it has a width"
SYMBOL_CLEAR_OF_DIVIDER_FT = _stroke_m / 0.3048 / 2 + 0.25

CONTRAFLOW_DASH_FT = 3.0
CONTRAFLOW_GAP_FT = 5.0


def bike_symbol_stations_ft(start_ft: float, end_ft: float, openings=()) -> list[float]:
    """Stations along one run of lane where a BIKE LANE symbol belongs.

    Both of NACTO's rules at once: one after every opening the lane crosses, and one at least
    every SYMBOL_INTERVAL_FT regardless. `openings` are (lo, hi) station pairs on this kerb.

    ONE PLACE, so the plan view, the 3D export and the corridor strip cannot disagree about how
    many symbols a design calls for - the same reason the section's own offsets live on BikeLane.
    """
    at = [start_ft + SYMBOL_CLEAR_OF_OPENING_FT]
    station = start_ft + SYMBOL_INTERVAL_FT
    while station < end_ft:
        at.append(station)
        station += SYMBOL_INTERVAL_FT
    for _lo, hi in openings:
        if start_ft < hi < end_ft:
            at.append(hi + SYMBOL_CLEAR_OF_OPENING_FT)
    # THINNED, because the two rules can land on top of each other: a mouth 15 ft before an
    # interval station puts two symbols in the same 5.5 ft of road, which the collision check
    # reads - correctly - as ground painted twice. One symbol per place; whichever rule asked
    # for it, the rider only needs telling once.
    kept: list[float] = []
    for station in sorted(s for s in at if start_ft <= s <= end_ft):
        if not kept or station - kept[-1] >= SYMBOL_LENGTH_FT * 1.5:
            kept.append(station)
    return kept


def bike_symbol_polygon(on, side: str, station_ft: float, centre_offset_ft: float,
                        forward: bool = True):
    """The symbol's painted footprint at one station, centred in the lane.

    An arrowhead on a shaft, pointing the way that half of the lane runs. `forward` is what makes
    a bidirectional lane's two halves face opposite ways, which is the whole reason a symbol earns
    its place here rather than being decoration: it tells a driver at a mouth which direction the
    rider bearing down on them is coming from.
    """
    from shapely.geometry import Polygon

    from src.geometry.model import place_in_measured_frame

    sign = 1.0 if side == "left" else -1.0
    nose = SYMBOL_LENGTH_FT / 2 * (1.0 if forward else -1.0)
    tail = -nose
    half = SYMBOL_WIDTH_FT / 2
    shaft = SYMBOL_WIDTH_FT / 6
    # (along, across) in the lane's own terms, then placed in the road frame once.
    outline = [(nose, 0.0), (nose - nose / 2, half), (nose - nose / 2, shaft),
               (tail, shaft), (tail, -shaft), (nose - nose / 2, -shaft),
               (nose - nose / 2, -half)]
    stations = np.array([station_ft + along for along, _across in outline])
    offsets = np.array([sign * (centre_offset_ft + across) for _along, across in outline])
    placed = place_in_measured_frame(on.centerline, stations, offsets)
    return Polygon([tuple(point) for point in placed])
