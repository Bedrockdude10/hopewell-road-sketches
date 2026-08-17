"""FITTING THE LEGS TO THE TRACED KERBS - the heaviest thing this package does.

A leg starts as NJDOT's alignment plus a width from config.yaml, and neither is a survey. Where the
kerbs have been traced they are better evidence than both, so the leg is resized onto them, centred
between them, and joined tangentially to its through partner. Every step is bounded (MATERIAL_*,
MAX_*) so a bad trace moves the leg a little or not at all rather than throwing it across the
street."""


import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point

from src.geometry.model import (
    Leg,
    place_in_measured_frame,
    unit_vector,
    assign_curb_points_to_legs,
    curb_line_from_points,
    curb_offsets_at_stations,
    through_street_sides,
    curb_station_span,
    station_offset_many,
)
from src.geometry.intersection.kerb_sources import (KERB_NEAR_JUNCTION_FT,
                                                    kerb_lines_with_tags_ft)

KERB_PLAUSIBLE_HALF_WIDTH_FT = (8.0, 45.0)  # a kerb this far off a centerline is that leg's kerb


def _widths_from_traced_kerbs(legs: dict, kerb_lines: list, legs_cfg: dict) -> dict:
    """First-pass leg half-widths from traced OSM kerbs: nearest approach, doubled.

    A corner return is TANGENT to its leg's curb line, so the closest approach of a traced
    kerb to a leg centerline is that leg's half-width. (Away from the tangent point the
    return curves off around the corner, so the measurement is an upper bound - the real
    curb is at most this far out, never further.)

    That one-sidedness is what makes it safe to apply: a traced kerb closer to the
    centerline than our modelled curb PROVES the modelled road is too wide. It is only
    used to narrow a leg, never to widen one, and never against a field measurement -
    those win outright (src/provenance.py), though a conflict is reported.

    WHY THIS IS ONLY A FIRST PASS. Doubling one side's distance assumes the centerline sits
    midway between the two kerbs, and NJDOT's route alignment frequently does not - it is
    10.4 ft off centre on w_broad_st_northeast, where CR 518 turns onto Louellen and the
    alignment cuts the corner. Doubling the NEAR kerb there gives 30.3 ft for a street the
    two traced kerbs measure at 35.6. Every leg at every site came out too narrow this way,
    by 1-6 ft. _resize_and_centre_from_traced_kerbs below re-measures each leg properly
    once both kerbs are in its frame, and that measurement governs.

    What this pass is still needed for: assign_curb_points_to_legs disqualifies a traced
    vertex whose |offset| / half-width falls outside CURB_POINT_MIN/MAX_WIDTH_RATIO, so a
    badly wrong configured width (w_broad_st_southwest's 50 ft parcel-gap estimate) throws
    away the very kerbs the second pass needs. This gets the width close enough to collect
    them, and nothing else depends on its result.
    """
    from src.provenance import field_measurement_governs_corner

    updates = {}
    for name, leg in legs.items():
        if leg.curb_to_curb_ft is None:
            continue
        if field_measurement_governs_corner(legs_cfg.get(name, {})):
            continue        # a width measured at this cross-section outranks any trace
        candidates = []
        for kerb in kerb_lines:
            distance = kerb.distance(leg.centerline)
            nearest_along = leg.centerline.project(kerb.interpolate(kerb.project(leg.centerline.centroid)))
            if not (0 < nearest_along < leg.centerline.length):
                continue
            if KERB_PLAUSIBLE_HALF_WIDTH_FT[0] <= distance <= KERB_PLAUSIBLE_HALF_WIDTH_FT[1]:
                candidates.append(distance)
        if not candidates:
            continue

        measured_half_ft = min(candidates)
        if measured_half_ft >= leg.curb_to_curb_ft / 2 - 0.5:
            continue  # traced kerb agrees, or sits outside our curb - nothing proven
        updates[name] = measured_half_ft * 2
    return updates


# How far outside its assumed half-width the SEEDING assignment will still claim a traced
# vertex. Deliberately loose: that pass is collecting the kerbs the widths will be measured
# FROM, so it must not throw one away for disagreeing with a width nobody has measured yet.
# At W Broad & Louellen the two real kerbs sit at 0.43x and 3.5x the seed half-width, and
# the normal window (0.45-2.6) excluded one of them on every leg. Later rounds use the
# normal window against a width that by then is a measurement.
SEED_RATIO_BOUNDS = (0.2, 5.0)

# A traced kerb within this of the junction is a corner RETURN, flaring out to as much as
# 2.3x the half-width (CURB_POINT_MAX_WIDTH_RATIO). A cross-section measured across two
# returns is the width of the corner, not of the street, so the measurement below starts
# beyond them. Same distance UNTRACED_CORNER_THRESHOLD_FT uses for the same reason.
TRACED_SECTION_START_FT = 35.0
# ...and it ENDS here, which is not the same as ending where the leg does. The window used to
# run to the traced curb line's far end, and a curb line is drawn to the leg's working length -
# so lengthening a leg to show more of it silently re-measured its width. Carrying
# broad_st_east from 130 to 170 ft moved it 52.0 -> 49.9 ft, because East Broad narrows as it
# leaves the junction and the extra 40 ft of narrower street pulled the median down. A width
# is a fact about the approach; how far we chose to DRAW the approach is a presentation
# choice, and a presentation choice may not move a measurement. So the cross-section is always
# taken over the same stretch of road whatever working_length_ft says.
TRACED_SECTION_END_FT = 130.0
# Below this there isn't enough of a traced overlap to call it a cross-section.
MIN_TRACED_SECTION_FT = 20.0
TRACED_SECTION_SAMPLES = 40
# How finely the kerbs' midpoint is sampled when the alignment is centred on it, and over how
# long a run each sample is averaged. The centring used to be a single constant per leg, on the
# reasoning that a striper lays a straight line - true of the line, but the thing being
# corrected is not a stripe. NJDOT's alignment is a linear-referencing reference that BENDS
# relative to the carriageway, and a constant cannot follow that: broad_st_east is centred to
# 0.07 ft over the 35-130 ft the constant was measured across and 4.28 ft off 290 ft out, where
# an 11 ft lane measured from it left 4.6 ft of kerb on one side and 13.0 ft on the other. The
# smoothing window is what keeps the result something a striper would lay - it follows the
# street's bend and not every wobble in the tracing.
CENTRE_SAMPLE_FT = 10.0
CENTRE_SMOOTH_FT = 60.0
# No two vertices of a corrected alignment closer than this - see _thinned.
MIN_CENTRE_VERTEX_GAP_FT = 2.5
# Over how far each half of a through street eases from the shared junction tangent back onto
# its own measured centre, and how finely that easing is sampled - see _join_through_legs.
# 60 ft is about the run over which a striper actually swings a centreline through a bend:
# much shorter reads as a kink again, much longer starts overriding the measurement it is
# meant to be blending into.
THROUGH_JOIN_BLEND_FT = 60.0
THROUGH_JOIN_SAMPLE_FT = 5.0
# How much the kerbs' midpoint may wander before a single CONSTANT stops describing it. Only
# the constant inside the fit is bounded by this now - past it the fit leaves the alignment
# alone and the final profile pass does the centring instead.
MAX_CENTRE_SPREAD_FT = 5.0
# A sanity bound on the correction. Past this the two "kerbs" are more likely to be one
# street's kerb and a neighbouring one's than the two sides of this leg.
MAX_CENTRE_SHIFT_FT = 15.0
# What counts as a real change between two rounds of the fit below, as opposed to the
# geometry jittering on the last decimal place and the loop never terminating.
MATERIAL_WIDTH_CHANGE_FT = 0.25
MATERIAL_SHIFT_FT = 0.1
MAX_FIT_ITERATIONS = 6


def _traced_cross_section(leg) -> tuple[np.ndarray, np.ndarray] | None:
    """(width, centre-offset) sampled along the run where BOTH of a leg's kerbs are traced.

    Both kerbs are read at the SAME centerline station, so the width there is left minus
    right and the centre is their midpoint. Neither quantity needs the alignment to be
    centred, or the street to be symmetrical, or the tracing to have started at the same
    place on the two sides - which is what makes this a measurement rather than a guess.
    Returns None unless both sides are traced: with one kerb there is no cross-section,
    only a distance to one edge.
    """
    if not {"left", "right"} <= leg.traced_sides:
        return None
    spans = [curb_station_span(leg, side) for side in ("left", "right")]
    if any(span is None for span in spans):
        return None
    lo = max(max(span[0] for span in spans), TRACED_SECTION_START_FT)
    hi = min(min(span[1] for span in spans), TRACED_SECTION_END_FT)
    if hi - lo < MIN_TRACED_SECTION_FT:
        return None
    stations = np.linspace(lo, hi, TRACED_SECTION_SAMPLES)
    left = curb_offsets_at_stations(leg, "left", stations)
    right = curb_offsets_at_stations(leg, "right", stations)
    if left is None or right is None:
        return None
    return left - right, (left + right) / 2


def _smoothed(values: np.ndarray, sample_ft: float, window_ft: float) -> np.ndarray:
    """A centred moving average over `window_ft`, with the ends held rather than tapered.

    Edge-padded, so the first and last samples average the run they can see instead of being
    dragged toward zero by absent neighbours - the ends are exactly where the correction is
    least forgiving, since the junction end sets where station 0 sits.
    """
    width = int(round(window_ft / sample_ft))
    width = min(max(width | 1, 1), len(values))     # odd, so the average is centred
    if width <= 1:
        return values
    pad = width // 2
    return np.convolve(np.pad(values, pad, mode="edge"), np.ones(width) / width, mode="valid")


def _thinned(stations: np.ndarray, gap_ft: float) -> np.ndarray:
    """Drop stations closer together than `gap_ft`, always keeping the first and the last.

    Two vertices a couple of inches apart are a hazard, not extra fidelity: the lateral
    correction at each is placed independently (place_in_measured_frame corrects per point),
    so a hundredth of a foot of difference between them turns a 0.17 ft segment into a 34
    degree turn - which is what louellen_st_west's alignment did, and a turn that sharp makes
    the offset frame meaningless for every marking measured across it.

    The alignment's own vertices are thinned along with the grid rather than held back as
    required points. Keeping every one of them reshapes the corrected line enough that the
    kerb-vertex claim window shifts underneath it and one to three surveyed vertices at each
    of two junctions stop being claimable by any leg - tracing thrown away to protect a vertex
    that carries no measurement of its own.
    """
    kept = [float(stations[0])]
    for station in stations[1:-1]:
        if station - kept[-1] >= gap_ft:
            kept.append(float(station))
    last = float(stations[-1])
    while len(kept) > 1 and last - kept[-1] < gap_ft:
        kept.pop()
    kept.append(last)
    return np.asarray(kept)


def _traced_centre_profile(leg) -> tuple[np.ndarray, np.ndarray] | None:
    """Where the kerbs' midpoint sits relative to the alignment, STATION BY STATION.

    _traced_cross_section answers the same question for the width, over the 35-130 ft window
    a width is a fact about (TRACED_SECTION_END_FT). The centre cannot borrow that window: a
    width describes the approach and is reported as one number, while the centre positions
    every offset in the proposal at every station of a leg that is DRAWN three times as far.
    Measured over the window and applied beyond it, it stops being a measurement of the road
    it is drawing - which is the whole defect this returns a profile to fix.

    Still MEASURED from TRACED_SECTION_START_FT out: nearer than that the kerbs are corner
    returns flaring to the cross street, and their midpoint is a fact about the junction mouth
    rather than the street. Outside the measured run the correction holds the nearest value it
    has, so it is flat exactly where it has nothing to measure.

    But the grid it is RETURNED on spans the whole centerline, including the original's own
    vertices, and that is not a detail. The correction is a lateral shift applied to this
    alignment, so the alignment can only survive it where there is a vertex to carry its
    shape: returning the correction on the measured run alone replaced everything inboard of
    station 35 with one straight chord, which cut the corner NJDOT rounds ~43 ft out and put
    w_broad_st_northeast's lane edge line 0.16 ft into the travel lane at station 11.
    """
    if not {"left", "right"} <= leg.traced_sides:
        return None
    spans = [curb_station_span(leg, side) for side in ("left", "right")]
    if any(span is None for span in spans):
        return None
    lo = max(max(span[0] for span in spans), TRACED_SECTION_START_FT)
    hi = min(min(span[1] for span in spans), leg.centerline.length)
    if hi - lo < MIN_TRACED_SECTION_FT:
        return None
    n = max(int(np.ceil((hi - lo) / CENTRE_SAMPLE_FT)) + 1, 2)
    measured = np.linspace(lo, hi, n)
    left = curb_offsets_at_stations(leg, "left", measured)
    right = curb_offsets_at_stations(leg, "right", measured)
    if left is None or right is None:
        return None

    # Extended to the full leg BEFORE smoothing, not after. Smoothing the measured run alone
    # and then holding its first value flat inboard leaves a corner in the correction where
    # the two meet, and a corner in the correction is a kink in the alignment: 1.74 deg at
    # station 35 on w_broad_st_northeast, which at an 11.4 ft offset throws the frame by
    # 0.17 ft and put the lane edge line inside the travel lane. Smoothed across the join,
    # the correction eases into the flat section instead.
    total = leg.centerline.length
    grid = np.append(np.arange(0.0, total, CENTRE_SAMPLE_FT), total)
    centres = _smoothed(np.interp(grid, measured, (left + right) / 2),
                        CENTRE_SAMPLE_FT, CENTRE_SMOOTH_FT)

    own, _offsets = station_offset_many(leg.centerline,
                                        np.asarray(leg.centerline.coords, dtype=float))
    stations = _thinned(np.unique(np.concatenate([grid, np.clip(own, 0.0, total)])),
                        MIN_CENTRE_VERTEX_GAP_FT)
    return stations, np.interp(stations, grid, centres)


def _resize_from_one_traced_kerb(legs: dict, name: str, legs_cfg: dict, quiet: bool) -> bool:
    """Width for a leg with only ONE kerb traced: that kerb's distance out, doubled.

    This is a guess and is labelled as one, because there is no way to make it a
    measurement - the other edge of the street was never mapped. It assumes the alignment
    runs down the middle, which the legs that DO have both kerbs traced show is wrong by
    0.2-10.3 ft. No leg at any of the four junctions needs it today; all twelve sides are
    traced. It is here for the next site, and it says so loudly in the phase output rather
    than presenting a doubled half-width as though it were a cross-section.

    Better than the nearest-approach figure it replaces only in that it uses the MEDIAN
    offset along the street rather than the single closest vertex, which lands on the
    tightest point of a corner return and biases every such leg narrow.
    """
    from src.provenance import field_measurement_governs_corner

    leg = legs[name]
    if len(leg.traced_sides) != 1 or field_measurement_governs_corner(legs_cfg.get(name, {})):
        return False
    side = next(iter(leg.traced_sides))
    span = curb_station_span(leg, side)
    if span is None:
        return False
    lo, hi = max(span[0], TRACED_SECTION_START_FT), min(span[1], TRACED_SECTION_END_FT)
    if hi - lo < MIN_TRACED_SECTION_FT:
        lo, hi = span            # short trace: all of it, corner return and all
    offsets = curb_offsets_at_stations(leg, side, np.linspace(lo, hi, TRACED_SECTION_SAMPLES))
    if offsets is None:
        return False
    width_ft = 2 * float(np.median(np.abs(offsets)))
    if not quiet:
        print(f"  NOTE: {name} is {width_ft:.1f} ft wide, but only its {side} kerb is traced - this "
              f"ASSUMES the street is symmetrical about NJDOT's alignment, which on the legs with both "
              f"kerbs traced is wrong by up to 10 ft. Trace the "
              f"{'right' if side == 'left' else 'left'} kerb to replace the assumption with a "
              f"measurement.")
    if abs(width_ft - leg.curb_to_curb_ft) < MATERIAL_WIDTH_CHANGE_FT:
        return False
    legs[name] = Leg(name=name, centerline=leg.centerline, curb_to_curb_ft=width_ft)
    return True


def _resize_and_centre_from_traced_kerbs(legs: dict, legs_cfg: dict, quiet: bool = False) -> bool:
    """Take each leg's width and its working centerline from its two traced kerbs.

    This is the measurement _widths_from_traced_kerbs could only approximate, and it
    replaces two separate approximations that were both wrong in the same direction:

      WIDTH. Doubling the nearer kerb's distance understates any leg whose alignment is off
      centre, which is all of them. Measured properly: greenwood_ave_south 25.1 -> 31.2,
      columbia_ave_west 21.8 -> 26.4, w_broad_st_northeast 30.3 -> 35.6. The last one is why
      W Broad at Louellen rendered as a road too narrow to hold the lanes drawn on it, and
      columbia_ave_west is why that leg looked like it had no room for a shoulder.

      CENTRE. An SRI line is a linear-referencing alignment, not a surveyed carriageway
      centre (see _snap_to_center), and every width in a proposal is an offset from it. Off
      centre, the paint comes out symmetrical about the wrong line and the drawing looks
      wrong even where it measures right.

    The shift here is a single constant per leg, measured over the 35-130 ft window, and it
    stays one because the fit's vertex assignment reads the frame this moves - see
    _centre_legs_on_traced_kerbs, which bends the alignment onto the kerbs' midpoint over the
    whole leg once the fit has settled and nothing is left to reassign.

    Mutates `legs` in place (replacing the Leg, so its derived curb lines are rebuilt) and
    returns whether anything moved materially; the caller re-reads the traced kerbs in the new
    frame and comes back, until the two agree.
    """
    from src.provenance import (FIELD_MEASURED, OSM_DERIVED, field_measurement_governs_corner,
                                 leg_width_provenance)

    changed = False
    for name in sorted(legs):
        leg = legs[name]
        section = _traced_cross_section(leg)
        if section is None:
            if _resize_from_one_traced_kerb(legs, name, legs_cfg, quiet):
                changed = True
            continue
        widths, centres = section
        width_ft = float(np.median(widths))
        shift_ft = float(np.median(centres))
        spread_ft = float(centres.max() - centres.min())

        cfg = legs_cfg.get(name, {})
        keep_width = field_measurement_governs_corner(cfg)
        if not quiet:
            if keep_width and abs(width_ft - leg.curb_to_curb_ft) > 1.0:
                print(f"  CONFLICT: {name} is field-measured AT THE INTERSECTION at "
                      f"{leg.curb_to_curb_ft:.1f} ft, but its two traced kerbs are {width_ft:.1f} ft "
                      f"apart. The measurement stands - it was taken at this cross-section. Check "
                      f"the tracing, or whether the measurement spanned a shoulder beyond the kerb.")
            elif not keep_width:
                tier_note = ("Its field measurement is not recorded as taken AT the intersection "
                             "(width_measured_at), so the kerbs traced at this corner govern here. "
                             if leg_width_provenance(cfg) == FIELD_MEASURED else "")
                print(f"  NOTE: {name} is {width_ft:.1f} ft curb to curb, measured between its two "
                      f"traced OSM kerbs at the same station (osm_derived; config says "
                      f"{cfg.get('curb_to_curb_ft', float('nan')):.1f}). {tier_note}Cross-sections "
                      f"range {widths.min():.1f}-{widths.max():.1f} ft.")

        moved_ft = 0.0
        if abs(shift_ft) < MATERIAL_SHIFT_FT:
            pass
        elif spread_ft > MAX_CENTRE_SPREAD_FT or abs(shift_ft) > MAX_CENTRE_SHIFT_FT:
            if not quiet:
                print(f"  NOTE: {name}'s kerb midpoint is {shift_ft:+.1f} ft off the NJDOT alignment "
                      f"and wanders {spread_ft:.1f} ft along the leg - no single shift centres that, "
                      f"so the constant is left alone and _centre_legs_on_traced_kerbs bends the "
                      f"alignment onto the midpoint instead.")
        else:
            moved_ft = shift_ft
            if not quiet:
                print(f"  NOTE: {name}'s centerline moved {shift_ft:+.1f} ft to sit midway between its "
                      f"two traced kerbs (the NJDOT alignment is a route reference, not a carriageway "
                      f"centre; midpoint holds to {spread_ft:.1f} ft along the leg).")

        if keep_width:
            width_ft = leg.curb_to_curb_ft
        if not moved_ft and abs(width_ft - leg.curb_to_curb_ft) < MATERIAL_WIDTH_CHANGE_FT:
            continue
        centerline = leg.centerline
        if moved_ft:
            (x0, y0), (x1, y1) = centerline.coords[0], centerline.coords[1]
            length = np.hypot(x1 - x0, y1 - y0)
            centerline = affinity.translate(centerline, -(y1 - y0) / length * moved_ft,
                                             (x1 - x0) / length * moved_ft)
        legs[name] = Leg(name=name, centerline=centerline, curb_to_curb_ft=width_ft,
                          width_provenance=None if keep_width else OSM_DERIVED)
        changed = True
    return changed


def _centre_legs_on_traced_kerbs(legs: dict, quiet: bool = False) -> None:
    """Bend each alignment onto its kerbs' midpoint, station by station. Runs LAST.

    Not inside _fit_legs_to_traced_kerbs, and that placement is the whole point. The fit
    decides which traced vertex belongs to which leg side by judging it against the side's
    current half-width and offset, so anything that moves the frame mid-fit changes the
    answer - and a correction that VARIES along the leg moves it differently at every
    station. Tried there, it did exactly what the fit's monotonicity guard was written to
    catch: louellen_st_west fell from 42.1 ft wide to 17.5 and w_broad_st_southwest from
    43.9 to 22.3, the same runaway that guard exists to make impossible.

    Here the widths are already settled and the traced kerbs are already assigned, so this
    can only re-express them in a better frame. THE KERBS DO NOT MOVE - they are surveyed
    world geometry, and their station/offset is recomputed against the new alignment - so
    there is no vertex to reassign and no width to re-measure. Only the derived curb of an
    UNTRACED side is rebuilt, which is right: it was never anything but an offset from this
    line.
    """
    for name in sorted(legs):
        leg = legs[name]
        profile = _traced_centre_profile(leg)
        if profile is None:
            continue
        stations, offsets = profile
        worst_ft = float(np.abs(offsets).max())
        if worst_ft < MATERIAL_SHIFT_FT:
            continue
        if worst_ft > MAX_CENTRE_SHIFT_FT:
            if not quiet:
                print(f"  NOTE: {name}'s kerb midpoint reaches {worst_ft:.1f} ft off its alignment - "
                      f"too far to be the two sides of one street, so it is left as surveyed. Check "
                      f"whether one kerb's tracing strays onto a neighbouring street.")
            continue

        centred = Leg(name=name, curb_to_curb_ft=leg.curb_to_curb_ft,
                      centerline=LineString(place_in_measured_frame(leg.centerline, stations,
                                                                     offsets)),
                      width_provenance=leg.width_provenance)
        # Traced sides keep the surveyor's line; untraced ones keep the offset __post_init__
        # just rebuilt from the corrected alignment.
        for side in leg.traced_sides:
            setattr(centred, f"{side}_curb", getattr(leg, f"{side}_curb"))
        centred.traced_sides = set(leg.traced_sides)
        legs[name] = centred
        if not quiet:
            print(f"  NOTE: {name}'s centerline bends onto its kerbs' midpoint - {offsets[0]:+.1f} ft "
                  f"at the junction to {offsets[-1]:+.1f} ft at {stations[-1]:.0f} ft out, "
                  f"{worst_ft:.1f} ft at its furthest. NJDOT's alignment is a route reference and it "
                  f"bends relative to the carriageway; every offset in a proposal is measured from "
                  f"this line, so it follows the street the whole way out.")


def _through_leg_pairs(legs: dict) -> list[tuple[str, str]]:
    """The (leg, leg) pairs that are one street running through the junction.

    Each leg is matched with whichever OTHER leg points most nearly back the way it came, and
    the pair is kept only if is_through_street agrees they are one street. Deliberately not the
    pairing through_street_sides uses: that one walks legs in bearing order and pairs each with
    its angular NEIGHBOUR, which is right for the question it asks - which corner has no kerb
    in it - and useless for this one, because at a four-way junction a leg's neighbours are the
    two cross-street legs and its opposite number is never adjacent to it. Copying it found the
    through pair at the two T-junctions and silently found nothing at Broad & Greenwood or
    Columbia & Princeton.
    """
    from src.geometry.model import leg_bearing_deg, is_through_street

    usable = {name: leg for name, leg in legs.items() if leg.centerline is not None}
    pairs = set()
    for name_a, leg_a in usable.items():
        opposed = None
        for name_b, leg_b in usable.items():
            if name_b == name_a or not is_through_street(leg_a, leg_b):
                continue
            apart = abs(180.0 - abs((leg_bearing_deg(leg_a) - leg_bearing_deg(leg_b) + 180.0)
                                     % 360.0 - 180.0))
            if opposed is None or apart < opposed[1]:
                opposed = (name_b, apart)
        if opposed is not None:
            pairs.add(tuple(sorted((name_a, opposed[0]))))
    return sorted(pairs)


def _near_direction(leg, reach_ft: float) -> np.ndarray:
    """Unit vector from a leg's junction end toward its station `reach_ft`.

    The chord over the blend, not the first segment: louellen_st_west leaves the junction on a
    15 ft stub bearing 239 deg before settling onto 269, and a tangent read off that stub
    describes nothing but the stub.
    """
    start = np.asarray(leg.centerline.coords[0], dtype=float)
    ahead = np.asarray(leg.centerline.interpolate(
        min(reach_ft, leg.centerline.length)).coords[0], dtype=float)
    return unit_vector(ahead - start)


def _join_through_legs(legs: dict, quiet: bool = False) -> None:
    """Make the two halves of one street meet at a point and a tangent, not at a joint.

    A through street is modelled as two legs, and nothing has ever required them to agree
    where they meet. At W Broad & Louellen they do not: the halves come off two different
    NJDOT routes - CR 518 turns west onto Louellen, CR 654 carries on southwest - so their
    junction ends sit 3.1 ft apart and their tangents differ by 16 deg where the traced kerbs
    say the street bends 13. Centring each half on its own carriageway (see
    _centre_legs_on_traced_kerbs) cannot fix that, because the disagreement is BETWEEN the
    halves and each is individually right.

    Drawn, it reads as a dog-leg at the node with the centreline paint kinking and the
    kerbside hatching fanning off it - which is not what is on the ground, where the paint
    runs through in one line.

    So the pair is given a shared junction point (the midpoint of the two ends) and one shared
    axis, and each half eases back onto its own measured alignment over THROUGH_JOIN_BLEND_FT.
    The REAL bend is untouched: it is still there, spread over the blend the way a striper
    would lay it rather than folded into one vertex. What goes away is the part that is an
    artefact of two route lines not meeting - up to half the 3.1 ft gap near the node, falling
    to nothing by the end of the blend.
    """
    for name_a, name_b in _through_leg_pairs(legs):
        leg_a, leg_b = legs[name_a], legs[name_b]
        blend_ft = min(THROUGH_JOIN_BLEND_FT, leg_a.centerline.length, leg_b.centerline.length)
        if blend_ft < MIN_TRACED_SECTION_FT:
            continue
        start_a = np.asarray(leg_a.centerline.coords[0], dtype=float)
        start_b = np.asarray(leg_b.centerline.coords[0], dtype=float)
        joint = (start_a + start_b) / 2
        # The shared axis bisects the two halves: leg_b points the other way, so its direction
        # is negated before averaging. Anti-parallel by construction, so the paint runs through.
        axis = unit_vector(_near_direction(leg_a, blend_ft) - _near_direction(leg_b, blend_ft))

        moved = []
        for name, leg, heading in ((name_a, leg_a, axis), (name_b, leg_b, -axis)):
            joined = _blend_onto(leg, joint, heading, blend_ft)
            if joined is None:
                continue
            legs[name] = joined
            moved.append(name)
        if moved and not quiet:
            gap_ft = float(np.hypot(*(start_a - start_b)))
            print(f"  NOTE: {name_a} and {name_b} are one street through this junction, and their "
                  f"NJDOT alignments ended {gap_ft:.1f} ft apart. Joined at a shared point and "
                  f"tangent, easing back onto each half's own measured centre over "
                  f"{blend_ft:.0f} ft - the street's real bend is kept, the joint between two "
                  f"route lines is not.")


def _blend_onto(leg, joint: np.ndarray, heading: np.ndarray, blend_ft: float):
    """`leg` re-laid to start at `joint` heading `heading`, easing back to itself by blend_ft.

    The correction is a lateral offset profile, so it can carry both requirements at once: its
    VALUE at station 0 moves the end onto the shared point and its SLOPE there turns the
    tangent onto the shared axis. A cubic Hermite with both zero at the far end returns the
    line to its own alignment with no kink to show for it.
    """
    centerline = leg.centerline
    _stations0, offsets0 = station_offset_many(centerline, np.asarray([joint], dtype=float))
    start_offset_ft = float(offsets0[0])
    here = _near_direction(leg, blend_ft)
    turn = np.arctan2(heading[1], heading[0]) - np.arctan2(here[1], here[0])
    slope = float(np.tan((turn + np.pi) % (2 * np.pi) - np.pi))
    if abs(start_offset_ft) < MATERIAL_SHIFT_FT and abs(slope * blend_ft) < MATERIAL_SHIFT_FT:
        return None

    own, _o = station_offset_many(centerline, np.asarray(centerline.coords, dtype=float))
    total = centerline.length
    stations = _thinned(np.unique(np.concatenate([
        np.arange(0.0, blend_ft, THROUGH_JOIN_SAMPLE_FT), [blend_ft],
        np.clip(own, 0.0, total), [total]])), MIN_CENTRE_VERTEX_GAP_FT)
    t = np.clip(stations / blend_ft, 0.0, 1.0)
    # Hermite basis for (value, slope) at t=0 easing to (0, 0) at t=1.
    corrections = (start_offset_ft * (2 * t ** 3 - 3 * t ** 2 + 1)
                   + slope * blend_ft * (t ** 3 - 2 * t ** 2 + t))
    joined = Leg(name=leg.name, curb_to_curb_ft=leg.curb_to_curb_ft,
                 centerline=LineString(place_in_measured_frame(centerline, stations, corrections)),
                 width_provenance=leg.width_provenance)
    for side in leg.traced_sides:
        setattr(joined, f"{side}_curb", getattr(leg, f"{side}_curb"))
    joined.traced_sides = set(leg.traced_sides)
    return joined


def _traced_side_count(legs: dict) -> int:
    """How many leg sides are currently drawn from a traced kerb rather than an offset.

    The fit's monotonicity measure: whatever else a round changes, it must never leave this
    lower than it found it. See _fit_legs_to_traced_kerbs.
    """
    return sum(len(leg.traced_sides) for leg in legs.values())


def _extend_curbs_with_far_tracing(legs: dict, center_wgs84: Point, center_ft: Point,
                                    near_coverage: dict | None = None) -> None:
    """Rebuild the curb lines once more, this time including kerb traced further out.

    KERB_NEAR_JUNCTION_FT keeps the fit's input to the ways around the junction, which is
    right for it: those are the ways a corner radius is fitted from, and at the 120 m fetch
    radius anything looser drags in neighbouring junctions. But a curb LINE wants kerb
    anywhere along a 130 ft leg, and 14 traced ways across the four junctions sit at stations
    76-127 with plausible kerb offsets and were being dropped for being >80 ft from the
    junction CENTRE - both sides of greenwood_ave_south from ~90 ft out among them. It never
    showed, because curb_line_from_points extrapolates to the working length: the outer half
    of those legs was drawn from a bearing while the tracing sat unused.

    Done AFTER the fit and with no re-measurement, which is the whole point. Feeding those
    ways to the fit itself shifts w_broad_st_southwest's measured width by half a foot, that
    reshuffles the vertex contest at the one junction with an acute Y and partial tracing, and
    louellen_st_west drops from two traced kerbs to one - more data in, less data used. With
    the widths already settled the extra ways can only lengthen a curb, never redefine one.

    Guarded anyway, on the same rule the fit uses: if the wider set somehow builds FEWER leg
    sides from tracing, the narrower result stands.
    """
    wide = kerb_lines_with_tags_ft(center_wgs84, center_ft, legs)
    before = _traced_side_count(legs)
    saved = {name: (leg.left_curb, leg.right_curb, set(leg.traced_sides))
             for name, leg in legs.items()}
    near_coverage = near_coverage or {}
    coverage = _apply_traced_curb_lines(legs, wide, center_ft, quiet=True)
    if _traced_side_count(legs) < before:
        for name, (left, right, traced) in saved.items():
            legs[name].left_curb, legs[name].right_curb = left, right
            legs[name].traced_sides = traced
        print(f"  NOTE: kerb traced beyond {KERB_NEAR_JUNCTION_FT:.0f} ft would have built "
              f"{before - _traced_side_count(legs)} fewer leg side(s) here - not used.")
        return

    # Correct the record. The fit reported each side's coverage from the NEAR set, because that
    # is all it was allowed to see, and those numbers are what a reader takes as "how much of
    # this curb is real". Left uncorrected they understate it - broad_st_east's left kerb reads
    # as traced to 76 ft when the drawing actually follows tracing to 173.7 - and a curb that
    # looks extrapolated but isn't is the same reporting failure as one that looks traced but
    # isn't, pointed the other way.
    grew = [(name, side, was, now_far)
            for (name, side), (_near, now_far) in sorted(coverage.items())
            for was in [near_coverage.get((name, side))]
            if was is not None and now_far > was[1] + 1.0]
    for name, side, was, now_far in grew:
        print(f"  NOTE: {name} {side} curb follows traced kerb out to {now_far:.0f} ft, not the "
              f"{was[1]:.0f} ft reported above - the rest is traced beyond the "
              f"{KERB_NEAR_JUNCTION_FT:.0f} ft junction radius the fit is restricted to.")


def _fit_legs_to_traced_kerbs(legs: dict, kerb_ways: list, center_ft: Point, legs_cfg: dict
                               ) -> dict[tuple[str, str], tuple[float, float]]:
    """Iterate assignment and measurement until they agree, then report the result.

    These two steps each need the other's answer. A traced vertex is assigned to the leg
    side whose half-width it best matches (assign_curb_points_to_legs), and the width is
    measured from the vertices assigned - so a bad starting width throws away the kerbs
    that would have corrected it, and keeps its own error. Both failure directions showed
    up at W Broad & Louellen: seeded too narrow, Louellen St's south kerb (155 ft of it, at
    a steady 34 ft offset) was 3.5x the assumed half-width and got discarded, leaving the
    leg "19 ft wide"; seeded from one shared width instead, W Broad's northeast leg lost
    its right kerb to the leg next door and came out at 56 ft.

    Iterating to a fixed point removes the dependence on where it starts. The first pass
    judges every leg against SEED_HALF_WIDTH_FT so no plausible kerb is excluded by a width
    nobody has measured yet; after that each leg is judged against its own current width,
    and the loop stops as soon as a round changes nothing material. It converges in 2-3
    rounds at all four junctions - and if some junction ever fails to settle, the cap ends
    it and the printed widths are still the ones actually used.

    Returns the {(leg, side): (near_ft, far_ft)} coverage it reported, so the far-tracing pass
    that runs next can correct those figures where it extends a curb past them.
    """
    started_at = {name: leg.centerline.coords[0] for name, leg in legs.items()}
    reported: dict[tuple[str, str], tuple[float, float]] = {}

    def apply_curbs(quiet=True, ratio_bounds=None):
        coverage = _apply_traced_curb_lines(legs, kerb_ways, center_ft, quiet=quiet,
                                             ratio_bounds=ratio_bounds)
        # Only the loud round, because `reported` exists to be corrected against what the
        # reader was actually shown. The quiet rounds print nothing to correct.
        if not quiet:
            reported.clear()
            reported.update(coverage)

    def snapshot():
        return {name: (leg.curb_to_curb_ft, leg.centerline) for name, leg in legs.items()}

    def restore(saved):
        for name, (width_ft, centerline) in saved.items():
            legs[name] = Leg(name=name, centerline=centerline, curb_to_curb_ft=width_ft)
        apply_curbs()       # a fresh Leg has no traced_sides until the kerbs are re-read

    apply_curbs(ratio_bounds=SEED_RATIO_BOUNDS)
    best, best_sides = snapshot(), _traced_side_count(legs)
    for _iteration in range(MAX_FIT_ITERATIONS):
        # THE FIT MAY NEVER USE LESS GROUND TRUTH THAN IT ALREADY HAD. A width feeds the
        # window that decides which traced vertices the NEXT round may claim, so a round can
        # talk itself out of a kerb it was already using - and the loss compounds. At W Broad
        # & Louellen, admitting three more (correct) kerb ways made w_broad_st_southwest
        # measure slightly differently, louellen_st_west lost its north kerb in the reshuffle,
        # its width was then guessed by doubling the south kerb's 40 ft offset into an 80 ft
        # "street", and at 80 ft its own north kerb fell below CURB_POINT_MIN_WIDTH_RATIO and
        # could never be recovered. More data in, less data used, every step defensible.
        #
        # So a round is provisional until it proves it kept every traced side. That makes the
        # fit monotone in the one quantity that matters, which is what makes the runaway
        # impossible rather than merely unlikely.
        changed = _resize_and_centre_from_traced_kerbs(legs, legs_cfg, quiet=True)
        apply_curbs()
        sides = _traced_side_count(legs)
        if sides >= best_sides:      # >=, so among equally good rounds the most converged wins
            best, best_sides = snapshot(), sides
        if not changed:
            break
    # Not "stop at the first round that does not improve" - a round may drop a side and the
    # next recover two. Run the fit out and keep the best state it visited, which is monotone
    # in the outcome without being greedy about the path.
    if _traced_side_count(legs) < best_sides:
        lost = best_sides - _traced_side_count(legs)
        restore(best)
        print(f"  NOTE: the width fit's last round built {lost} fewer leg side(s) from traced "
              f"kerb than its best round did. Kept the better geometry.")

    # Once more out loud, on the geometry that survived - the notes above describe the
    # scaffold, and a note about a width superseded two rounds later is worse than none. The
    # reporting resize replaces Legs, so the kerbs are re-read after it or every leg ends up
    # claiming no traced sides at all.
    apply_curbs(quiet=False)
    _resize_and_centre_from_traced_kerbs(legs, legs_cfg)
    apply_curbs()

    # The per-round shifts were reported quietly and are individually meaningless; what a
    # reader needs is how far the working centerline ended up from the alignment NJDOT
    # published, because every dimension in the proposal is measured off it.
    for name, leg in sorted(legs.items()):
        (x0, y0), (x1, y1) = started_at[name], leg.centerline.coords[0]
        moved_ft = float(np.hypot(x1 - x0, y1 - y0))
        if moved_ft >= MATERIAL_SHIFT_FT:
            print(f"  NOTE: {name}'s working centerline sits {moved_ft:.1f} ft off NJDOT's alignment, "
                  f"midway between its two traced kerbs. The alignment is a linear-referencing "
                  f"reference, not a surveyed carriageway centre; the paint is measured from here.")
    return dict(reported)


# Past this distance out from the junction, a traced kerb says nothing about the corner.
UNTRACED_CORNER_THRESHOLD_FT = 35.0


def _apply_traced_curb_lines(legs: dict, kerb_ways: list, center_ft: Point,
                              quiet: bool = False,
                              ratio_bounds: tuple[float, float] | None = None
                              ) -> dict[tuple[str, str], tuple[float, float]]:
    """Replace a leg's derived curb lines with the surveyor's traced kerbs.

    Returns {(leg, side): (near_ft, far_ft)} - the station span of each side that a traced
    kerb actually covers, which is what the phase output reports as "how much of this curb is
    real" and what _extend_curbs_with_far_tracing corrects where it lengthens one.

    This is the last place NJDOT's alignment was still leaking into the geometry. Position
    was fixed earlier by snapping the centerline to the OSM junction node, but the BEARING
    stayed NJDOT's - and it measured 4-8 deg off the real street at these junctions. An
    offset curb inherits that error and splays ~10 ft away from the true kerb over a 100 ft
    leg, however accurate the width is. Measured on the traced runs: greenwood_ave_north's
    offset varies 11.9 ft along 105 ft while the curb itself bends only 1.3 ft.

    So where a side is traced, the traced points ARE the curb. Untraced sides keep the
    centerline offset. Mutates `legs` in place; curb_to_curb_ft is left as the reported
    width, which no longer drives that side's geometry.

    Every traced kerb way counts, whatever its `kerb` value. A corner return is tagged
    kerb=lowered because it's a ramp - that is a statement about its height, not about
    whether it is the edge of the roadway, and filtering to kerb=raised dropped whole
    traced corners (the SW corner of Broad & Greenwood) in favour of a fitted guess.

    How far to carry a curb comes from the leg's OWN centerline, not from a site-wide working
    length: legs may be different lengths (see load_intersection_model's leg_lengths), and a
    global would draw every curb to the longest leg's end.
    """
    lines = [line for line, *_ in kerb_ways]
    if not lines:
        return {}
    coverage: dict[tuple[str, str], tuple[float, float]] = {}
    assigned = assign_curb_points_to_legs(legs, lines, ratio_bounds)
    # Which kerbs have no corner return at their junction end, and so should be extended in
    # to the node rather than stopping where the tracing happens to stop.
    straight_through = through_street_sides(legs)
    for name, sides in assigned.items():
        leg = legs[name]
        for side, points in sides.items():
            curb = curb_line_from_points(points, leg, leg.centerline.length,
                                          extend_to_junction=(name, side) in straight_through)
            if curb is None:
                continue
            setattr(leg, f"{side}_curb", curb)
            leg.traced_sides.add(side)
            near, far = min(p[0] for p in points), max(p[0] for p in points)
            coverage[(name, side)] = (near, far)
            if not quiet:
                print(f"  NOTE: {name} {side} curb is the traced OSM kerb itself, {len(points)} points "
                      f"covering {near:.0f}-{far:.0f} ft out from the junction.")

    # A side with nothing traced near the junction leaves its corner to be bridged across a
    # gap (or, with nothing traced at all, fitted from a radius). Both are weaker than a
    # traced corner, and a big enough gap is what stops the pavement ring closing - so name
    # the sides. That's the difference between "this junction isn't representable" and
    # "trace these two and it will be".
    gaps = []
    for name in legs:
        for side in ("left", "right"):
            points = assigned.get(name, {}).get(side, [])
            if not points:
                gaps.append(f"{name} {side} (nothing traced)")
            elif min(p[0] for p in points) > UNTRACED_CORNER_THRESHOLD_FT:
                gaps.append(f"{name} {side} (traced only from {min(p[0] for p in points):.0f} ft out)")
    if gaps and not quiet:
        print(f"  NOTE: no traced kerb within {UNTRACED_CORNER_THRESHOLD_FT:.0f} ft of the junction on: "
              f"{'; '.join(gaps)}. Those corners are bridged, not traced - tracing the kerb up to "
              f"the corner return would fix them.")
    return coverage
