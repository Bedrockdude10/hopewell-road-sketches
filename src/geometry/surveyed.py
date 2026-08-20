"""SURVEYED CROSSINGS: every crossing the surveyor traced inside the drawn frame, drawn as traced.

The invariant (docs/network-renderer-plan.md): every feature the surveyor recorded inside the drawn
frame is either drawn from its OWN traced geometry, or named in the notes as deliberately not
drawn. Drawing a marked crosswalk as bare asphalt is a false statement about the street, not a
conservative simplification.

Geometry is NJ state-plane FEET throughout.

Not the per-leg path. src/render/crosswalks.py:crosswalk_axes reduces a traced way to
(station, skew) and rebuilds the band from the leg's frame, so it draws nothing at a junction this
site does not model and discards the way's own vertices where it does. Membership here is instead a
question about the DRAWING - the same argument src/geometry/kerbs.py makes for kerb openings: a
surveyed feature is already in state-plane feet, so it needs no leg and no road, only to be in the
frame. `leg` is CARRIED rather than required, so a renderer can still prefer the treated version
where a proposal restyles a modelled leg's crossing.

ABSENT IS NOT "lines". src/render/crosswalks.py reads OSM_MARKINGS_TO_STYLE with a default -
`.get(tag, "lines")` - so an untagged crossing comes out drawn with paint nobody surveyed. That
mapping is reused here; its default is not. No tag draws nothing, and `crossing:markings=no` draws
nothing for the opposite reason - the surveyor recorded that there is no paint. See is_marked.

The deprecated `crossing=zebra` tag is read too (LEGACY_CROSSING_MARKINGS); in OSM it says the
crossing is marked, so ignoring it renders a marked crossing as bare asphalt. A crossing way is
trimmed to the carriageway at the traced kerbs it crosses (carriageway_geometry_ft).
"""
from dataclasses import dataclass, field

import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import substring

from src.render.coords import wgs84_to_state_plane
from src.render.crosswalks import (CONTINENTAL_BAR_WIDTH_FT, CROSSWALK_DEPTH_FT,
                                   OSM_MARKINGS_TO_STYLE, continental_bar_count,
                                   match_crossing_lines_to_legs)
from src.render.frame import junction_frame
from typing import TYPE_CHECKING

if TYPE_CHECKING:    # annotation-only: these types are layered above this module,
    # so importing them for real would close a cycle.
    from src.geometry.intersection.junction import IntersectionModel
    from src.geometry.treatments.state import DesignState


@dataclass(frozen=True)
class SurveyedCrossing:
    """One traced OSM crossing way inside the frame, in NJ state-plane FEET.

    `markings` is the `crossing:markings` tag verbatim - "zebra", "lines", "no" at these sites -
    or None where the surveyor recorded nothing. None is not "unmarked"; it is "unknown", and both
    draw nothing (see is_marked). `tags` is the whole tag dict, so a consumer can cite the survey
    rather than only the two fields this project happens to read today.

    `leg` is the modelled leg whose crossing this is, or None. None is not a defect: it is the
    ordinary state of a crossing at a junction nobody modelled.
    """
    geometry: LineString
    markings: str | None
    # OUT OF THE HASH, still in the equality test: a dict is unhashable, and these must go in a set
    # to ask "which surveyed features are missing from the drawing". Equal tags are implied by equal
    # geometry here, so nothing that compares equal hashes differently.
    tags: dict = field(hash=False)
    leg: str | None
    distance_ft: float

    @property
    def centre(self) -> Point:
        """The middle of the traced way - what a band drawn from it is centred on.

        Measured along the ARC, not between the endpoints: these ways have 3-5 vertices and a bent
        way's chord midpoint is not on the way (up to 1.62 ft off here).
        """
        return self.geometry.interpolate(0.5, normalized=True)

    @property
    def is_marked(self) -> bool:
        """Whether this crossing has paint on it that this project knows how to draw.

        False for three findings a coverage report must keep apart: `crossing:markings=no`
        (surveyed as unpainted), no `crossing:markings` at all (a survey gap - drawing it either
        way asserts something nobody recorded), and a value outside OSM_MARKINGS_TO_STYLE.

        FALSE IS NOT "THERE IS NO PAINT HERE" - it is "the crossing way records none". The site
        config's own `intersection.existing_marked_crosswalks` is a SEPARATE observation and the
        two disagree (Columbia & Princeton: no tags, config lists all four as marked). Field
        observation beats a missing tag, so a renderer must reconcile the two rather than read this
        property as permission to stop painting them (docs/network-renderer-plan.md, stream D).
        """
        return _style_of(self.markings) is not None

    @property
    def band_ft(self) -> Polygon:
        """The footprint the crossing occupies: the traced way, CROSSWALK_DEPTH_FT deep.

        Flat-capped, so the band ends where the surveyor's way ends instead of bulging half the
        depth past it. Exists whether or not the crossing is marked: the ground a crossing occupies
        is a fact about the street, and the paint on it is a separate question.
        """
        return _band_along(self.geometry, 0.0, self.geometry.length)


def surveyed_crossings_in_frame(model: "IntersectionModel", crossings: list[dict] | None = None
                                ) -> list[SurveyedCrossing]:
    """Every OSM crossing whose traced way reaches inside the frame `model` will be drawn at.

    THE FRAME, not the legs and not a fixed radius: src/render/frame.py:junction_frame is what
    both views point at. Nearest first, so a report reads outward from the junction.

    DATUM: distance is measured from the JUNCTION CENTRE for both the membership test and the
    reported number. The frame's own centre is the pavement bbox's centre, 12.8 ft away at
    Greenwood; using one origin to test and the other to report would print a number beside a
    crossing that is not the number that decided whether it is in the picture.

    Against the frame's RADIUS - the inscribed circle of the square the plan view's axes draw - so
    a crossing this returns is inside both views' frames. The corners of the 2D square are the
    deliberate remainder.

    `crossings` is the fetched OSM layer, which a renderer that already has it should pass so this
    resolves the same layer object the rest of the scene did (src/render/crosswalks.py caches its
    leg match on that list's identity).

    Fetched otherwise from CROSSING_CONTEXT_RADIUS_M taken THROUGH context_radius_m, so the search
    widens with the frame as the kerbs, roads and driveways already do. src/render/export.py and
    src/render/plan_view.py still fetch crossings at a flat 130 m (426.5 ft), narrower than the
    2.5x frame's 431.2 ft reach, and every step past 2.5x widens that gap.
    """
    frame = junction_frame(model)
    if crossings is None:
        # Local, following src/geometry/treatments/crossings.py: a caller that already has the
        # layer should not pay for importing the OSM stack to be handed back its own list.
        from src.geometry.treatments import CROSSING_CONTEXT_RADIUS_M
        from src.render.frame import context_radius_m
        from src.sources.osm_context import fetch_crossings

        crossings = fetch_crossings(model.center_wgs84,
                                    radius_m=context_radius_m(CROSSING_CONTEXT_RADIUS_M))
    traced = []
    for record in crossings:
        line = _traced_line_ft(record)
        if line is not None:
            traced.append((record, line))

    # Reused rather than re-derived, so a crossing lands on the same leg here as it does in the
    # per-leg path. Its guards (80 ft along the leg, 30 deg square-on, one crossing to one leg) are
    # what stop a neighbouring junction's crossing being credited to this one; re-implementing them
    # would be a second opinion about the same question. `legs` is guarded because a model can be a
    # stand-in with none.
    #
    # It is handed the TRACED records only: _matched_crossings builds a LineString from every record
    # unconditionally, so one node-form record raises GEOSException and takes the whole junction's
    # crossings with it. `crossings` itself is passed whenever nothing was dropped, so the matcher's
    # identity-keyed cache still hits.
    legs = getattr(model, "legs", None) or {}
    matchable = crossings if len(traced) == len(crossings) else [record for record, _ in traced]
    matched = match_crossing_lines_to_legs(legs, matchable) if legs else {}

    found = []
    for record, line in traced:
        distance_ft = model.center_ft.distance(line)
        if distance_ft > frame.radius_ft:
            continue
        tags = record.get("tags") or {}
        found.append(SurveyedCrossing(geometry=line, markings=_markings_from_tags(tags), tags=tags,
                                      leg=_leg_of(line, matched), distance_ft=distance_ft))
    found.sort(key=lambda crossing: crossing.distance_ft)
    return found


def carriageway_geometry_ft(crossing: SurveyedCrossing, kerb_lines=None) -> LineString:
    """The part of the traced way that is actually ROAD, trimmed at the kerbs it crosses.

    DATUM: a crossing way is traced SIDEWALK-CENTRELINE TO SIDEWALK-CENTRELINE, not kerb to kerb -
    6.9-12.2 ft longer than the carriageway on Greenwood's four legs. Bars laid along the whole way
    are painted across the footway at both ends.

    Trimmed against the TRACED KERBS rather than a leg's pavement polygon, which is what lets it
    work at the junctions this site does not model.

    Where the way crosses no traced kerb it is returned WHOLE rather than guessed at: the crossing
    is real and recorded and only the kerb position is missing, so the drawing overstates its length
    rather than dropping it. `trimmed_to_carriageway` says which happened so a caller can report it.
    """
    if kerb_lines is None:
        return crossing.geometry
    line = crossing.geometry
    hits = []
    for kerb in kerb_lines:
        crossed = line.intersection(kerb)
        if crossed.is_empty:
            continue
        for part in getattr(crossed, "geoms", [crossed]):
            hits.append(line.project(Point(part.coords[0]) if part.geom_type == "LineString"
                                      else part))
    # Two crossings of the kerb line are the two ends of the carriageway. One is a way that stops in
    # the road - keep it whole rather than trimming to a point.
    if len(hits) < 2:
        return line
    return substring(line, min(hits), max(hits)) or line


def crossing_style_in(state: "DesignState", crossing: SurveyedCrossing) -> str | None:
    """How `state` calls for this crossing to be drawn, or None to draw nothing.

    THE ONE PLACE a design's marking policy meets a crossing that has no leg; the per-leg path asks
    src/render/crosswalks.py:resolve_crosswalk_style instead.

    RESTYLES, NEVER INVENTS. `is_marked` is the gate and cannot be relaxed: a crossing recorded as
    unpainted (`crossing:markings=no`), or recorded nothing about, draws nothing however loudly a
    policy says "all crosswalks continental". Painting one there would be a NEW crossing at an
    uncontrolled approach, which MUTCD 3C.02(04) wants an engineering study for and this project has
    no pedestrian counts to do (STANDARDS.md section 2).

    So the policy can only move a crossing UP the visibility ranking that already applies to it:
    lines -> continental is a repaint of something that exists. Absent a policy the surveyed style
    stands, which is what existing conditions must always show.
    """
    if not crossing.is_marked:
        return None
    from src.geometry.targets import Everywhere
    from src.geometry.treatments import UpgradeCrosswalkMarkings

    treatment = (state.treatment_for(UpgradeCrosswalkMarkings, Everywhere())
                 if state is not None else None)
    return treatment.style if treatment is not None else _style_of(crossing.markings)


def crossing_bars_ft(crossing: SurveyedCrossing, kerb_lines=None, style: str | None = None
                      ) -> list[Polygon]:
    """The continental bars painted along this crossing, or [] if it carries no markings.

    [] FOR AN UNMARKED CROSSING IS THE POINT, not an edge case: painting bars there would invent
    survey data.

    Each bar is CONTINENTAL_BAR_WIDTH_FT of the way's own arc length, CROSSWALK_DEPTH_FT deep. Laid
    out by continental_bar_count and the same arithmetic
    scripts/blender/blender_crosswalks.py:_crosswalk_bars uses - n bars, the leftover spread across
    the GAPS so the two end bars' outer edges land exactly on the ends of the way.

    Along the ARC, so the bars follow a way that bends; stepping along the chord walks them off a
    bent crossing by 3.98-7.58 ft (the error crosswalk_axes and centerline_paint_ft each had).

    `style` overrides the surveyed one where a design has a marking policy - see crossing_style_in,
    the only thing allowed to compute it because it is the only thing that also enforces "restyle,
    never invent".
    """
    if (style or _style_of(crossing.markings)) not in ("continental", "ladder"):
        return []
    line = carriageway_geometry_ft(crossing, kerb_lines)
    count = continental_bar_count(line.length)
    span_ft = max(line.length - CONTINENTAL_BAR_WIDTH_FT, 0.0)  # centre-to-centre, outermost pair
    pitch_ft = span_ft / (count - 1) if count > 1 else 0.0
    first_ft = (line.length - span_ft) / 2
    bars = [_band_along(line, at_ft - CONTINENTAL_BAR_WIDTH_FT / 2,
                        at_ft + CONTINENTAL_BAR_WIDTH_FT / 2)
            for at_ft in (first_ft + index * pitch_ft for index in range(count))]
    return _kept_apart(bar for bar in bars if not bar.is_empty)


def _kept_apart(bars) -> list[Polygon]:
    """The bars with any overlap taken off the later one, so no ground is painted twice.

    THE PITCH IS ALONG THE ARC AND A BENT WAY TURNS BETWEEN TWO BARS, so on the inside of the turn
    their ends converge and the flat caps cross even though the gap measured along the way is a
    healthy 1.64 ft. Greenwood's third zebra bends 4 times in 38.2 ft and its first pair overlapped
    by 0.05 sq ft. Small, and still paint laid over paint - the same defect as the round-cap bug
    _band_along documents, arriving from the other direction.

    Clipped rather than re-pitched: widening the pitch at a bend would move every bar off the arc
    length the count was built from, and the end bars off the ends of the way, to fix a sliver.
    A bar reduced to nothing is dropped - it was entirely inside its neighbour.
    """
    kept: list[Polygon] = []
    for bar in bars:
        for earlier in kept:
            if bar.intersects(earlier):
                bar = bar.difference(earlier)
        # A difference can split a bar in two round a bend; the larger piece is the bar.
        if bar.geom_type == "MultiPolygon":
            bar = max(bar.geoms, key=lambda piece: piece.area)
        if not bar.is_empty and bar.area > 1e-6:
            kept.append(bar)
    return kept


def crossing_lines_ft(crossing: SurveyedCrossing, kerb_lines=None, style: str | None = None
                       ) -> list[LineString]:
    """The two transverse lines bounding this crossing, or [] if it carries no markings.

    The traced way's own two edges, CROSSWALK_DEPTH_FT apart. Offset from the way rather than built
    square to a leg: at a skewed junction the surveyed line and our idea of square disagree, and the
    line is the survey.

    Either edge is dropped if the offset degenerates - offset_curve of a way that doubles back on
    itself can return a MultiLineString or nothing at all - rather than handing a renderer a
    geometry type it will fail on later.

    `style` is crossing_style_in's - see crossing_bars_ft.
    """
    if (style or _style_of(crossing.markings)) not in ("lines", "ladder"):
        return []
    trimmed = carriageway_geometry_ft(crossing, kerb_lines)
    edges = (trimmed.offset_curve(sign * CROSSWALK_DEPTH_FT / 2) for sign in (1, -1))
    return [edge for edge in edges if edge.geom_type == "LineString" and not edge.is_empty]


# OSM's older way of saying the same thing, and this project reads it: before `crossing:markings`
# the style lived on `crossing` itself, and a substantial share of the ways here carry only the
# legacy value. "Deprecated" describes the tag's status in OSM's schema, not the surveyor's intent.
#
# `marked` says the crossing is marked without saying how, so it draws as two transverse lines - NOT
# continental, because inventing a zebra where the survey only says "marked" would overstate it.
# `unmarked` is a positive statement that there is no paint, so it maps to the modern key's "no"
# rather than to "nothing recorded".
LEGACY_CROSSING_MARKINGS = {"zebra": "zebra", "marked": "lines", "unmarked": "no"}


def _markings_from_tags(tags: dict) -> str | None:
    """What the surveyor recorded about this crossing's paint, or None if they recorded nothing.

    `crossing:markings` wins where it exists, including its `no` - a positive statement that a
    crossing is unmarked is a survey result and must not be overridden by an older, vaguer tag on
    the same way. Only where it is absent does the legacy `crossing` key speak.

    Distinguishing None from "no" is load-bearing downstream: "no" draws nothing AND is fully
    covered, while None draws nothing and is a GAP. src/geometry/coverage.py must not report the
    first as dropped ground truth.
    """
    tags = tags or {}
    modern = tags.get("crossing:markings")
    if modern is not None:
        return modern
    return LEGACY_CROSSING_MARKINGS.get(tags.get("crossing"))


def drawable_markings(tags: dict | None) -> str | None:
    """The rendered style this crossing's survey calls for, or None if it calls for nothing.

    THE ONE DECISION about whether a surveyed crossing has paint this project draws, so the renderer
    and src/geometry/coverage.py cannot disagree - a disagreement either makes the build permanently
    red for a reason no code change can fix, or hides a real omission behind a green check.

    Reads the tags, so a caller that has not built a SurveyedCrossing yet can still ask.
    """
    return _style_of(_markings_from_tags(tags))


def _style_of(markings: str | None) -> str | None:
    """Which of the three rendered styles a surveyed marking value draws as, or None.

    OSM_MARKINGS_TO_STYLE without its default. That is the only difference between this and what
    src/render/crosswalks.py:_match_crossings_to_legs does with the same tag, and it is the
    difference between "no tag means we do not know" and "no tag means two transverse lines".
    """
    return OSM_MARKINGS_TO_STYLE.get(markings)


def _traced_line_ft(record: dict) -> LineString | None:
    """One fetched crossing record as a state-plane line, or None if it is not a line at all.

    A `highway=crossing` NODE is legal OSM, and a point carries no direction, so there is no band to
    build from one and no honest way to invent its bearing. Said out loud rather than skipped in
    silence, because "nothing is silently dropped" is the property this module exists for.
    """
    coords = record.get("coords_wgs84") or []
    if len(coords) < 2:
        print(f"  NOTE: OSM crossing (nodes {record.get('node_ids') or '?'}) is mapped as a single "
              f"point, not a traced way. A point has no direction, so there is no band to draw "
              f"from it - it is not in this frame's crossings.")
        return None
    xs, ys = wgs84_to_state_plane.transform([c[0] for c in coords], [c[1] for c in coords])
    return LineString(zip(xs, ys))


def _leg_of(line: LineString, matched: dict) -> str | None:
    """The modelled leg this traced way is the crossing of, or None.

    Associated by GEOMETRY, not by position in the fetched list: match_crossing_lines_to_legs
    returns the lines and tags it matched, not their indices. Matched on equality rather than float
    identity, so it stays true if the matcher ever rounds its own copy.
    """
    return next((name for name, (matched_line, _tags) in matched.items()
                 if matched_line.equals(line)), None)


def _band_along(line: LineString, from_ft: float, to_ft: float) -> Polygon:
    """A CROSSWALK_DEPTH_FT-deep footprint over one stretch of a traced way's arc length.

    FLAT CAPS, because the stretch's ends are real ends and a round cap pushes paint half the depth
    (3 ft) past both. The nominal gap between continental bars is 1.64 ft, so round caps would make
    every bar overlap its neighbours.

    Round joins, left as shapely's default, because the buffer of a piece of a line is then always
    inside the buffer of the whole line - the property "drawn as traced" rests on. A mitred join is
    not: at a sharp enough traced bend it spikes out to shapely's mitre limit, 5x the offset.
    """
    piece = shapely.ops.substring(line, max(from_ft, 0.0), min(to_ft, line.length))
    if piece.is_empty or piece.geom_type != "LineString":
        return Polygon()
    return piece.buffer(CROSSWALK_DEPTH_FT / 2, cap_style="flat")
