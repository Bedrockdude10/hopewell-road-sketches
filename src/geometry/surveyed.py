"""SURVEYED CROSSINGS: every crossing the surveyor traced inside the drawn frame, drawn as traced.

The property this exists to make true (docs/network-renderer-plan.md): every feature the surveyor
recorded inside the drawn frame is either drawn from its OWN traced geometry, or named in the notes
as deliberately not drawn. A picture that shows a marked crosswalk as bare asphalt is not a
conservative simplification - it is a false statement about the street, made to an audience
deciding whether to build something.

Measured at Broad & Greenwood with HOPEWELL_FRAME_SCALE=2.5, a 431.2 ft frame radius:

    30  OSM crossings fetched at the frame's own context radius, every one of them a traced WAY
        (2-5 points, 29.0-76.8 ft long). Not one is a node.
    10  whose traced line comes inside the frame
     4  drawn today - exactly those matched to this junction's four modelled legs
     6  dropped, 263-420 ft out, three of them tagged as a zebra

The cause is not the fetch radius. A drawn crossing is rebuilt from a LEG: match the traced way to
a leg, reduce it to (station, skew), then re-derive the band from the leg's frame
(src/render/crosswalks.py:crosswalk_axes). So a junction this site does not model has no legs and
therefore no crossings at any radius, and even where it does work the survey is used as a HINT to
reconstruct geometry we already had. At 1x the frame contains exactly the 4 that are drawn, which
is why this went unnoticed for so long; at 2.5x it contains 10 and still draws 4.

So membership here is a question about the DRAWING - the same argument src/geometry/kerbs.py makes
for kerb openings, and cb9c8b6 for the kerbs themselves: a surveyed feature is already in
state-plane feet, so it needs no leg and no road, only to be in the frame. `leg` is CARRIED rather
than required, so a renderer can still prefer the treated version where a proposal restyles a
modelled leg's crossing.

ABSENT IS NOT "lines". src/render/crosswalks.py reads OSM_MARKINGS_TO_STYLE with a default -
`.get(tag, "lines")` - so a crossing with no `crossing:markings` at all comes out drawn with two
transverse lines, which is paint nobody surveyed. That mapping is reused here; its default is not.
No tag draws nothing, and `crossing:markings=no` (1 of the 30) draws nothing for the opposite
reason - the surveyor recorded that there is no paint. See is_marked.

TWO THINGS THAT WERE OPEN QUESTIONS WHEN THIS MODULE WAS FIRST WRITTEN, both now decided:

  * THE DEPRECATED `crossing=zebra` TAG IS READ. 12 of the 30 ways carry it with no
    `crossing:markings` at all, and one of those is inside Greenwood's frame 419.6 ft out. In OSM
    that tag says the crossing is marked, so reporting it as unrecorded rendered a marked crossing
    as bare asphalt - this module's opening complaint, committed by this module. The modern key
    still wins wherever it exists, including its `no`. See LEGACY_CROSSING_MARKINGS.
  * A CROSSING WAY IS TRIMMED TO THE CARRIAGEWAY, at the traced kerbs it crosses. The way runs
    sidewalk-centreline to sidewalk-centreline, 6.9-12.2 ft longer than the kerb-to-kerb reach on
    Greenwood's four legs, so bars laid along its whole length are painted over the footway.
    Trimmed against the traced KERBS rather than a leg's pavement polygon, which is what lets it
    work at the unmodelled junctions this module exists for; where those kerbs are not traced the
    way is kept whole rather than guessed at, and says so. See carriageway_geometry_ft.
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


@dataclass(frozen=True)
class SurveyedCrossing:
    """One traced OSM crossing way inside the frame, in NJ state-plane FEET.

    The traced way itself, not a station and a skew. That distinction is the whole of stream A:
    the existing per-leg path reduces this same way to (station, skew) and rebuilds a rectangle
    from the leg's frame, which needs a leg to exist and throws away the way's own vertices and
    length on the way through.

    `markings` is the `crossing:markings` tag verbatim - "zebra", "lines", "no" at these sites -
    or None where the surveyor recorded nothing. None is not "unmarked"; it is "unknown", and both
    draw nothing (see is_marked). `tags` is the whole tag dict, so a consumer can cite the survey
    rather than only the two fields this project happens to read today.

    `leg` is the modelled leg whose crossing this is, or None for the crossings that belong to no
    leg of this junction - 6 of the 10 at Greenwood. Being None is not a defect and must not be
    read as one: it is the ordinary state of a crossing at a junction nobody modelled.
    """
    geometry: LineString
    markings: str | None
    # OUT OF THE HASH, still in the equality test. A frozen dataclass hashes its fields, a dict is
    # unhashable, and a coverage check that puts these in a set (the natural way to ask "which
    # surveyed features are missing from the drawing") would hit TypeError on a type whose whole
    # job is to be compared against what was drawn. Equal tags are implied by equal geometry here,
    # so nothing that compares equal hashes differently.
    tags: dict = field(hash=False)
    leg: str | None
    distance_ft: float

    @property
    def centre(self) -> Point:
        """The middle of the traced way - what a band drawn from it is centred on.

        Measured along the ARC rather than between the endpoints, because 9 of the 10 ways at
        Greenwood have 3-5 vertices and a bent way's chord midpoint is not on the way: the two
        differ by up to 1.62 ft here, which is a third of a crossing's depth of paint placed off
        the line that was traced.
        """
        return self.geometry.interpolate(0.5, normalized=True)

    @property
    def is_marked(self) -> bool:
        """Whether this crossing has paint on it that this project knows how to draw.

        False for three different findings, and a consumer reporting coverage needs to keep them
        apart rather than count them together:

          * `crossing:markings=no` - surveyed, and surveyed as unpainted. Nothing to draw.
          * no `crossing:markings` at all - nobody recorded it. Drawing it as marked would invent
            paint; drawing it as unmarked asserts bare asphalt. It is a survey gap, and the honest
            rendering of a gap is nothing plus a note.
          * a value outside OSM_MARKINGS_TO_STYLE. None of the 30 ways here has one, so this is a
            statement about what should happen rather than a description of current output.

        FALSE IS NOT "THERE IS NO PAINT HERE" - it is "the crossing way records none". The site
        config's own `intersection.existing_marked_crosswalks` is a separate observation, and the
        two disagree: all four of Columbia & Princeton's traced crossings carry no
        `crossing:markings` tag at all while that config lists all four as marked. Field
        observation beats a missing tag, so a renderer must reconcile the two rather than read this
        property as permission to stop painting them (docs/network-renderer-plan.md, stream D).
        """
        return _style_of(self.markings) is not None

    @property
    def band_ft(self) -> Polygon:
        """The footprint the crossing occupies: the traced way, CROSSWALK_DEPTH_FT deep.

        Flat-capped, so the band ends where the surveyor's way ends instead of bulging half the
        depth past it. This is the shape to compare against what a drawing actually contains - it
        exists whether or not the crossing is marked, because the ground a crossing occupies is a
        fact about the street and the paint on it is a separate question.
        """
        return _band_along(self.geometry, 0.0, self.geometry.length)


def surveyed_crossings_in_frame(model, crossings: list[dict] | None = None
                                ) -> list[SurveyedCrossing]:
    """Every OSM crossing whose traced way reaches inside the frame `model` will be drawn at.

    THE FRAME, not the legs and not a fixed radius: src/render/frame.py:junction_frame is what
    both views point at, so "will this be in the picture" is asked of the thing that decides the
    picture. Nearest first, so a report reads outward from the junction.

    Measured from the JUNCTION CENTRE, and both the test and the reported distance use that one
    origin. The frame's own centre is the pavement bbox's centre, 12.8 ft from the junction node
    at Greenwood; testing membership against one origin while reporting the distance from the
    other would print a number beside a crossing that is not the number that decided whether it
    is in the picture. Both give the same 10 crossings here.

    Against the frame's RADIUS, which is the inscribed circle of the square the plan view's axes
    draw - so a crossing this returns is inside both views' frames, not just the 3D camera's. The
    corners of the 2D square are the deliberate remainder.

    `crossings` is the fetched OSM layer, which a renderer that already has it should pass so this
    resolves the same layer object the rest of the scene did (src/render/crosswalks.py caches its
    leg match on that list's identity).

    Fetched otherwise from the existing CROSSING_CONTEXT_RADIUS_M, taken THROUGH context_radius_m
    so the search widens with the frame - which the kerbs, roads and driveways already do and the
    crossings are the one surveyed layer that does not. It matters at exactly the frame this stream
    was measured at: src/render/export.py and src/render/plan_view.py fetch crossings at a flat
    130 m, whose bbox reaches 426.5 ft, while the 2.5x frame reaches 431.2 ft. The picture is
    already 4.7 ft wider than the data it is drawn from, and every step past 2.5x widens the gap.
    Nothing falls in that band at Greenwood today, so it is a latent miss rather than a visible
    one - the same shape as the miss this whole module is about, one layer further up.
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

    # Reused rather than re-derived. This is the public view of the same matching that decides
    # every leg's crosswalk offset, skew and kerbside hardware, so a crossing lands on the same
    # leg here as it does there - the point of that function existing. Its guards (80 ft along the
    # leg, 30 deg square-on, one crossing to one leg) are what stop a neighbouring junction's
    # crossing being credited to this one, and re-implementing them would be a second opinion
    # about the same question. `legs` is guarded because a model can be a stand-in with none, the
    # same way src/geometry/kerbs.py:kerb_openings_from_model guards its own model access.
    #
    # It is handed the TRACED records only. _matched_crossings builds a LineString from every
    # record it is given, unconditionally, so one node-form record raises GEOSException out of
    # shapely and takes the whole junction's crossings with it. `crossings` itself is passed
    # whenever nothing was dropped - which is every record at every site today - so the matcher's
    # identity-keyed cache still hits and the leg match is not recomputed per call.
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

    A CROSSING WAY IS TRACED SIDEWALK-CENTRELINE TO SIDEWALK-CENTRELINE, not kerb to kerb. Measured
    on Greenwood's four legs it overshoots the carriageway by 6.9-12.2 ft, so bars laid along the
    whole way are painted across the footway at both ends - which no striper does and no borough
    would build. Untrimmed it is a picture of paint on the pavement.

    Trimmed against the TRACED KERBS rather than against a leg's pavement polygon, which is what
    lets it work at the junctions this site does not model - the ones this whole stream exists to
    draw. The kerbs are already collected by drawing radius (cb9c8b6), so every crossing in the
    frame has them if the surveyor traced them.

    Where the way crosses no traced kerb - an unmodelled junction whose kerbs are not traced - it is
    returned WHOLE rather than guessed at. That is the honest failure: the crossing is real and
    recorded, and the only thing missing is where the kerb is, so the drawing overstates its length
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


def crossing_bars_ft(crossing: SurveyedCrossing, kerb_lines=None) -> list[Polygon]:
    """The continental bars painted along this crossing, or [] if it carries no markings.

    [] FOR AN UNMARKED CROSSING IS THE POINT, not an edge case: 2 of the 10 in Greenwood's frame
    record no markings, and painting bars on them would be this project's signature failure run
    backwards - inventing survey data instead of dropping it.

    Bars run along the direction of travel and are spaced across the crossing, which here means
    across the traced way: each bar is CONTINENTAL_BAR_WIDTH_FT of the way's own arc length,
    CROSSWALK_DEPTH_FT deep. Laid out by continental_bar_count and the same arithmetic
    scripts/blender/blender_crosswalks.py:_crosswalk_bars uses - n bars, the leftover spread
    across the GAPS so the two end bars' outer edges land exactly on the ends of the way rather
    than up to a whole period short of them.

    Along the ARC, so the bars follow a way that bends. 9 of the 10 at Greenwood have 3-5
    vertices; stepping along the chord instead would walk the bars off a bent crossing, which is
    the error crosswalk_axes had for the crossing frame and centerline_paint_ft had for the double
    yellow (3.98-7.58 ft off, both of them visible in the render).
    """
    if _style_of(crossing.markings) not in ("continental", "ladder"):
        return []
    line = carriageway_geometry_ft(crossing, kerb_lines)
    count = continental_bar_count(line.length)
    span_ft = max(line.length - CONTINENTAL_BAR_WIDTH_FT, 0.0)  # centre-to-centre, outermost pair
    pitch_ft = span_ft / (count - 1) if count > 1 else 0.0
    first_ft = (line.length - span_ft) / 2
    bars = [_band_along(line, at_ft - CONTINENTAL_BAR_WIDTH_FT / 2,
                        at_ft + CONTINENTAL_BAR_WIDTH_FT / 2)
            for at_ft in (first_ft + index * pitch_ft for index in range(count))]
    return [bar for bar in bars if not bar.is_empty]


def crossing_lines_ft(crossing: SurveyedCrossing, kerb_lines=None) -> list[LineString]:
    """The two transverse lines bounding this crossing, or [] if it carries no markings.

    The traced way's own two edges, CROSSWALK_DEPTH_FT apart - the least visible of the three
    styles, and what every marked crossing at these four junctions is tagged (6 of the 10 in
    Greenwood's frame). Offset from the way rather than built square to a leg, for the reason the
    module docstring gives: at a skewed junction the surveyed line and our idea of square
    disagree, and the line is the survey.

    Either edge is dropped if the offset degenerates - offset_curve of a way that doubles back on
    itself can return a MultiLineString or nothing at all. Guarded the way
    src/render/crosswalks.py:centerline_paint_ft guards the same call, rather than handing a
    renderer a geometry type it will fail on later.
    """
    if _style_of(crossing.markings) not in ("lines", "ladder"):
        return []
    trimmed = carriageway_geometry_ft(crossing, kerb_lines)
    edges = (trimmed.offset_curve(sign * CROSSWALK_DEPTH_FT / 2) for sign in (1, -1))
    return [edge for edge in edges if edge.geom_type == "LineString" and not edge.is_empty]


# OSM's OLDER WAY OF SAYING THE SAME THING, and this project reads it. `crossing:markings` is the
# modern key; before it, the style lived on `crossing` itself. Both are in the data here - 12 of the
# 30 fetched ways carry a legacy value and NO modern one, and one of those is inside Broad &
# Greenwood's 2.5x frame.
#
# Reading only the modern key was the first version of this module, and it is wrong for exactly the
# reason this module exists: a mapper who wrote `crossing=zebra` recorded a MARKED crossing, and
# rendering it as bare asphalt is the false claim about the street that docs/network-renderer-plan.md
# opens with. "Deprecated" describes the tag's status in OSM's schema, not the surveyor's intent.
#
# `marked` says the crossing is marked without saying how. It draws as two transverse lines, which
# is the most common marked form and the same fallback src/render/crosswalks.py already applies -
# NOT continental, because inventing a zebra where the survey only says "marked" would overstate it.
# `unmarked` is a positive statement that there is no paint, so it maps to the same "no" the modern
# key uses rather than to "nothing recorded".
LEGACY_CROSSING_MARKINGS = {"zebra": "zebra", "marked": "lines", "unmarked": "no"}


def _markings_from_tags(tags: dict) -> str | None:
    """What the surveyor recorded about this crossing's paint, or None if they recorded nothing.

    `crossing:markings` wins where it exists, including its `no` - a positive statement that a
    crossing is unmarked is a survey result and must not be overridden by an older, vaguer tag on
    the same way. Only where it is absent does the legacy `crossing` key speak.

    Distinguishing None from "no" is load-bearing downstream: "no" draws nothing AND is fully
    covered (the surveyor told us there is no paint), while None draws nothing and is a GAP - a
    crossing nobody has recorded the markings of. src/geometry/coverage.py must not report the first
    as dropped ground truth.
    """
    tags = tags or {}
    modern = tags.get("crossing:markings")
    if modern is not None:
        return modern
    return LEGACY_CROSSING_MARKINGS.get(tags.get("crossing"))


def drawable_markings(tags: dict | None) -> str | None:
    """The rendered style this crossing's survey calls for, or None if it calls for nothing.

    THE ONE DECISION about whether a surveyed crossing has paint on it that this project draws, so
    the renderer and src/geometry/coverage.py cannot disagree. Those two disagreeing is not a
    hypothetical: a crossing the renderer declines to draw and the coverage check counts as dropped
    would make the build permanently red for a reason no code change can fix, and the opposite
    pairing would hide a real omission behind a green check.

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

    Every crossing at these sites is a traced WAY - 30 of 30 - so the node form is not the main
    path, but a `highway=crossing` NODE is legal OSM and a point carries no direction, so there is
    no band to build from one and no honest way to invent its bearing. Said out loud rather than
    skipped in silence, because "nothing is silently dropped" is the property this module exists
    for and a report with one fewer crossing than OSM has is exactly what it must not produce.
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

    Associated by GEOMETRY rather than by position in the fetched list, because
    match_crossing_lines_to_legs returns the lines and tags it matched and not their indices, and
    the instruction for this stream is to reuse it unmodified. Both lines come from the same
    coords through the same transformer, so they are equal; matching on equality rather than on
    float identity keeps that true if the matcher ever rounds its own copy.
    """
    return next((name for name, (matched_line, _tags) in matched.items()
                 if matched_line.equals(line)), None)


def _band_along(line: LineString, from_ft: float, to_ft: float) -> Polygon:
    """A CROSSWALK_DEPTH_FT-deep footprint over one stretch of a traced way's arc length.

    FLAT CAPS, because the stretch's ends are real ends - the way's own endpoints for a band, a
    bar's own edges for a bar - and a round cap pushes paint half the depth (3 ft) past both of
    them. On a continental crossing that is more than untidy: the nominal gap between bars is
    1.64 ft, so 3 ft of cap at each end makes every bar overlap its neighbours, and doubled paint
    is a thing this project has already had to hunt down once (crosswalk_reaches_ft, 2.07 sq ft of
    it at a shared corner).

    Round joins, left as shapely's default, because the buffer of a piece of a line is then always
    inside the buffer of the whole line - which is the property "drawn as traced" rests on. A
    mitred join is not: at a sharp enough traced bend it spikes out to shapely's mitre limit, 5x
    the offset. No bar at these four sites bends hard enough for the two to differ measurably, so
    this is a statement about what must hold for a way we have not seen yet rather than a
    description of any current output.
    """
    piece = shapely.ops.substring(line, max(from_ft, 0.0), min(to_ft, line.length))
    if piece.is_empty or piece.geom_type != "LineString":
        return Polygon()
    return piece.buffer(CROSSWALK_DEPTH_FT / 2, cap_style="flat")
