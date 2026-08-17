"""One resolved answer to "where is this scenario's paint", shared by every consumer.

There are three consumers of that answer - the 2D plan view, the 3D export, and the scene
invariants - and the whole premise of the plan view is that it shows what the render will
show. So all three have to be looking at the SAME geometry. They were not:

  * The plan view resolved the crosswalk offsets, skews and stop bars three times per
    figure (once to draw, once via stop_bar_offsets_for, once in _mark_violations) and built
    the crosswalk bands TWICE with different arguments - with the two-pass mutual-exclusion
    reaches for the paint it drew, without them for the invariants it checked. At W Broad &
    Louellen those two bands differ by 15 sq ft, so the 2D check validated a crossing the 2D
    view did not draw and the 3D export did not build.
  * The plan view's stop bar was built without the skew stretch factor that
    stop_bar_bands_ft applies, putting the drawn bar 3.8 ft from the checked one on
    Louellen's -44 deg crossing.
  * tests/test_sites.py's scene_violations said in its docstring that it checked "exactly
    what src/render/export.py and the plan view check" while making the same
    bands-without-reaches substitution.

None of those was reachable by reading any one of the four call sites, because each looked
locally reasonable; they were only visible side by side. Resolving once and handing the
result around is what makes the claim structurally true instead of a convention four places
have to keep remembering.

Nothing here decides anything new - every value still comes from the same function in
src/render/crosswalks.py it always did. The only thing this module adds is that there is one
of each.
"""
from dataclasses import dataclass

from shapely.geometry import Polygon

from src.geometry.model import build_pavement_polygon
from src.render.crosswalks import (CROSSWALK_DEPTH_FT, crosswalk_bands_ft, crosswalk_reaches_ft,
                                   resolve_crosswalk_offsets, resolve_crosswalk_skews,
                                   resolve_stop_bar_offsets, stop_bar_bands_ft)
from src.sources.osm_context import fetch_stop_lines

# A bar governing this junction sits 33-67 ft out; this is generous. One constant, because
# the export, the plan view and the tests were each naming their own and a stop bar resolved
# at a different radius is a differently-placed stop bar.
STOP_LINE_RADIUS_M = 130


@dataclass(frozen=True)
class SceneGeometry:
    """Every marking position one DesignState implies, resolved once and shared.

    Frozen on purpose: this is the agreed description of a scenario, and a consumer that
    could adjust one field in passing is exactly how the three views drifted apart before.

    `stop_bar_offsets` and `stop_bar_bands` are empty for an unsignalized junction - a stop
    bar is only drawn where the site config declares a `signals` block, the same gate
    src/render/props.py uses for the signal hardware itself.
    """
    model: object
    state: object
    pavement: Polygon | None
    # Legs that carry a PAINTED crossing today, not merely a resolved offset. Every leg gets
    # an offset (a proposal may mark a leg that has nothing today); only a marked one is
    # something other paint has to keep clear of.
    marked_crosswalks: frozenset
    crosswalk_offsets: dict          # leg -> CrosswalkOffset(offset_ft, source)
    crosswalk_skews: dict            # leg -> degrees off square, surveyed legs only
    crosswalk_reaches: dict          # leg -> (left_ft, right_ft) out to the real kerbs
    crosswalk_bands: dict            # leg -> the painted footprint
    stop_bar_offsets: dict           # leg -> station, signalized junctions only
    stop_bar_bands: dict             # leg -> the painted footprint
    # EVERY SURVEYED CROSSING IN THE FRAME, drawn from its own traced way - including the ones at
    # junctions this site does not model, which the per-leg fields above cannot reach at all. Six of
    # the ten in Broad & Greenwood's 2.5x frame, three of them a zebra.
    #
    # Resolved HERE rather than in each renderer, which is the whole reason this class exists: the
    # 2D view, the 3D export and the coverage check all have to draw and audit the same crossings,
    # and the first version of this change computed them separately in export.py and plan_view.py.
    # That is the "two consumers assembling the same markings independently" failure this module's
    # docstring is about, and it showed immediately - the coverage check could not see what the
    # export drew, so it reported crossings as dropped that were on the page.
    surveyed_crossings: tuple = ()
    # The traced kerbs the crossings above are trimmed against, kept so a consumer that wants to
    # draw them does not fetch a second, possibly different set.
    drawn_kerbs: tuple = ()

    @classmethod
    def resolve(cls, model, state, crossings: list[dict], stop_lines: list[dict] | None = None
                 ) -> "SceneGeometry":
        """Resolve one scenario's marking geometry. `crossings` is the fetched OSM layer.

        The order below is a real dependency chain, which is the other reason this belongs in
        one place: the reaches need the pavement and the marked set, the bands need the
        reaches, and the stop bars need the crosswalk offsets.
        """
        try:
            pavement = build_pavement_polygon(state.corner_fillets)
        except ValueError:
            pavement = None     # an unclosable ring is reported by check_pavement_ring
        marked = frozenset(model.config["intersection"].get("existing_marked_crosswalks", []))
        offsets = resolve_crosswalk_offsets(state, crossings)
        skews = resolve_crosswalk_skews(state, crossings)
        # Two passes inside crosswalk_reaches_ft, so adjoining crossings at a shared corner
        # stop reaching for the same kerb. Passing these into crosswalk_bands_ft is what the
        # plan view's invariant pass used to skip.
        reaches = crosswalk_reaches_ft(state, offsets, skews, pavement, marked)
        bands = crosswalk_bands_ft(state, offsets, skews, CROSSWALK_DEPTH_FT, pavement, reaches)

        if model.config.get("signals"):
            if stop_lines is None:
                stop_lines = fetch_stop_lines(model.center_wgs84, radius_m=STOP_LINE_RADIUS_M)
            stop_bar_offsets = resolve_stop_bar_offsets(state, offsets, stop_lines)
        else:
            stop_bar_offsets = {}
        from src.geometry.intersection import drawn_kerb_radius_ft, kerb_lines_with_tags_ft
        from src.geometry.surveyed import surveyed_crossings_in_frame

        drawn_kerbs = tuple(line for line, _tags, _way_id in kerb_lines_with_tags_ft(
            model.center_wgs84, model.center_ft, radius_ft=drawn_kerb_radius_ft()))
        return cls(
            model=model, state=state, pavement=pavement, marked_crosswalks=marked,
            crosswalk_offsets=offsets, crosswalk_skews=skews, crosswalk_reaches=reaches,
            crosswalk_bands=bands, stop_bar_offsets=stop_bar_offsets,
            stop_bar_bands=stop_bar_bands_ft(state, stop_bar_offsets, skews),
            # `crossings` is the same fetched layer the per-leg offsets above came from, so the two
            # cannot disagree about which crossings exist - only about which of them belong to a leg.
            surveyed_crossings=tuple(surveyed_crossings_in_frame(model, crossings)),
            drawn_kerbs=drawn_kerbs,
        )

    @property
    def unmodelled_crossings(self) -> tuple:
        """The surveyed crossings belonging to NO modelled leg - the ones only this path can draw.

        The four a junction models are drawn from their leg's own band, including whatever a proposal
        restyles them to; drawing both would put two crossings 1.44-2.73 ft apart on one piece of
        ground. So every consumer wants this, not `surveyed_crossings`, and it is a property here
        rather than a filter each of them repeats.
        """
        return tuple(c for c in self.surveyed_crossings if c.leg is None)

    def surveyed_crossing_paint(self) -> list:
        """The bars and lines for every unmodelled surveyed crossing, trimmed to the carriageway.

        One list, so the 2D view, the 3D export and the coverage check draw and audit exactly the
        same geometry rather than three near-copies of it.
        """
        from src.geometry.surveyed import crossing_bars_ft, crossing_lines_ft

        kerbs = list(self.drawn_kerbs)
        return [piece for crossing in self.unmodelled_crossings
                for piece in (crossing_bars_ft(crossing, kerbs)
                              + crossing_lines_ft(crossing, kerbs))]

    def build_paint(self, props: list[dict] | None = None) -> list:
        """Every painted marking this scenario puts down (src/geometry/paint.py).

        Here rather than at each call site so the paint is always cut around the same bands
        the crossings are drawn from, and always told which crossings are actually marked.
        """
        from src.geometry.paint import curbside_paint_ft

        return curbside_paint_ft(self.state, self.crosswalk_offsets, self.model.center_ft,
                                  self.crosswalk_bands, props,
                                  marked_crosswalks=self.marked_crosswalks)

    def build_paint_and_posts(self, props: list[dict]) -> tuple[list, list[dict]]:
        """The paint, and `props` extended with the posts only the paint knows the place of.

        The dependency runs both ways, which is why both come back from one call: the paint
        needs the props (a hydrant or a stop sign lengthens a daylight zone), and a bike
        lane's bollards need the paint (the row starts where the crossing stops reaching, a
        station resolved in the paint builder). Returning them together is what stops one
        renderer from having posts the other does not - see props.bollard_props_from_paint.
        """
        from src.render.props import bollard_props_from_paint

        paint = self.build_paint(props)
        return paint, props + bollard_props_from_paint(self.state, paint)

    def metrics(self, paint: list):
        """What this scenario achieves, measured off this resolution (src/metrics.py).

        Here for the same reason `context` is: the outcome numbers a summary panel reports
        have to be measured from the geometry the figure drew, not recomputed from the config
        it was built out of. A crossing distance re-derived from `leg.curb_to_curb_ft` would
        agree with the drawing on a symmetric leg and quietly disagree everywhere else.
        """
        from src.metrics import SceneMetrics

        return SceneMetrics.of(self.state, reaches=self.crosswalk_reaches,
                                offsets=self.crosswalk_offsets, skews=self.crosswalk_skews,
                                paint=paint, marked=self.marked_crosswalks,
                                surveyed_leg_lengths=getattr(self.model, "surveyed_leg_lengths", None))

    def context(self, props: list[dict], paint: list):
        """This scene as the one object every invariant reads (src/checks.py:SceneContext).

        Built here because this class is already the single resolution of the geometry both
        renderers draw - so the invariants are checked against that same resolution rather than
        against whatever subset of it a call site remembered to pass.
        """
        from src.checks import SceneContext

        return SceneContext(model=self.model, state=self.state, pavement=self.pavement,
                             props=tuple(props), paint=tuple(paint),
                             crosswalk_bands=self.crosswalk_bands,
                             crosswalk_offsets=self.crosswalk_offsets,
                             stop_bars=self.stop_bar_bands)

    def check(self, props: list[dict], paint: list) -> list:
        """Every scene invariant, all violations, no raising (src/checks.py)."""
        from src.checks import check_scene

        return check_scene(self.context(props, paint))

    def report_coverage(self, props: list[dict], paint: list) -> list:
        """Print, and return, the surveyed features inside the frame that the drawing does not draw.

        A NOTE RATHER THAN A FAILURE, deliberately, and the distinction is the point of putting it
        here. Two of the four layers are genuinely still dropped - kerb ramps and traffic control are
        PROPS placed per leg, so a neighbouring junction's have nowhere to come from, the same
        structural problem crossings had. Raising on that would make every wide render fail for a
        reason no scenario can fix, and a check that cannot go green is a check people learn to
        ignore. Printed, it is in front of whoever reads the build.

        The crossings layer IS clean now, at all four sites and both frame scales, which is what
        makes the rest worth printing: this is not a permanent grumble, it is three layers of which
        one has been closed and two are named. See src/geometry/coverage.py.
        """
        from src.geometry.coverage import coverage_gaps, describe_coverage

        gaps = coverage_gaps(self.model, [*paint, *self.crosswalk_bands.values(), *props,
                                          *self.surveyed_crossing_paint()])
        if gaps:
            print(describe_coverage(gaps))
        return gaps

    def assert_valid(self, props: list[dict], paint: list, scenario: str = "") -> None:
        """Raise SceneInvariantError listing every violation, or return quietly."""
        from src.checks import assert_scene_valid

        assert_scene_valid(self.context(props, paint), scenario=scenario)
