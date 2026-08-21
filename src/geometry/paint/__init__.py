"""Every piece of curbside paint a DesignState calls for, built once.

THE GEOMETRY IS DECIDED HERE AND BOTH RENDERERS DRAW WHAT THEY ARE HANDED. A second construction
of the same locus is a second chance to disagree with the first; src/checks.py inspects this
package rather than rebuilding the paint for the same reason.

WHY THIS IS A PACKAGE. It was one 1,435-line module in which the 416-line PaintContext sat between
the vocabulary it emits and the openings it clips against, so a question about any one of the three
meant reading all of it. Split by the QUESTION each part answers:

    pieces    what a painted piece IS - one marking, its kind, how wide it is painted
    anchors   where on a leg-side a treatment may paint at all, and what is too small to draw
    openings  where the kerb opens for a vehicle, and what a marking does across the gap
    context   the machinery every treatment paints through, and the one function that runs it

The layering is pieces <- anchors, openings <- context, with no cycles - the whole graph is a DAG,
which the old module had no way to show.

Returns shapely in state-plane feet; each renderer converts.

EVERY NAME IS RE-EXPORTED HERE, including the underscored ones, because `from src.geometry.paint
import X` is done in ~30 modules and an import site should not have to know which file a function
landed in.
"""
from src.geometry.paint.pieces import (
                                    LANE_EDGE_LINE_WIDTH_FT,
                                    PaintPiece,
                                    RimCause,
                                    SURFACE_PAINT_GROUP,
                                    _dot,
                                    _one,
                                    in_channel,
                                    of_kind,
                                    stroke_width_ft,
)
from src.geometry.paint.anchors import (
                                    LegAnchors,
                                    MAX_TAPER_DEPTH_PER_RUN,
                                    MIN_LINE_LENGTH_FT,
                                    MIN_PARKING_RUN_FT,
                                    MIN_RIM_LENGTH_FT,
                                    MIN_ZONE_AREA_SQ_FT,
                                    PAINT_TO_CROSSWALK_GAP_FT,
                                    RIM_SNAP_FT,
                                    _lies_wholly_behind,
                                    end_against_crossing,
                                    lane_edge_stripes,
                                    leg_anchors,
                                    parking_runs,
                                    tapers_cleanly,
                                    zone_end_line_ft,
)
from src.geometry.paint.openings import (
                                    DASH_BAND_MARGIN_FT,
                                    DASH_BAND_REACH,
                                    DASH_BAND_STEP_FT,
                                    DASH_CROSSING_SLACK_FT,
                                    DOTTED_GAP_FT,
                                    DOTTED_MARK_FT,
                                    FILLET_ARC_POINTS,
                                    KERB_HOLD_SAMPLE_FT,
                                    KerbOpenings,
                                    KerbSideOpenings,
                                    OPENING_FILLET_PER_DEPTH,
                                    OPENING_TRIM_FT,
                                    _dash_spans,
                                    _held_inside_the_kerb,
                                    _inside_the_traced_kerb,
                                    _opening_run_out,
                                    _stands_in_a_crossing,
                                    _station_band,
                                    _union,
                                    apron_polygon,
                                    junction_mouths_ft,
                                    kerb_opening_bands,
                                    stands_in_an_opening,
)
from src.geometry.paint.context import (
                                    PaintContext,
                                    curbside_paint_ft,
)

__all__ = [
                                    "DASH_BAND_MARGIN_FT",
                                    "DASH_BAND_REACH",
                                    "DASH_BAND_STEP_FT",
                                    "DASH_CROSSING_SLACK_FT",
                                    "DOTTED_GAP_FT",
                                    "DOTTED_MARK_FT",
                                    "FILLET_ARC_POINTS",
                                    "KERB_HOLD_SAMPLE_FT",
                                    "LANE_EDGE_LINE_WIDTH_FT",
                                    "MAX_TAPER_DEPTH_PER_RUN",
                                    "MIN_LINE_LENGTH_FT",
                                    "MIN_PARKING_RUN_FT",
                                    "MIN_RIM_LENGTH_FT",
                                    "MIN_ZONE_AREA_SQ_FT",
                                    "OPENING_FILLET_PER_DEPTH",
                                    "OPENING_TRIM_FT",
                                    "PAINT_TO_CROSSWALK_GAP_FT",
                                    "RIM_SNAP_FT",
                                    "SURFACE_PAINT_GROUP",
                                    "KerbOpenings",
                                    "KerbSideOpenings",
                                    "LegAnchors",
                                    "PaintContext",
                                    "PaintPiece",
                                    "RimCause",
                                    "_dash_spans",
                                    "_dot",
                                    "_held_inside_the_kerb",
                                    "_inside_the_traced_kerb",
                                    "_lies_wholly_behind",
                                    "_one",
                                    "_opening_run_out",
                                    "_stands_in_a_crossing",
                                    "_station_band",
                                    "_union",
                                    "apron_polygon",
                                    "curbside_paint_ft",
                                    "end_against_crossing",
                                    "in_channel",
                                    "junction_mouths_ft",
                                    "kerb_opening_bands",
                                    "lane_edge_stripes",
                                    "leg_anchors",
                                    "of_kind",
                                    "parking_runs",
                                    "stands_in_an_opening",
                                    "stroke_width_ft",
                                    "tapers_cleanly",
                                    "zone_end_line_ft",
]
