"""A street as ONE object, with continuous stationing - through a junction, or through a borough.

docs/network-model.md, steps 1 to 3. Nothing renders from these directly; they exist because a leg
cannot answer a corridor question. A leg starts at the junction and runs outward, so a street
through a junction is two legs pointing away from each other, each with its own station 0. A
marking measured from one cannot be continued onto the other, and a fact that straddles the
junction has no object to live on.

WHY THIS IS A PACKAGE. It was one 1,636-line module holding four jobs, and the layering between
them was invisible. Split by the QUESTION each part answers:

    road      one street through one junction, as two through legs joined head-to-head
    kerb      where the kerb is TRACED along a road, and where it is a junction corner instead
    corridor  a chain of roads, bridged along the NJDOT alignment each end's leg was cut from
    facts     what is actually there, stationed once for the whole street

The layering is road | kerb <- corridor <- facts. There is exactly ONE back edge, and it is
deliberate: `_build_corridor` needs to know where the cross streets land before it can break the
traced kerb runs at them, and cross streets are resolved FROM a corridor. That import is
function-level, with the reason written at the call site, so the module graph stays a DAG.

EVERY NAME IS RE-EXPORTED HERE, including the underscored ones. `_complement_spans`, `_merged_spans`
and `_kerb_offset_at` are imported by scripts/corridor_render.py and src/geometry/corridor_paint.py,
and `_street_name` by treatments/corridor.py - the underscore records "not a designed API", not
"not used outside this file", so removing them from this list breaks four callers.
"""
from src.geometry.network.road import (
                                    Approach,
                                    Road,
                                    SAME_POINT_FT,
                                    _joined_centerline,
                                    _joined_kerb,
                                    _kerb_offset_at,
                                    _same_point,
                                    approaches_of,
                                    road_station_of_leg_station,
                                    roads_from_model,
)
from src.geometry.network.kerb import (
                                    CORRIDOR_KERB_RADIUS_M,
                                    KERB_FROM_JUNCTION,
                                    KERB_FROM_TRACING,
                                    KERB_RUN_JOIN_FT,
                                    KERB_SAMPLE_MIN_GAP_FT,
                                    KerbRun,
                                    _complement_spans,
                                    _corridor_kerb_ways,
                                    _dense_kerb_points,
                                    _grouped_stretches,
                                    _intersect_spans,
                                    _junction_kerb_runs,
                                    _kerb_samples_on,
                                    _merged_spans,
                                    _traced_kerb_runs,
                                    _tracing_reach_ft,
                                    junction_corner_reach_ft,
)
from src.geometry.network.corridor import (
                                    CORRIDOR_EXTENSION_FT,
                                    Corridor,
                                    JunctionOnRoad,
                                    _ALIGNMENT_BBOX_MARGIN_M,
                                    _COMPASS_WORDS,
                                    _alignment_stations,
                                    _build_corridor,
                                    _chains,
                                    _corridor_name,
                                    _cumulative_ft,
                                    _ease,
                                    _eased_alignment,
                                    _extension,
                                    _junction_road_ends,
                                    _linked_ends,
                                    _oriented_chain,
                                    _oriented_piece,
                                    _road_joint_ft,
                                    _seam,
                                    _sri_alignment,
                                    _sri_spans,
                                    _street_name,
                                    corridors_from_models,
)
from src.geometry.network.facts import (
                                    CorridorFacts,
                                    _WINDOW_SAMPLE_FT,
                                    _corridor_nodes,
                                    _corridor_ways,
                                    _cross_streets_on,
                                    _driveway_meeting,
                                    _half_width_at,
                                    _marked_crossings_on,
                                    _no_parking_zones_on,
                                    _openings_on,
                                    _placed_on_corridor,
                                    _road_spans_on,
                                    corridor_facts,
                                    marked_parking_capacity,
                                    osm_window_spans,
)

__all__ = [
                                    "CORRIDOR_EXTENSION_FT",
                                    "CORRIDOR_KERB_RADIUS_M",
                                    "KERB_FROM_JUNCTION",
                                    "KERB_FROM_TRACING",
                                    "KERB_RUN_JOIN_FT",
                                    "KERB_SAMPLE_MIN_GAP_FT",
                                    "SAME_POINT_FT",
                                    "_ALIGNMENT_BBOX_MARGIN_M",
                                    "_COMPASS_WORDS",
                                    "_WINDOW_SAMPLE_FT",
                                    "Approach",
                                    "Corridor",
                                    "CorridorFacts",
                                    "JunctionOnRoad",
                                    "KerbRun",
                                    "Road",
                                    "_alignment_stations",
                                    "_build_corridor",
                                    "_chains",
                                    "_complement_spans",
                                    "_corridor_kerb_ways",
                                    "_corridor_name",
                                    "_corridor_nodes",
                                    "_corridor_ways",
                                    "_cross_streets_on",
                                    "_cumulative_ft",
                                    "_dense_kerb_points",
                                    "_driveway_meeting",
                                    "_ease",
                                    "_eased_alignment",
                                    "_extension",
                                    "_grouped_stretches",
                                    "_half_width_at",
                                    "_intersect_spans",
                                    "_joined_centerline",
                                    "_joined_kerb",
                                    "_junction_kerb_runs",
                                    "_junction_road_ends",
                                    "_kerb_offset_at",
                                    "_kerb_samples_on",
                                    "_linked_ends",
                                    "_marked_crossings_on",
                                    "_merged_spans",
                                    "_no_parking_zones_on",
                                    "_openings_on",
                                    "_oriented_chain",
                                    "_oriented_piece",
                                    "_placed_on_corridor",
                                    "_road_joint_ft",
                                    "_road_spans_on",
                                    "_same_point",
                                    "_seam",
                                    "_sri_alignment",
                                    "_sri_spans",
                                    "_street_name",
                                    "_traced_kerb_runs",
                                    "_tracing_reach_ft",
                                    "approaches_of",
                                    "corridor_facts",
                                    "corridors_from_models",
                                    "junction_corner_reach_ft",
                                    "marked_parking_capacity",
                                    "osm_window_spans",
                                    "road_station_of_leg_station",
                                    "roads_from_model",
]
