"""Build an IntersectionModel - legs, curb lines, corner fillets, context parcels -
from a site's config.yaml + the data sources it points to. Shared by every phase.

    junction      what a junction IS once built: the model and its value types. Loads nothing.
    kerb_sources  getting the traced kerb out of OSM and into state-plane feet
    paved         driveways, aisles and lots - the minor carriageways drawn as asphalt
    osm_roads     tying legs to NJDOT's SRI centrelines and to OSM's ways
    fitting       resizing, centring and joining the legs onto the traced kerbs
    load          load_intersection_model, which is the order those steps happen in

Layering: junction <- kerb_sources <- {paved, fitting}, junction <- osm_roads, load on top.
The order inside load is load-bearing: each step measures against what the last one produced.
"""
from src.geometry.intersection.junction import (DRAWN_WIDTH_FT, DRIVEWAY_DRAWN_WIDTH_FT,
                                                IntersectionModel,
                                                OSMDataUnavailableError,
                                                PARKING_AISLE_ONEWAY_WIDTH_FT,
                                                PARKING_AISLE_WIDTH_FT,
                                                PARKING_RESTRICTION_KEYS,
                                                PavedKind,
                                                PavedSurface,
                                                ROOT_DIR,
                                                RoadSpan,
                                                parking_is_restricted,
                                                parking_restriction_by_side)
from src.geometry.intersection.kerb_sources import (KERB_ALONG_LEG_TOLERANCE_FT,
                                                    KERB_CONTEXT_RADIUS_M,
                                                    KERB_NEAR_JUNCTION_FT,
                                                    drawn_kerb_radius_ft,
                                                    kerb_lines_with_tags_ft)
from src.geometry.intersection.paved import (DRIVEWAY_CONTEXT_RADIUS_M, to_state_plane)
from src.geometry.intersection.osm_roads import (MIN_ROAD_SPAN_FT, ROAD_CONTEXT_RADIUS_M,
                                                 ROAD_MATCH_HIGHWAY_CLASSES,
                                                 ROAD_MATCH_MAX_ANGLE_DEG,
                                                 ROAD_MATCH_MAX_OFFSET_FT,
                                                 SNAP_REPORT_THRESHOLD_FT)
from src.geometry.intersection.fitting import (CENTRE_SAMPLE_FT, CENTRE_SMOOTH_FT,
                                               KERB_PLAUSIBLE_HALF_WIDTH_FT,
                                               MATERIAL_SHIFT_FT,
                                               MATERIAL_WIDTH_CHANGE_FT,
                                               MAX_CENTRE_SHIFT_FT,
                                               MAX_CENTRE_SPREAD_FT,
                                               MAX_FIT_ITERATIONS,
                                               MIN_CENTRE_VERTEX_GAP_FT,
                                               MIN_TRACED_SECTION_FT,
                                               SEED_RATIO_BOUNDS,
                                               THROUGH_JOIN_BLEND_FT,
                                               THROUGH_JOIN_SAMPLE_FT,
                                               TRACED_SECTION_END_FT,
                                               TRACED_SECTION_SAMPLES,
                                               TRACED_SECTION_START_FT,
                                               UNTRACED_CORNER_THRESHOLD_FT)
from src.geometry.intersection.load import (load_intersection_model)

__all__ = [
                                                "CENTRE_SAMPLE_FT",
                                                "CENTRE_SMOOTH_FT",
                                                "DRAWN_WIDTH_FT",
                                                "DRIVEWAY_CONTEXT_RADIUS_M",
                                                "DRIVEWAY_DRAWN_WIDTH_FT",
                                                "KERB_ALONG_LEG_TOLERANCE_FT",
                                                "KERB_CONTEXT_RADIUS_M",
                                                "KERB_NEAR_JUNCTION_FT",
                                                "KERB_PLAUSIBLE_HALF_WIDTH_FT",
                                                "MATERIAL_SHIFT_FT",
                                                "MATERIAL_WIDTH_CHANGE_FT",
                                                "MAX_CENTRE_SHIFT_FT",
                                                "MAX_CENTRE_SPREAD_FT",
                                                "MAX_FIT_ITERATIONS",
                                                "MIN_CENTRE_VERTEX_GAP_FT",
                                                "MIN_ROAD_SPAN_FT",
                                                "MIN_TRACED_SECTION_FT",
                                                "PARKING_AISLE_ONEWAY_WIDTH_FT",
                                                "PARKING_AISLE_WIDTH_FT",
                                                "PARKING_RESTRICTION_KEYS",
                                                "ROAD_CONTEXT_RADIUS_M",
                                                "ROAD_MATCH_HIGHWAY_CLASSES",
                                                "ROAD_MATCH_MAX_ANGLE_DEG",
                                                "ROAD_MATCH_MAX_OFFSET_FT",
                                                "ROOT_DIR",
                                                "SEED_RATIO_BOUNDS",
                                                "SNAP_REPORT_THRESHOLD_FT",
                                                "THROUGH_JOIN_BLEND_FT",
                                                "THROUGH_JOIN_SAMPLE_FT",
                                                "TRACED_SECTION_END_FT",
                                                "TRACED_SECTION_SAMPLES",
                                                "TRACED_SECTION_START_FT",
                                                "UNTRACED_CORNER_THRESHOLD_FT",
                                                "IntersectionModel",
                                                "OSMDataUnavailableError",
                                                "PavedKind",
                                                "PavedSurface",
                                                "RoadSpan",
                                                "drawn_kerb_radius_ft",
                                                "kerb_lines_with_tags_ft",
                                                "load_intersection_model",
                                                "parking_is_restricted",
                                                "parking_restriction_by_side",
                                                "to_state_plane",
]
