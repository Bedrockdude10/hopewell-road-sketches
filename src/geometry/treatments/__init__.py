"""Parametric pedestrian-safety treatments: composable geometry transforms over a
DesignState. Each treatment returns a new DesignState so scenarios can be stacked
without mutating the baseline (existing-conditions) model.

SPLIT BY THE THING BEING TREATED - what a treatment CHANGES about the street - rather than by
mechanism, because that is the question anyone arrives with.

    base       the Treatment ABC, the shared value objects, and constants more than one
               family reads. Imports nothing else here.
    state      DesignState, which everything transforms.
    lanes      lane narrowing, and the posts that hold it
    bikeways   the cross-sections and the treatments that place them
    corners    radius, curb extension, apron, hatching, daylighting
    crossings  refuge islands, raised crossings, markings, the centreline
    parking    marked stalls, buffers, and the borough's restrictions
    extras     ExtraProp and the sidewalk band

THE LAYERING IS base <- state <- everything else, one-directional except in one place:
DesignState reaches for SetCenterlineStyle, AddTwoWayBikeLane and divider_shift_toward_ft, which
sit above it. Those are imported INSIDE the methods that use them.

EVERY PUBLIC NAME IS RE-EXPORTED HERE, so `from src.geometry.treatments import X` works for the
38 modules that do it. Deliberate, not a transitional shim: an import site should not have to
know which file a treatment lives in, and moving one between modules should not be a 38-file
change.
"""

from src.geometry.treatments.base import (BOLLARD_DEFAULT_SPACING_FT,
                                          CORNER_APRON_DEFAULT_EXTENT_FT,
                                          CORNER_HATCHING_DEFAULT_DEPTH_FT,
                                          CornerApron,
                                          DEFAULT_CENTERLINE_STYLE,
                                          LANE_NARROWING_DEFAULT_STRIPE_FT,
                                          LANE_WIDTH_SLACK_FT,
                                          LEGAL_PARKING_SETBACK_FT,
                                          MIN_MARKED_PARKING_DEPTH_FT,
                                          NACTO_MIN_REFUGE_ISLAND_WIDTH_FT,
                                          PARKING_STALL_DEPTH_DEFAULT_FT,
                                          PARKING_STALL_LENGTH_DEFAULT_FT,
                                          ParkingRestriction,
                                          TARGET_LANE_WIDTH_FT,
                                          Treatment,
                                          VALID_CENTERLINE_STYLES,
                                          VALID_CROSSWALK_STYLES,
                                          kerbside_allowance_ft)
from src.geometry.treatments.state import (DesignState)
from src.geometry.treatments.lanes import (LaneNarrowing, LaneNarrowingBollards)
from src.geometry.treatments.bikeways import (AASHTO_MIN_BIKE_LANE_FT, AddBikeLane,
                                              AddBikeLaneBollards,
                                              AddTwoWayBikeLane,
                                              ExtendBikeLaneThroughJunction,
                                              BIKE_LANE_BUFFER_FT,
                                              BIKE_LANE_DEFAULT_SHY_FT,
                                              BIKE_LANE_WIDTH_FT,
                                              BikeLane,
                                              BIKE_LANE_BOLLARD_SPACING_FT,
                                              CONSTRAINED_TWO_WAY_BIKE_LANE_FT,
                                              CORRIDOR_SIDE,
                                              CONTRAFLOW_DASH_FT,
                                              CONTRAFLOW_GAP_FT,
                                              MIN_BIKE_LANE_FT,
                                              MIN_TRAVEL_LANE_BESIDE_TWO_WAY_FT,
                                              MIN_TWO_WAY_BIKE_LANE_FT,
                                              NJDOT_TWO_WAY_OBJECTION,
                                              TWO_WAY_BIKE_LANE_BUFFER_FT,
                                              TWO_WAY_BIKE_LANE_WIDTH_FT,
                                              TwoWayBikeLane,
                                              bike_lane_spare_ft,
                                              divider_shift_toward_ft,
                                              far_kerb_surplus_ft,
                                              min_bike_lane_buffer_ft,
                                              travel_lane_divider_shift_ft,
                                              travel_lane_width_ft,
                                              widest_protected_lane_ft)
from src.geometry.treatments.corners import (AddCurbExtension, CURB_EXTENSION_DEVICES,
                                             CURB_EXTENSION_FACE_RADIUS_FT,
                                             CornerHatching,
                                             DAYLIGHT_DEVICES_AS_POSTS,
                                             DAYLIGHT_DEVICE_SPACING_FT,
                                             MountableApron,
                                             ProtectDaylightZone,
                                             SetCornerRadius,
                                             VALID_DAYLIGHT_DEVICES,
                                             bulb_out_corner_pair,
                                             find_corner)
from src.geometry.treatments.crossings import (CROSSING_CONTEXT_RADIUS_M, RaiseCrossing,
                                               RefugeIsland,
                                               SetCenterlineStyle,
                                               ShiftCrosswalk,
                                               UpgradeCrosswalkMarkings,
                                               all_crosswalks_continental,
                                               complete_centerlines,
                                               resolved_crossing_stations)
from src.geometry.treatments.corridor import (BROAD_ST_TWO_WAY_BIKEWAY, CorridorFacility,
                                             Section)
from src.geometry.treatments.parking import (MIN_HATCHED_ZONE_FT, MIN_USABLE_STALL_FT,
                                            lane_surplus_that_cannot_be_striped_ft,
                                            MarkedParking,
                                             narrow_lanes_and_recover_parking,
                                             osm_derived_baseline,
                                             ParkingBufferBollards,
                                             RESTRICTION_COVERAGE_SLACK_FT,
                                             RestrictionSummary,
                                             apply_osm_parking,
                                             hold_travel_lane_at_target,
                                             kerb_may_hold_parking,
                                             restriction_summary)
from src.geometry.treatments.extras import (ExtraProp, build_sidewalk_pieces)

__all__ = [
                                          "AASHTO_MIN_BIKE_LANE_FT",
                                          "BIKE_LANE_BOLLARD_SPACING_FT",
                                          "BIKE_LANE_BUFFER_FT",
                                          "BIKE_LANE_DEFAULT_SHY_FT",
                                          "BIKE_LANE_WIDTH_FT",
                                          "BOLLARD_DEFAULT_SPACING_FT",
                                          "BROAD_ST_TWO_WAY_BIKEWAY",
                                          "CONSTRAINED_TWO_WAY_BIKE_LANE_FT",
                                          "CONTRAFLOW_DASH_FT",
                                          "CONTRAFLOW_GAP_FT",
                                          "CORNER_APRON_DEFAULT_EXTENT_FT",
                                          "CORNER_HATCHING_DEFAULT_DEPTH_FT",
                                          "CORRIDOR_SIDE",
                                          "CROSSING_CONTEXT_RADIUS_M",
                                          "CURB_EXTENSION_DEVICES",
                                          "CURB_EXTENSION_FACE_RADIUS_FT",
                                          "DAYLIGHT_DEVICES_AS_POSTS",
                                          "DAYLIGHT_DEVICE_SPACING_FT",
                                          "DEFAULT_CENTERLINE_STYLE",
                                          "LANE_NARROWING_DEFAULT_STRIPE_FT",
                                          "LANE_WIDTH_SLACK_FT",
                                          "LEGAL_PARKING_SETBACK_FT",
                                          "MIN_BIKE_LANE_FT",
                                          "MIN_HATCHED_ZONE_FT",
                                          "MIN_MARKED_PARKING_DEPTH_FT",
                                          "MIN_TRAVEL_LANE_BESIDE_TWO_WAY_FT",
                                          "MIN_TWO_WAY_BIKE_LANE_FT",
                                          "MIN_USABLE_STALL_FT",
                                          "NACTO_MIN_REFUGE_ISLAND_WIDTH_FT",
                                          "NJDOT_TWO_WAY_OBJECTION",
                                          "PARKING_STALL_DEPTH_DEFAULT_FT",
                                          "PARKING_STALL_LENGTH_DEFAULT_FT",
                                          "RESTRICTION_COVERAGE_SLACK_FT",
                                          "TARGET_LANE_WIDTH_FT",
                                          "TWO_WAY_BIKE_LANE_BUFFER_FT",
                                          "TWO_WAY_BIKE_LANE_WIDTH_FT",
                                          "VALID_CENTERLINE_STYLES",
                                          "VALID_CROSSWALK_STYLES",
                                          "VALID_DAYLIGHT_DEVICES",
                                          "AddBikeLane",
                                          "AddBikeLaneBollards",
                                          "AddCurbExtension",
                                          "AddTwoWayBikeLane",
                                          "BikeLane",
                                          "CornerApron",
                                          "CornerHatching",
                                          "CorridorFacility",
                                          "DesignState",
                                          "ExtendBikeLaneThroughJunction",
                                          "ExtraProp",
                                          "LaneNarrowing",
                                          "LaneNarrowingBollards",
                                          "MarkedParking",
                                          "MountableApron",
                                          "ParkingBufferBollards",
                                          "ParkingRestriction",
                                          "ProtectDaylightZone",
                                          "RaiseCrossing",
                                          "RefugeIsland",
                                          "RestrictionSummary",
                                          "Section",
                                          "SetCenterlineStyle",
                                          "SetCornerRadius",
                                          "ShiftCrosswalk",
                                          "Treatment",
                                          "TwoWayBikeLane",
                                          "UpgradeCrosswalkMarkings",
                                          "all_crosswalks_continental",
                                          "apply_osm_parking",
                                          "bike_lane_spare_ft",
                                          "build_sidewalk_pieces",
                                          "bulb_out_corner_pair",
                                          "complete_centerlines",
                                          "divider_shift_toward_ft",
                                          "far_kerb_surplus_ft",
                                          "find_corner",
                                          "hold_travel_lane_at_target",
                                          "kerb_may_hold_parking",
                                          "kerbside_allowance_ft",
                                          "lane_surplus_that_cannot_be_striped_ft",
                                          "min_bike_lane_buffer_ft",
                                          "narrow_lanes_and_recover_parking",
                                          "osm_derived_baseline",
                                          "resolved_crossing_stations",
                                          "restriction_summary",
                                          "travel_lane_divider_shift_ft",
                                          "travel_lane_width_ft",
                                          "widest_protected_lane_ft",
]
