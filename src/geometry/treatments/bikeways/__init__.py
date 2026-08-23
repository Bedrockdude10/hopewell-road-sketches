"""BIKEWAYS: the cross-sections, the treatments that place them, and whether one fits.

WHY THIS IS A PACKAGE. It was one 1,552-line module holding four unrelated questions plus a
369-line treatment, so a question about the standards figures meant scrolling past the paint. Split
by the QUESTION each part answers:

    sections          what a bikeway IS in cross-section, and every figure that sizes one
    fit               whether a section fits this kerb, and what it leaves
    symbols           the bike symbol and the contraflow dash
    place             the treatments that place a bikeway, and all the paint one puts down
    divider           where the travel-lane divider sits once a two-way lane took one kerbside
    bollards          what stands in the buffer
    through_junction  carrying a lane across the junction

The layering is sections, symbols <- fit <- place <- divider, bollards, through_junction, with no
cycles - the whole graph is a DAG, which the old module had no way to show. `divider` and
`through_junction` are above `place` because they read the treatments a state ENDED UP with; the
sections below it can be asked anything without a state at all.

EVERY NAME IS RE-EXPORTED HERE, including the underscored ones, because an import site should not
have to know which file a function landed in.
"""
from src.geometry.treatments.bikeways.sections import (
                                            AASHTO_MIN_BIKE_LANE_FT,
                                            BIKE_LANE_BUFFER_FT,
                                            BIKE_LANE_DEFAULT_SHY_FT,
                                            BIKE_LANE_WIDTH_FT,
                                            BikeLane,
                                            CONSTRAINED_TWO_WAY_BIKE_LANE_FT,
                                            CORRIDOR_SIDE,
                                            MIN_BIKE_LANE_FT,
                                            MIN_TRAVEL_LANE_BESIDE_TWO_WAY_FT,
                                            MIN_TWO_WAY_BIKE_LANE_FT,
                                            NJDOT_TWO_WAY_OBJECTION,
                                            TWO_WAY_BIKE_LANE_BUFFER_FT,
                                            TWO_WAY_BIKE_LANE_WIDTH_FT,
                                            TwoWayBikeLane,
                                            _feet,
                                            _lane_line_ft,
                                            min_bike_lane_buffer_ft,
)
from src.geometry.treatments.bikeways.fit import (
                                            bike_lane_spare_ft,
                                            far_kerb_surplus_ft,
                                            governing_half_widths_ft,
                                            travel_way_profile,
                                            divided_lane_width_ft,
                                            travel_lane_divider_shift_ft,
                                            MIN_FACILITY_RUN_FT,
                                            section_at,
                                            widest_protected_lane_ft)
from src.geometry.treatments.bikeways.symbols import (
                                            CONTRAFLOW_DASH_FT,
                                            CONTRAFLOW_GAP_FT,
                                            SYMBOL_CLEAR_OF_OPENING_FT,
                                            SYMBOL_INTERVAL_FT,
                                            SYMBOL_LENGTH_FT,
                                            SYMBOL_WIDTH_FT,
                                            bike_symbol_polygon,
                                            bike_symbol_stations_ft,
)
from src.geometry.treatments.bikeways.place import (
                                            AddBikeLane,
                                            AddTwoWayBikeLane,
                                            THROUGH_JUNCTION_OVERLAP_FT,
)
from src.geometry.treatments.bikeways.divider import (
                                            divider_shift_toward_ft,
                                            travel_lane_edge_ft,
                                            travel_lane_width_ft,
)
from src.geometry.treatments.bikeways.bollards import (
                                            AddBikeLaneBollards,
                                            BIKE_LANE_BOLLARD_SPACING_FT,
)
from src.geometry.treatments.bikeways.through_junction import (
                                            ExtendBikeLaneThroughJunction,
                                            LANE_END_FACE_SAMPLE_FT,
                                            MIN_EXTENSION_GAP_FT,
                                            MIN_MARK_FRACTION,
                                            lane_end_face,
)

__all__ = [
                                            "AASHTO_MIN_BIKE_LANE_FT",
                                            "BIKE_LANE_BOLLARD_SPACING_FT",
                                            "BIKE_LANE_BUFFER_FT",
                                            "BIKE_LANE_DEFAULT_SHY_FT",
                                            "BIKE_LANE_WIDTH_FT",
                                            "CONSTRAINED_TWO_WAY_BIKE_LANE_FT",
                                            "CONTRAFLOW_DASH_FT",
                                            "CONTRAFLOW_GAP_FT",
                                            "CORRIDOR_SIDE",
                                            "LANE_END_FACE_SAMPLE_FT",
                                            "MIN_BIKE_LANE_FT",
                                            "MIN_EXTENSION_GAP_FT",
                                            "MIN_FACILITY_RUN_FT",
                                            "MIN_MARK_FRACTION",
                                            "MIN_TRAVEL_LANE_BESIDE_TWO_WAY_FT",
                                            "MIN_TWO_WAY_BIKE_LANE_FT",
                                            "NJDOT_TWO_WAY_OBJECTION",
                                            "SYMBOL_CLEAR_OF_OPENING_FT",
                                            "SYMBOL_INTERVAL_FT",
                                            "SYMBOL_LENGTH_FT",
                                            "SYMBOL_WIDTH_FT",
                                            "THROUGH_JUNCTION_OVERLAP_FT",
                                            "TWO_WAY_BIKE_LANE_BUFFER_FT",
                                            "TWO_WAY_BIKE_LANE_WIDTH_FT",
                                            "AddBikeLane",
                                            "AddBikeLaneBollards",
                                            "AddTwoWayBikeLane",
                                            "BikeLane",
                                            "ExtendBikeLaneThroughJunction",
                                            "TwoWayBikeLane",
                                            "_feet",
                                            "_lane_line_ft",
                                            "bike_lane_spare_ft",
                                            "bike_symbol_polygon",
                                            "bike_symbol_stations_ft",
                                            "divided_lane_width_ft",
                                            "divider_shift_toward_ft",
                                            "far_kerb_surplus_ft",
                                            "governing_half_widths_ft",
                                            "lane_end_face",
                                            "min_bike_lane_buffer_ft",
                                            "section_at",
                                            "travel_lane_divider_shift_ft",
                                            "travel_lane_edge_ft",
                                            "travel_lane_width_ft",
                                            "travel_way_profile",
                                            "widest_protected_lane_ft",
]
