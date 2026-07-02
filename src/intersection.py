"""Assemble the full intersection model (legs, curb lines, corner fillets, parcels)
from config/intersection_config.yaml + the data sources. Shared by every phase script."""
from dataclasses import dataclass

import geopandas as gpd
from shapely.geometry import Point

from src.config import load_config
from src.data_loader import load_parcels_near, load_road_network
from src.geometry_model import (
    Leg,
    build_corner_fillets,
    buffer_point_wgs84,
    clip_to_radius,
    label_quadrants,
    nearest_per_quadrant,
    reproject_to_state_plane,
    split_leg_centerlines,
)

# (SRI, name if far endpoint is positive along axis, name if negative, axis)
LEG_AXES = [
    ("00000518__", "broad_st_east", "broad_st_west", "x"),
    ("11051089__", "greenwood_ave_north", "greenwood_ave_south", "y"),
]


@dataclass
class IntersectionModel:
    config: dict
    center_wgs84: Point
    center_ft: Point
    legs: dict[str, Leg]
    corner_fillets: dict
    parcels: gpd.GeoDataFrame
    corner_parcels: gpd.GeoDataFrame


def load_intersection_model(config: dict | None = None) -> IntersectionModel:
    config = config or load_config()
    lon, lat = config["intersection"]["center_wgs84"]
    center = Point(lon, lat)
    center_ft = gpd.GeoSeries([center], crs="EPSG:4326").to_crs("EPSG:3424").iloc[0]

    clip_radius_m = config["intersection"]["clip_radius_m"]
    bbox = buffer_point_wgs84(center, clip_radius_m * 1.3)
    network = load_road_network(bbox=bbox)
    clipped = clip_to_radius(network, center, clip_radius_m)
    clipped_ft = reproject_to_state_plane(clipped)

    working_len = config["intersection"]["leg_working_length_ft"]
    legs_cfg = config["legs"]
    legs: dict[str, Leg] = {}
    for sri, positive_name, negative_name, axis in LEG_AXES:
        rows = clipped_ft[clipped_ft["SRI"] == sri]
        if rows.empty:
            print(f"  WARNING: SRI {sri} not found in clipped network - skipping its legs.")
            continue
        line = rows.iloc[0].geometry
        for piece in split_leg_centerlines(line, center_ft, working_len):
            # NJDOT digitization sometimes has sub-foot vertex noise near
            # intersections; that noise creates enough local curvature to make
            # offset_curve() self-intersect (returns a MultiLineString) at
            # typical curb-to-curb widths. A few feet of simplification removes
            # it without affecting real road geometry at this scale.
            piece = piece.simplify(3.0)
            far = Point(piece.coords[-1])
            positive = (far.x > center_ft.x) if axis == "x" else (far.y > center_ft.y)
            name = positive_name if positive else negative_name
            cfg = legs_cfg[name]
            legs[name] = Leg(name=name, centerline=piece, curb_to_curb_ft=cfg.get("curb_to_curb_ft"))

    radius_ft = config["treatments"]["existing_corner_radius_ft"]
    corner_fillets = build_corner_fillets(legs, radius_ft) if radius_ft else {}

    parcels = load_parcels_near(center, radius_ft=300)
    corner_parcels = nearest_per_quadrant(label_quadrants(parcels, center_ft))

    return IntersectionModel(
        config=config,
        center_wgs84=center,
        center_ft=center_ft,
        legs=legs,
        corner_fillets=corner_fillets,
        parcels=parcels,
        corner_parcels=corner_parcels,
    )
