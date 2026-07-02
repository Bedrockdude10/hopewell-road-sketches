"""Assemble the full intersection model (legs, curb lines, corner fillets, parcels)
from a site's config.yaml + the data sources it points to. Shared by every phase
script, for every site - nothing in this module is specific to any one
intersection (see sites/README.md for what a site provides instead)."""
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

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
from src.site import load_site_config

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class IntersectionModel:
    config: dict
    center_wgs84: Point
    center_ft: Point
    legs: dict[str, Leg]
    corner_fillets: dict
    parcels: gpd.GeoDataFrame
    corner_parcels: gpd.GeoDataFrame


def _bearing_deg(from_pt, to_pt) -> float:
    """Compass bearing (0=N, 90=E, clockwise) from from_pt to to_pt."""
    dx, dy = to_pt[0] - from_pt[0], to_pt[1] - from_pt[1]
    return (90 - np.degrees(np.arctan2(dy, dx))) % 360


def _bearing_diff(a: float, b: float) -> float:
    """Smallest angular difference between two compass bearings, in [0, 180]."""
    return abs((a - b + 180) % 360 - 180)


def _assign_leg_pieces(pieces: list, leg_names: list[str], legs_cfg: dict, center_ft: Point) -> dict[str, object]:
    """
    Match centerline pieces (all sharing one SRI, split at the intersection) to
    the configured leg names that reference that SRI, by nearest compass bearing.
    Generalizes to any number of pieces per SRI (2 for a through road, 1 for a
    dead-end/stub leg) and any intersection shape - nothing here assumes a
    4-way or perpendicular roads, only that each leg's config entry has an
    accurate `bearing_deg`.
    """
    assigned = {}
    remaining_names = list(leg_names)
    for piece in pieces:
        far_bearing = _bearing_deg((center_ft.x, center_ft.y), piece.coords[-1])
        best_name = min(remaining_names, key=lambda n: _bearing_diff(far_bearing, legs_cfg[n]["bearing_deg"]))
        assigned[best_name] = piece
        remaining_names.remove(best_name)
    return assigned


def load_intersection_model(config: dict | None = None, site: str | None = None) -> IntersectionModel:
    """Pass either a pre-loaded `config` dict, or a `site` name to load it fresh
    (defaults to src.site.DEFAULT_SITE if neither is given)."""
    if config is None:
        from src.site import DEFAULT_SITE
        config = load_site_config(site or DEFAULT_SITE)

    lon, lat = config["intersection"]["center_wgs84"]
    center = Point(lon, lat)
    center_ft = gpd.GeoSeries([center], crs="EPSG:4326").to_crs("EPSG:3424").iloc[0]

    data_sources = config.get("data_sources", {})
    road_network_path = ROOT_DIR / data_sources["road_network"]
    parcels_path = ROOT_DIR / data_sources["parcels"]

    clip_radius_m = config["intersection"]["clip_radius_m"]
    bbox = buffer_point_wgs84(center, clip_radius_m * 1.3)
    network = load_road_network(bbox=bbox, path=road_network_path)
    clipped = clip_to_radius(network, center, clip_radius_m)
    clipped_ft = reproject_to_state_plane(clipped)

    working_len = config["intersection"]["leg_working_length_ft"]
    legs_cfg = config["legs"]
    sri_to_leg_names: dict[str, list[str]] = {}
    for name, leg_cfg in legs_cfg.items():
        sri_to_leg_names.setdefault(leg_cfg["sri"], []).append(name)

    legs: dict[str, Leg] = {}
    for sri, leg_names in sri_to_leg_names.items():
        rows = clipped_ft[clipped_ft["SRI"] == sri]
        if rows.empty:
            print(f"  WARNING: SRI {sri} not found in clipped network - skipping legs {leg_names}.")
            continue
        line = rows.iloc[0].geometry
        # NJDOT digitization sometimes has sub-foot vertex noise near
        # intersections; that noise creates enough local curvature to make
        # offset_curve() self-intersect (returns a MultiLineString) at typical
        # curb-to-curb widths. A few feet of simplification removes it without
        # affecting real road geometry at this scale.
        pieces = [p.simplify(3.0) for p in split_leg_centerlines(line, center_ft, working_len)]
        for name, piece in _assign_leg_pieces(pieces, leg_names, legs_cfg, center_ft).items():
            legs[name] = Leg(name=name, centerline=piece, curb_to_curb_ft=legs_cfg[name].get("curb_to_curb_ft"))

    radius_ft = config["treatments"]["existing_corner_radius_ft"]
    corner_fillets = build_corner_fillets(legs, radius_ft) if radius_ft else {}

    parcels = load_parcels_near(center, radius_ft=300, path=parcels_path)
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
