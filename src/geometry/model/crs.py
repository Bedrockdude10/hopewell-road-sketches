"""Projections, and the operations that only make sense in one.

WGS84 in, NJ State Plane feet out, plus the buffering and radius clipping that has to happen in a
metric CRS on the way. Every distance in the rest of this package is FEET in EPSG:3424; this is
the only module that knows another datum exists, and reading a bbox in the wrong one returns zero
rows rather than an error - which is why the two constants live here and are imported, never
retyped."""
from functools import lru_cache

import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.ops import substring


WGS84 = "EPSG:4326"
NJ_STATE_PLANE_FT = "EPSG:3424"  # NAD83(HARN) / New Jersey (ftUS)


@lru_cache(maxsize=64)
def _utm_crs_at(lon: float, lat: float):
    """The local UTM CRS for a WGS84 point.

    Cached because geopandas' estimate_utm_crs() queries the PROJ database for every candidate
    CRS at ~38 ms a call and is the single most expensive call in this pipeline (43 of them,
    1.67 s of a 2.76 s scenario export, all about one intersection). It is a pure function of
    the point, called with a handful of distinct arguments.

    Keyed on (lon, lat) rather than the Point, because a Point is unhashable and two Points at
    the same place are different objects.
    """
    return gpd.GeoSeries([Point(lon, lat)], crs=WGS84).estimate_utm_crs()


@lru_cache(maxsize=256)
def _buffer_bounds_wgs84(lon: float, lat: float, radius_m: float) -> tuple[float, float, float, float]:
    utm_crs = _utm_crs_at(lon, lat)
    point_gs = gpd.GeoSeries([Point(lon, lat)], crs=WGS84)
    buffered = point_gs.to_crs(utm_crs).buffer(radius_m).to_crs(WGS84)
    return tuple(buffered.total_bounds)


def buffer_point_wgs84(point: Point, radius_m: float) -> tuple[float, float, float, float]:
    """Buffer a WGS84 point by radius_m meters (via a local UTM projection) and
    return a WGS84 bbox as (minx, miny, maxx, maxy).

    Memoized on (lon, lat, radius): src/sources/osm_context.py calls this twice per OSM fetch
    (once to bound the layer, once in assert_within_snapshot) and there are eight fetchers
    called repeatedly per site, all about the same centre. See _utm_crs_at.
    """
    return _buffer_bounds_wgs84(point.x, point.y, float(radius_m))


def clip_to_radius(gdf: gpd.GeoDataFrame, center: Point, radius_m: float) -> gpd.GeoDataFrame:
    """Clip a WGS84 GeoDataFrame to a circular radius (meters) around center,
    trimming feature geometry (not just filtering by bbox)."""
    center_gs = gpd.GeoSeries([center], crs=WGS84)
    utm_crs = _utm_crs_at(center.x, center.y)
    center_utm = center_gs.to_crs(utm_crs).iloc[0]
    circle_wgs84 = gpd.GeoSeries([center_utm.buffer(radius_m)], crs=utm_crs).to_crs(WGS84).iloc[0]

    clipped = gdf[gdf.intersects(circle_wgs84)].copy()
    clipped["geometry"] = clipped.intersection(circle_wgs84)
    return clipped[~clipped.geometry.is_empty]


def reproject_to_state_plane(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to NJ State Plane, NAD83(HARN) (feet)."""
    return gdf.to_crs(NJ_STATE_PLANE_FT)


def label_quadrants(gdf_ft: gpd.GeoDataFrame, center_ft: Point) -> gpd.GeoDataFrame:
    """Label each feature's compass quadrant (NE/NW/SE/SW) relative to a center
    point, plus its distance in feet - used to locate corner parcels."""
    out = gdf_ft.copy()
    out["dist_ft"] = out.geometry.distance(center_ft)
    centroids = out.geometry.centroid
    out["quadrant"] = [
        ("N" if cy > center_ft.y else "S") + ("E" if cx > center_ft.x else "W")
        for cx, cy in zip(centroids.x, centroids.y)
    ]
    return out


def nearest_per_quadrant(gdf_ft: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Given the output of label_quadrants, return the closest feature per quadrant."""
    return gdf_ft.sort_values("dist_ft").groupby("quadrant", as_index=False).first()


def split_leg_centerlines(line: LineString, center: Point, working_length_ft: float) -> list[LineString]:
    """
    Split a line at the point on it nearest `center`, returning up to two pieces
    that each start at that snapped point and extend outward (trimmed to at most
    working_length_ft) - one piece per side of the split.
    """
    snap_dist = line.project(center)
    total = line.length
    legs = []
    if snap_dist > 0:
        head = substring(line, 0, snap_dist)
        head = LineString(list(head.coords)[::-1])  # start at the snap point, head outward
        legs.append(substring(head, 0, min(working_length_ft, head.length)))
    if snap_dist < total:
        tail = substring(line, snap_dist, total)  # already starts at the snap point
        legs.append(substring(tail, 0, min(working_length_ft, tail.length)))
    return legs
