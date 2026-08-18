"""Driveways, parking aisles and lots: the minor carriageways drawn as asphalt.

They exist in the render to explain why the kerbside markings stop where they do, so what matters
here is the OUTLINE - traced where OSM maps the area, widened from a line where it maps only a
centreline, and PavedSurface.extent_is_surveyed keeps the two honestly apart."""


from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from src.render.coords import wgs84_to_state_plane
from src.geometry.model import (
    build_pavement_polygon,
)
from src.geometry.intersection.junction import (DRAWN_WIDTH_FT,
                                                PARKING_AISLE_ONEWAY_WIDTH_FT,
                                                PavedKind, PavedSurface)
from src.geometry.intersection.kerb_sources import drawn_kerb_radius_ft



# One radius for driveways, here rather than in each consumer. Matches the building/crossing
# context radius the renderers use, so a driveway drawn in a view is a driveway the openings were
# derived from - the divergence Driveway's docstring is about.
DRIVEWAY_CONTEXT_RADIUS_M = 130


def to_state_plane(coords) -> list:
    xs, ys = wgs84_to_state_plane.transform([c[0] for c in coords], [c[1] for c in coords])
    return list(zip(xs, ys))


def _context_roadways_ft(center_wgs84: Point, radius_m: float, exclude) -> tuple:
    """The streets AROUND the junction, widened from the kerb that was actually traced.

    `exclude` is the modelled pavement (and the mapped lots): subtracted from every surface, so
    where this project has measured geometry that geometry wins, and no two asphalt polygons end
    up coplanar for Blender to z-fight over.
    """
    from src.geometry.context_roads import (SAMPLE_SPACING_FT, assign_kerbs_to_roads,
                                            is_carriageway, kerb_points, roadway_surface)
    from src.sources.osm_context import fetch_kerbs, fetch_roads

    center_ft = Point(*to_state_plane([(center_wgs84.x, center_wgs84.y)])[0])
    # Clipped to where the KERB data reaches, not to the road fetch radius. Two reasons, and they
    # are the same reason: a way carries on well past it - West Broad Street is one 1,226 ft way -
    # so judging how much of it is traced over a length whose kerbs were never fetched reports a
    # surveyed street as unmapped; and drawing asphalt further out than the kerbs go leaves the
    # corridor losing its edges partway along, which is exactly the mismatch that made the first
    # wide render show street to 938 ft and kerb to 379. A way clipped into two pieces is two
    # surfaces; each is measured on its own.
    reach_ft = drawn_kerb_radius_ft()
    in_range = center_ft.buffer(reach_ft)
    ways = []
    for way in fetch_roads(center_wgs84, radius_m=radius_m):
        tags = way.get("tags", {})
        if not is_carriageway(tags) or len(way.get("coords_wgs84") or []) < 2:
            continue
        clipped = LineString(to_state_plane(way["coords_wgs84"])).intersection(in_range)
        for piece in getattr(clipped, "geoms", [clipped]):
            if piece.geom_type == "LineString" and piece.length > SAMPLE_SPACING_FT:
                ways.append((tags, piece))
    if not ways:
        return ()
    kerb_lines = [LineString(to_state_plane(k["coords_wgs84"]))
                  for k in fetch_kerbs(center_wgs84, radius_m=radius_m)
                  if len(k.get("coords_wgs84") or []) >= 2]

    centerlines = [line for _tags, line in ways]
    per_road = assign_kerbs_to_roads(centerlines, kerb_points(kerb_lines))

    out = []
    for (tags, line), (stations, offsets) in zip(ways, per_road):
        surface, traced, width_ft = roadway_surface(line, stations, offsets, tags)
        if surface is None:
            continue
        if exclude is not None:
            surface = surface.difference(exclude)
        for piece in getattr(surface, "geoms", [surface]):
            if piece.geom_type == "Polygon" and not piece.is_empty and piece.area > 1.0:
                out.append(PavedSurface(kind=PavedKind.ROADWAY, line=line, tags=tags,
                                        surface=piece, traced_sides=frozenset(traced),
                                        drawn_width_ft=width_ft))
    return tuple(out)


def _paved_surfaces_ft(center_wgs84: Point, corner_fillets: dict | None = None) -> tuple:
    """Every mapped driveway, parking aisle and parking lot near this junction, projected once.

    The lots are built first because they SUBTRACT from the aisles. An aisle inside a mapped lot
    is already paved by the lot's own surveyed outline, and drawing both leaves two coplanar
    surfaces at the same height - which in Blender is not redundancy, it is z-fighting (the
    project has hit that before; see MARKING_CLEARANCE_M). 6 of the borough's 20 aisles are inside
    a lot, so the other 14 still need their strips.

    ...and the surrounding STREETS, once `corner_fillets` says where the modelled pavement is so
    they can be cut around it. Same reason they are here rather than fetched by each renderer:
    a roadway is street geometry, resolved once at load, not render dressing.

    The radius follows the frame (src/render/frame.py:context_radius_m), so a zoom-out widens
    what there is to see and not just how much ground is in shot. Imported lazily: this is the
    geometry layer, and a module-level import of the render layer would invert the dependency
    for what is one environment variable.
    """
    from src.render.frame import context_radius_m
    from src.sources.osm_context import (fetch_driveways, fetch_parking_aisles,
                                         fetch_parking_lots)

    radius_m = context_radius_m(DRIVEWAY_CONTEXT_RADIUS_M)
    lots = []
    for lot in fetch_parking_lots(center_wgs84, radius_m=radius_m):
        coords = lot.get("coords_wgs84") or []
        if len(coords) < 4:
            continue
        surface = Polygon(to_state_plane(coords)).buffer(0)
        if surface.geom_type == "Polygon" and not surface.is_empty:
            lots.append(PavedSurface(kind=PavedKind.PARKING_LOT, surface=surface,
                                     way_id=lot.get("id"), tags=lot.get("tags", {})))
    paved_by_lots = unary_union([lot.surface for lot in lots]) if lots else None

    out = list(lots)
    for kind, fetch in ((PavedKind.DRIVEWAY, fetch_driveways),
                        (PavedKind.PARKING_AISLE, fetch_parking_aisles)):
        for way in fetch(center_wgs84, radius_m=radius_m):
            coords = way.get("coords_wgs84") or []
            if len(coords) < 2:
                continue
            tags = way.get("tags", {})
            line = LineString(to_state_plane(coords))
            width_ft = (PARKING_AISLE_ONEWAY_WIDTH_FT
                        if kind == PavedKind.PARKING_AISLE and tags.get("oneway") == "yes"
                        else DRAWN_WIDTH_FT[kind])
            # Flat caps: a round cap would put a half-disc on the end of every driveway, out in
            # the garden it leads to.
            surface = line.buffer(width_ft / 2, cap_style=2)
            if kind == PavedKind.PARKING_AISLE and paved_by_lots is not None:
                surface = surface.difference(paved_by_lots)
            for piece in getattr(surface, "geoms", [surface]):
                if piece.geom_type == "Polygon" and not piece.is_empty:
                    out.append(PavedSurface(kind=kind, line=line, way_id=way.get("id"),
                                            tags=tags, surface=piece, drawn_width_ft=width_ft))

    # The streets last, cut around everything already resolved: the junction's own measured
    # pavement first of all, and the mapped lots for the same reason the aisles are.
    keep_clear = [build_pavement_polygon(corner_fillets)] if corner_fillets else []
    if paved_by_lots is not None:
        keep_clear.append(paved_by_lots)
    exclude = unary_union([g for g in keep_clear if g is not None and not g.is_empty]) or None
    return tuple(out) + _context_roadways_ft(center_wgs84, radius_m, exclude)
