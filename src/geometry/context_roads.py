"""The streets around the junction, built from the kerb that was actually traced.

WHAT A CONTEXT ROAD IS. One OSM highway way, widened into asphalt from the traced kerbs:

  * BOTH sides traced - West and East Broad Street - and the surface is measured.
  * ONE side traced - mirrored at the class-assumed half width. Surface flagged assumed.
  * NEITHER traced - Model Ave, Railroad Pl - a plain ribbon at the assumed width.

"Is this kerb ours" has had two answers (near the centre; along a leg). Drawing needs a third:
**is it in the picture**. A kerb 400 ft down Broad St is not this junction's corner return and
not on a modelled leg, but it is a fact somebody surveyed, and a drawing that omits it is worse
than one that includes it.

A kerb vertex is assigned to the NEAREST road centreline, not to every road within reach -
without that, Broad St claims Railroad Place's kerbs across the back of a lot.

The modelled pavement is subtracted from every surface this builds, so measured geometry always
wins where the two overlap.
"""
import numpy as np
from shapely.geometry import LineString, Polygon

from src.geometry.model import station_offset_many

# A kerb further from a centreline than this belongs to some other street. Generous enough for
# a 68 ft leg's own kerbs plus a corner return, tight enough that a parallel street one lot
# back is never mistaken for the far kerb of this one.
MAX_HALF_WIDTH_FT = 45.0

# How often the edges are sampled along a way, and how wide a station window each sample takes
# its kerb offset from. The window is what makes an intermittently traced kerb produce a smooth
# edge rather than a sawtooth: the offsets inside it are reduced by MEDIAN, so one stray vertex
# on a corner return does not pull the whole edge out.
SAMPLE_SPACING_FT = 15.0
STATION_WINDOW_FT = 30.0

# A side counts as SURVEYED only if this much of its length carried a kerb to measure.
# It sets the PROVENANCE FLAG only. A side below it still uses every kerb that was traced;
# discarding real measurements because there were not enough of them would be the same
# over-correction as trusting a third of a street and calling it surveyed.
MIN_TRACED_FRACTION = 0.7

# Assumed widths, by highway class, for a street nobody has traced. Every one of these is a
# guess and is labelled as one (extent_is_surveyed is False), on the same terms
# DRIVEWAY_DRAWN_WIDTH_FT is. They are curb-to-curb, not pavement-marking widths.
ROADWAY_DRAWN_WIDTH_FT = {
    "motorway": 48.0, "trunk": 44.0, "primary": 40.0, "secondary": 36.0,
    "tertiary": 32.0, "unclassified": 26.0, "residential": 26.0,
    "living_street": 20.0, "service": 18.0, "track": 12.0,
}
ROADWAY_DEFAULT_WIDTH_FT = 24.0

# Ways that are not carriageway at all. A footway is a sidewalk (drawn from its own layer), and
# a driveway or a parking aisle is already a PavedSurface - drawing it twice puts two coplanar
# asphalt polygons at the same height, which is z-fighting, not redundancy.
NOT_CARRIAGEWAY = frozenset({"footway", "path", "cycleway", "steps", "pedestrian", "bridleway",
                             "corridor", "construction", "proposed", "raceway", "elevator"})
NOT_CARRIAGEWAY_SERVICE = frozenset({"driveway", "parking_aisle"})


def is_carriageway(tags: dict) -> bool:
    """Whether an OSM way is a street this project should draw asphalt for."""
    highway = tags.get("highway")
    if not highway or highway in NOT_CARRIAGEWAY:
        return False
    if highway == "service" and tags.get("service") in NOT_CARRIAGEWAY_SERVICE:
        return False
    return True


def assumed_width_ft(tags: dict) -> float:
    """How wide to draw a street nobody traced.

    OSM's own `width` tag first (a mapper who recorded one measured something), then the highway
    class. A `_link` takes its parent class.
    """
    raw = tags.get("width")
    if raw is not None:
        try:                                  # metres in OSM unless a unit is given
            metres = float(str(raw).replace("m", "").strip())
            if 6.0 <= metres * 3.28084 <= 80.0:
                return metres * 3.28084
        except ValueError:
            pass
    highway = (tags.get("highway") or "").removesuffix("_link")
    return ROADWAY_DRAWN_WIDTH_FT.get(highway, ROADWAY_DEFAULT_WIDTH_FT)


# How finely a traced kerb is resampled before it is read as an edge. DENSIFIED, not taken at
# its vertices: a straight kerb is mapped with as few vertices as it takes to be straight, so
# reading vertices alone misses most of it.
KERB_SAMPLE_SPACING_FT = 8.0


def kerb_points(kerb_lines) -> np.ndarray:
    """Every traced kerb, resampled to an (n, 2) array of points along it.

    Along it, not at its corners - see KERB_SAMPLE_SPACING_FT.
    """
    points = []
    for line in kerb_lines:
        if line.length <= 0:
            points.extend(line.coords)
            continue
        n = max(int(line.length // KERB_SAMPLE_SPACING_FT) + 1, 2)
        points.extend((line.interpolate(d).x, line.interpolate(d).y)
                      for d in np.linspace(0.0, line.length, n))
    return np.asarray(points, dtype=float) if points else np.empty((0, 2), dtype=float)


def assign_kerbs_to_roads(centerlines: list, kerb_points: np.ndarray) -> list:
    """Which road each traced kerb vertex belongs to: the one it is nearest, or none.

    Returns [(stations, offsets)] per centreline, holding only that road's own vertices.

    Nearest rather than "within reach of", because reach alone lets two parallel streets claim
    each other's kerbs and both widen to meet in the middle. Computed as one numpy pass per
    road over all vertices.
    """
    if not centerlines or not len(kerb_points):
        return [(np.empty(0), np.empty(0)) for _ in centerlines]

    stations, offsets = [], []
    for line in centerlines:
        s, o = station_offset_many(line, kerb_points)
        stations.append(s)
        offsets.append(o)
    station_grid, offset_grid = np.vstack(stations), np.vstack(offsets)

    # Off either end of a way, the station leaves [0, length] and the vertex is past what this
    # way covers - not its kerb, however close the perpendicular distance looks.
    lengths = np.asarray([line.length for line in centerlines], dtype=float)[:, None]
    in_span = (station_grid >= 0) & (station_grid <= lengths)
    reach = np.where(in_span, np.abs(offset_grid), np.inf)
    nearest = np.argmin(reach, axis=0)
    claimed = np.take_along_axis(reach, nearest[None, :], axis=0)[0] <= MAX_HALF_WIDTH_FT

    out = []
    for i in range(len(centerlines)):
        mine = claimed & (nearest == i)
        out.append((station_grid[i][mine], offset_grid[i][mine]))
    return out


def _edge_offsets(line: LineString, stations: np.ndarray, offsets: np.ndarray,
                   assumed_width_ft_: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, set]:
    """(sample stations, left offsets, right offsets, which sides are SURVEYED).

    Left is positive and right negative, the sign convention station_offset_many already uses.
    Each sample takes the MEDIAN of its side's kerb offsets inside a STATION_WINDOW_FT window.
    Fallback when a sample has nothing in the window, most-measured first:

      1. This side's OWN median, where the side was traced anywhere.
      2. The OTHER side's traced edge, less the assumed width - placing it a full width off
         the traced kerb rather than half off the centreline is the difference between using
         the measurement and ignoring it.
      3. Half the assumed width either side of the centreline.

    The fraction test decides only which sides are reported SURVEYED. It never discards a
    measurement - see MIN_TRACED_FRACTION.
    """
    n = max(int(line.length // SAMPLE_SPACING_FT) + 1, 2)
    samples = np.linspace(0.0, line.length, n)
    half_window = STATION_WINDOW_FT / 2

    measured, coverage = {}, {}
    for side, keep in (("left", offsets > 0), ("right", offsets < 0)):
        st, off = stations[keep], offsets[keep]
        values = np.full(len(samples), np.nan)
        for i, station in enumerate(samples):
            window = np.abs(st - station) <= half_window
            if window.any():
                values[i] = float(np.median(off[window]))
        measured[side] = values
        coverage[side] = float(np.isfinite(values).mean()) if len(samples) else 0.0

    edges = {}
    for side in ("left", "right"):
        sign = 1.0 if side == "left" else -1.0
        values, other = measured[side].copy(), measured["right" if side == "left" else "left"]
        gaps = ~np.isfinite(values)
        if gaps.any():
            if np.isfinite(values).any():                                  # (1) its own median
                values[gaps] = float(np.nanmedian(values))
            elif np.isfinite(other).any():                                 # (2) off the far kerb
                fallback = np.where(np.isfinite(other), other, np.nanmedian(other))
                values[gaps] = (fallback + sign * assumed_width_ft_)[gaps]
            else:                                                          # (3) nothing traced
                values[gaps] = sign * assumed_width_ft_ / 2
        edges[side] = values

    traced = {side for side in ("left", "right") if coverage[side] >= MIN_TRACED_FRACTION}
    return samples, edges["left"], edges["right"], traced


def _edge_points(line: LineString, stations: np.ndarray, offsets: np.ndarray) -> list:
    """The inverse of station_offset_many: a point per (station, offset) in the road's frame."""
    from src.geometry.model import point_at

    return [point_at(line, float(s), float(o)) for s, o in zip(stations, offsets)]


def roadway_surface(line: LineString, stations: np.ndarray, offsets: np.ndarray,
                     tags: dict) -> tuple[Polygon | None, set, float]:
    """One street's asphalt, from its traced kerbs where there are any.

    Returns (polygon, traced sides, assumed width used). Walks the left edge out and the right
    edge back at the same stations.
    """
    width_ft = assumed_width_ft(tags)
    samples, left, right, traced = _edge_offsets(line, stations, offsets, width_ft)
    ring = _edge_points(line, samples, left) + _edge_points(line, samples[::-1], right[::-1])
    if len(ring) < 4:
        return None, traced, width_ft
    surface = Polygon(ring).buffer(0)       # buffer(0) fixes a self-touch where a way doubles back
    if surface.is_empty:
        return None, traced, width_ft
    if surface.geom_type != "Polygon":      # a bow-tie splits; keep the substantial piece
        surface = max(surface.geoms, key=lambda g: g.area)
    return surface, traced, width_ft
