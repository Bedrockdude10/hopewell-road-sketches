"""The streets around the junction, built from the kerb that was actually traced.

WHY THIS EXISTS. The render framed a wide view and the road stopped dead at the modelled
legs - a cross of asphalt with sharply cut ends floating on grass, while the buildings and
driveways around it ran out to 200 m. The legs were not the cause. The cause was that the
exported kerbs came from `kerb_lines_with_tags_ft` with no `legs` argument, i.e. the NEAR
set: everything within KERB_NEAR_JUNCTION_FT (80 ft) of the junction centre. That test is
right for what its docstring says it is for - fitting a corner radius, measuring a width -
and it drops 8,938 ft of traced kerb within 600 m of Broad & Greenwood on the floor.

"Is this kerb ours" has had two answers here (near the centre; along a leg). Drawing needs a
third, and it is a different question from both: **is it in the picture**. A kerb 400 ft down
Broad St is not this junction's corner return and not on a modelled leg, but it is a fact
somebody surveyed and it is inside the frame, so a drawing that omits it is worse than one
that includes it.

WHAT A CONTEXT ROAD IS. One OSM highway way, widened into asphalt. The width comes from the
traced kerbs wherever they exist, per side and per station:

  * BOTH sides traced - West and East Broad Street, essentially the whole corridor - and the
    surface is measured. Its extent is as surveyed as a parking lot's outline.
  * ONE side traced. The traced kerb is used where it is, and the other edge is mirrored at
    the class-assumed half width. Half the outline is real; the surface is flagged assumed,
    because the street was never mapped on the far side and a symmetric guess is what the
    legs that DO have both sides show to be wrong by 0.2-10.3 ft.
  * NEITHER traced - Model Ave, Railroad Pl, Lafayette, Front St - and it is a plain ribbon
    at the assumed width for its highway class, on exactly the terms a driveway already is.

This is the same surveyed/assumed split the curb lines have carried since Phase 2, and it
reaches the drawing the same way: `extent_is_surveyed` decides whether the plan view outlines
a surface solid or dashed, so a reader can see which edges were measured.

A kerb vertex is assigned to the NEAREST road centreline, not to every road within reach.
Without that, Broad St claims Railroad Place's kerbs across the back of a lot and widens to
swallow them, which is the same failure `_runs_along_a_leg` exists to prevent one level down.

Nothing here touches the junction itself. The modelled pavement is subtracted from every
surface this builds, so the measured geometry always wins where the two overlap and there is
no coplanar z-fighting in Blender - the rule parking aisles already follow against lots.
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

# A side counts as SURVEYED only if this much of its length carried a kerb to measure. The
# measured coverage at Broad & Greenwood is sharply bimodal - West Broad 100%/100%, East Broad
# 95%/100%, against 44%/59% on North Greenwood, 38%/38% on Blackwell and 30%/30% on South
# Greenwood - so the threshold sits in the gap and is not a knob anyone has to tune.
#
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

    OSM's own `width` tag first, because a mapper who recorded one measured something. Then the
    highway class. A `_link` takes its parent class - `primary_link` is a slip road off a
    primary, and there is no separate table for it.
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
# its vertices: a straight kerb is mapped with as few vertices as it takes to be straight, and
# the borough's 55 ways near Broad & Greenwood carry about three apiece - so a 300 ft run
# contributes two points, both at the ends. Reading vertices alone found kerb at 28 of 82
# stations along West Broad and concluded the street was untraced, which is the opposite of
# what is on the ground.
KERB_SAMPLE_SPACING_FT = 8.0


def kerb_points(kerb_lines) -> np.ndarray:
    """Every traced kerb, resampled to an (n, 2) array of points along it.

    Along it, not at its corners. See KERB_SAMPLE_SPACING_FT - this is the difference between
    measuring a street that was traced and reporting it as unmapped.
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
    road over all vertices - station_offset_many is vectorized, and the alternative (a shapely
    distance call per vertex per road) is ~110k calls at one site.
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
    What a sample with nothing in the window falls back to is the whole question, and it is
    answered in three steps, most-measured first:

      1. This side's OWN median, where the side was traced anywhere. A street traced for a third
         of its length still knows how far out its kerb sits; carrying that across the untraced
         stretch keeps the edge continuous and keeps it at the width somebody measured.
      2. The OTHER side's traced edge, less the assumed width for the class, where only one side
         was mapped. Placing it a full width off the traced kerb rather than a half width off
         the centreline is the difference between using the measurement and ignoring it - OSM
         alignments are not centred, and the legs that have both kerbs show that by 0.2-10.3 ft.
      3. Half the assumed width either side of the centreline, for a street nobody traced.

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
    from src.geometry.model import _point_at

    return [_point_at(line, float(s), float(o)) for s, o in zip(stations, offsets)]


def roadway_surface(line: LineString, stations: np.ndarray, offsets: np.ndarray,
                     tags: dict) -> tuple[Polygon | None, set, float]:
    """One street's asphalt, from its traced kerbs where there are any.

    Returns (polygon, traced sides, assumed width used). The polygon walks the left edge out
    and the right edge back, which is a simple ring because both edges are sampled at the same
    stations in the same frame.
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
