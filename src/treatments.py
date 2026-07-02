"""Parametric pedestrian-safety treatments: composable geometry transforms over a
DesignState. Each treatment returns a new DesignState so scenarios can be stacked
without mutating the baseline (existing-conditions) model."""
from copy import deepcopy
from dataclasses import dataclass, field

from shapely.geometry import Polygon

from src.geometry_model import Leg, fillet_curb_corner, leg_clearance_ft

NACTO_MIN_REFUGE_ISLAND_WIDTH_FT = 6


@dataclass
class DesignState:
    """A mutable-by-copy snapshot of intersection geometry. Treatments clone the
    state, apply one change, and return the clone - so `state = bump_out(state, ...)`
    chains cleanly and the original scenario is never touched."""
    legs: dict
    corner_fillets: dict
    refuge_islands: dict = field(default_factory=dict)   # name -> {"polygon": Polygon, "width_ft": float}
    raised_crossings: dict = field(default_factory=dict)  # leg name -> Polygon
    crosswalk_styles: dict = field(default_factory=dict)  # leg name -> "lines" | "continental" | "ladder"
    notes: list = field(default_factory=list)

    @classmethod
    def from_model(cls, model) -> "DesignState":
        return cls(legs=deepcopy(model.legs), corner_fillets=deepcopy(model.corner_fillets))

    def clone(self) -> "DesignState":
        return deepcopy(self)


def bump_out(state: DesignState, corner: tuple[str, str], radius_ft: float) -> DesignState:
    """
    Curb extension / tightened turn radius: rebuild one corner's fillet at a
    smaller (or larger) radius. A smaller radius simultaneously shortens the
    pedestrian crossing distance (the curb physically extends into the corner)
    and slows/tightens the vehicle turning path - the same geometric move
    serves both treatments described in the design brief.

    `corner` is a (leg_a, leg_b) key as produced by build_corner_fillets.
    """
    new_state = state.clone()
    leg_a, leg_b = corner
    if leg_a not in new_state.legs or leg_b not in new_state.legs:
        raise KeyError(f"Corner {corner} references a leg not present in this state.")
    trimmed_a, arc, trimmed_b = fillet_curb_corner(
        new_state.legs[leg_a].left_curb, new_state.legs[leg_b].right_curb, radius_ft
    )
    new_state.corner_fillets[corner] = {"trimmed_a": trimmed_a, "arc": arc, "trimmed_b": trimmed_b, "radius_ft": radius_ft}
    new_state.notes.append(f"bump_out({corner}, radius_ft={radius_ft})")
    return new_state


def refuge_island(state: DesignState, leg_name: str, offset_ft: float, width_ft: float,
                   along_road_ft: float = 20, name: str | None = None) -> DesignState:
    """
    Add a raised pedestrian refuge island splitting `leg_name`'s roadway,
    centered `offset_ft` from the intersection along the centerline.

    width_ft is the island's extent in the direction pedestrians cross (i.e.
    perpendicular to the road) - NACTO's minimum is 6 ft so a person/wheelchair
    can wait clear of both travel directions. along_road_ft is the island's
    length parallel to the road (how much of the crosswalk it shelters).
    """
    if width_ft < NACTO_MIN_REFUGE_ISLAND_WIDTH_FT:
        raise ValueError(
            f"Refuge island width {width_ft} ft is below the NACTO minimum of {NACTO_MIN_REFUGE_ISLAND_WIDTH_FT} ft."
        )
    new_state = state.clone()
    leg = new_state.legs[leg_name]
    centerline = leg.centerline

    p0 = centerline.interpolate(max(offset_ft - along_road_ft / 2, 0))
    p1 = centerline.interpolate(offset_ft)
    p2 = centerline.interpolate(min(offset_ft + along_road_ft / 2, centerline.length))
    dx, dy = p2.x - p0.x, p2.y - p0.y
    length = (dx**2 + dy**2) ** 0.5
    ux, uy = dx / length, dy / length          # unit vector along the road
    nx, ny = -uy, ux                            # unit normal (perpendicular)

    half_w = width_ft / 2
    corners = [
        (p0.x + nx * half_w, p0.y + ny * half_w),
        (p2.x + nx * half_w, p2.y + ny * half_w),
        (p2.x - nx * half_w, p2.y - ny * half_w),
        (p0.x - nx * half_w, p0.y - ny * half_w),
    ]
    island_name = name or f"{leg_name}_refuge_{int(offset_ft)}ft"
    new_state.refuge_islands[island_name] = {"polygon": Polygon(corners), "width_ft": width_ft}
    new_state.notes.append(f"refuge_island({leg_name}, offset_ft={offset_ft}, width_ft={width_ft})")
    return new_state


def raise_crossing(state: DesignState, leg_name: str, crossing_width_ft: float = 10) -> DesignState:
    """
    Mark the crosswalk over `leg_name`'s roadway (right at the intersection) as
    a raised crossing (speed table to sidewalk grade). In plan view this is
    just the crosswalk footprint, rendered distinctly; Phase 4 gives it height.
    """
    new_state = state.clone()
    leg = new_state.legs[leg_name]
    if leg.left_curb is None or leg.right_curb is None:
        raise ValueError(f"Leg {leg_name!r} has no curb lines (width unknown) - can't place a crossing on it.")

    centerline = leg.centerline
    # Start beyond the curve of this leg's corner fillets, not at the
    # intersection point itself - a crossing placed right at the corner point
    # lands inside the curb-return curve rather than on the straight section
    # of roadway where a real crosswalk would sit.
    start = leg_clearance_ft(leg_name, new_state.legs, new_state.corner_fillets)
    p0 = centerline.interpolate(min(start, centerline.length))
    p1 = centerline.interpolate(min(start + crossing_width_ft, centerline.length))
    dx, dy = p1.x - p0.x, p1.y - p0.y
    length = (dx**2 + dy**2) ** 0.5
    ux, uy = dx / length, dy / length

    half_w = leg.curb_to_curb_ft / 2
    nx, ny = -uy, ux
    corners = [
        (p0.x + nx * half_w, p0.y + ny * half_w),
        (p1.x + nx * half_w, p1.y + ny * half_w),
        (p1.x - nx * half_w, p1.y - ny * half_w),
        (p0.x - nx * half_w, p0.y - ny * half_w),
    ]
    new_state.raised_crossings[leg_name] = Polygon(corners)
    new_state.notes.append(f"raise_crossing({leg_name}, crossing_width_ft={crossing_width_ft})")
    return new_state


VALID_CROSSWALK_STYLES = ("lines", "continental", "ladder")


def upgrade_crosswalk_markings(state: DesignState, leg_name: str, style: str) -> DesignState:
    """
    Repaint a leg's crosswalk to a more visible marking style. FHWA/NACTO both
    rank visibility roughly lines < continental < ladder - "lines" (two thin
    transverse boundary lines) is what most of this intersection has today;
    upgrading to continental or ladder is a real, low-cost pedestrian-safety
    treatment on its own, independent of any geometry change.
    """
    if style not in VALID_CROSSWALK_STYLES:
        raise ValueError(f"Unknown crosswalk style {style!r} - expected one of {VALID_CROSSWALK_STYLES}")
    new_state = state.clone()
    new_state.crosswalk_styles[leg_name] = style
    new_state.notes.append(f"upgrade_crosswalk_markings({leg_name}, style={style!r})")
    return new_state


def build_sidewalk_pieces(state: DesignState, sidewalk_width_ft: float = 6) -> list[Polygon]:
    """
    Approximate sidewalk band around the pavement: re-run the fillet pipeline on
    the same legs widened by sidewalk_width_ft per side, at (existing corner
    radius + sidewalk_width_ft), then take each corner's arc-to-arc and
    curb-to-curb gap as a separate sidewalk piece. This is visual context for
    the Phase 4 render, not survey-grade geometry.
    """
    outer_legs = {
        name: Leg(name=name, centerline=leg.centerline, curb_to_curb_ft=leg.curb_to_curb_ft + 2 * sidewalk_width_ft)
        for name, leg in state.legs.items()
    }

    pieces = []
    outer_by_corner = {}
    for corner, inner in state.corner_fillets.items():
        if "error" in inner:
            continue
        leg_a, leg_b = corner
        radius_ft = inner.get("radius_ft", 20) + sidewalk_width_ft
        try:
            trimmed_a, arc, trimmed_b = fillet_curb_corner(
                outer_legs[leg_a].left_curb, outer_legs[leg_b].right_curb, radius_ft
            )
        except ValueError:
            continue
        outer_by_corner[corner] = {"trimmed_a": trimmed_a, "arc": arc, "trimmed_b": trimmed_b}
        pieces.append(Polygon(list(inner["arc"].coords) + list(reversed(arc.coords))))

    for corner, inner in state.corner_fillets.items():
        outer = outer_by_corner.get(corner)
        if outer is None:
            continue
        pieces.append(Polygon(list(inner["trimmed_a"].coords) + list(reversed(outer["trimmed_a"].coords))))
        pieces.append(Polygon(list(inner["trimmed_b"].coords) + list(reversed(outer["trimmed_b"].coords))))

    return [p for p in pieces if p.is_valid and not p.is_empty]
