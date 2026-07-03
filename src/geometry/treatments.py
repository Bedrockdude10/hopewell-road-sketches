"""Parametric pedestrian-safety treatments: composable geometry transforms over a
DesignState. Each treatment returns a new DesignState so scenarios can be stacked
without mutating the baseline (existing-conditions) model."""
from copy import deepcopy
from dataclasses import dataclass, field

from shapely.geometry import Polygon

from src.geometry.model import Leg, fillet_curb_corner, leg_clearance_ft

NACTO_MIN_REFUGE_ISLAND_WIDTH_FT = 6
LANE_NARROWING_DEFAULT_STRIPE_FT = 5.0  # common low-cost NACTO paint buffer/shoulder-stripe width
CORNER_HATCHING_DEFAULT_DEPTH_FT = 6.0  # paint-only zone depth, comparable footprint to a modest real curb extension
CORNER_APRON_DEFAULT_EXTENT_FT = 5.0  # mountable-apron zone depth - same shape as hatching, different surface finish

# What's actually painted down the middle of a leg today: a single dashed
# yellow line (default - the ordinary two-way-undivided-road marking), a solid
# double yellow (no-passing zone), or none at all (some real local streets
# genuinely have no centerline paint). Unlike crosswalk style, there's no OSM
# tag for this - it's read directly from a site's config.yaml per leg (see
# sites/README.md), confirmed the same way as the `signals` block (street-view
# photo review, not a field survey).
DEFAULT_CENTERLINE_STYLE = "single_yellow_dashed"
VALID_CENTERLINE_STYLES = ("single_yellow_dashed", "double_yellow", "none")


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
    centerline_styles: dict = field(default_factory=dict)  # leg name -> one of VALID_CENTERLINE_STYLES - seeded
                                                             # from config.yaml in from_model(), see set_centerline_style
    lane_narrowing: dict = field(default_factory=dict)  # leg name -> stripe_width_ft (paint-only, no curb change)
    lane_narrowing_line_only: set = field(default_factory=set)  # leg names (subset of lane_narrowing) that get
                                                                  # ONLY the solid edge/taper line delineating the
                                                                  # lane, no diagonal chevron fill - see add_lane_narrowing's
                                                                  # line_only param
    lane_narrowing_sides: dict = field(default_factory=dict)  # leg name -> ("left",) | ("right",) | ("left","right") -
                                                                 # which side(s) of the leg add_lane_narrowing actually
                                                                 # narrowed (defaults to both if a leg is absent here -
                                                                 # see add_lane_narrowing's sides param). Only ever a
                                                                 # strict subset when something else (e.g. add_marked_parking)
                                                                 # already owns the other side's edge.
    bollard_lines: dict = field(default_factory=dict)  # leg name -> spacing_ft (see add_bollards - requires
                                                         # lane_narrowing on the same leg, sits inside that buffer)
    parking_zones: dict = field(default_factory=dict)  # (leg name, "left"|"right") -> {"depth_ft", "stall_length_ft",
                                                         # "curb_offset_ft"} - marked curbside parking (paint-only,
                                                         # no curb change) - see add_marked_parking
    parking_buffer_bollards: dict = field(default_factory=dict)  # (leg name, "left"|"right") -> spacing_ft - flex-post
                                                                   # bollards centered in that zone's curb_offset_ft
                                                                   # buffer (requires curb_offset_ft > 0) - see
                                                                   # add_parking_buffer_bollards
    corner_hatching: dict = field(default_factory=dict)  # corner tuple -> depth_ft (paint-only, no curb change)
    corner_aprons: dict = field(default_factory=dict)  # corner tuple -> extent_ft (mountable apron, no curb change)
    crosswalk_offset_overrides: dict = field(default_factory=dict)  # leg name -> +/- delta_ft on top of the
                                                                     # normally-resolved offset (see shift_crosswalk)
    extra_props: list = field(default_factory=list)  # [{"leg","type","offset_ft","side","note"}] - see add_extra_prop
    notes: list = field(default_factory=list)

    @classmethod
    def from_model(cls, model) -> "DesignState":
        centerline_styles = {
            name: leg_cfg.get("centerline_style", DEFAULT_CENTERLINE_STYLE)
            for name, leg_cfg in model.config["legs"].items()
        }
        return cls(legs=deepcopy(model.legs), corner_fillets=deepcopy(model.corner_fillets),
                   centerline_styles=centerline_styles)

    def clone(self) -> "DesignState":
        return deepcopy(self)


def find_corner(state: DesignState, leg_a: str, leg_b: str) -> tuple[str, str]:
    """Look up the (name_a, name_b) key in state.corner_fillets for the corner
    where leg_a and leg_b meet, regardless of which order build_corner_fillets
    happened to store it in (it sorts by compass bearing, not by call-site
    convenience) - corners are identified by which two legs meet there, not by
    tuple order."""
    wanted = {leg_a, leg_b}
    for corner in state.corner_fillets:
        if set(corner) == wanted:
            return corner
    raise KeyError(f"No corner between {leg_a!r} and {leg_b!r} in this state.")


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


def set_centerline_style(state: DesignState, leg_name: str, style: str) -> DesignState:
    """
    Change what's painted down the middle of a leg: 'single_yellow_dashed'
    (ordinary two-way marking), 'double_yellow' (solid no-passing zone), or
    'none' (some real local streets have no centerline paint at all). Unlike
    upgrade_crosswalk_markings, this isn't a visibility ranking - it's just
    what's actually there, or a proposal's choice to change it - so any value
    is a valid target, not just an "upgrade."
    """
    if style not in VALID_CENTERLINE_STYLES:
        raise ValueError(f"Unknown centerline style {style!r} - expected one of {VALID_CENTERLINE_STYLES}")
    new_state = state.clone()
    new_state.centerline_styles[leg_name] = style
    new_state.notes.append(f"set_centerline_style({leg_name}, style={style!r})")
    return new_state


def add_lane_narrowing(state: DesignState, leg_name: str,
                        stripe_width_ft: float = LANE_NARROWING_DEFAULT_STRIPE_FT,
                        line_only: bool = False, sides: tuple = ("left", "right")) -> DesignState:
    """Paint-only visual lane narrowing: a striped buffer/shoulder painted along
    one or both curbs of a leg (sides - see below). Zero curb/pavement
    geometry change - the lowest-cost alternative to bump_out()'s real curb
    extension, achieving the same 'narrower-looking travel way' cue with
    paint instead of concrete.

    line_only=True skips the diagonal chevron fill entirely - just the solid
    line (straight run + corner taper) delineating the outside of the real
    travel lane, nothing painted in the buffer itself. Useful as a debugging/
    comparison scenario (bare minimum lane-width marking, easy to check by eye
    or by measurement against the plan view without chevron hatch density
    affecting the read) as well as a real low-cost treatment option in its
    own right.

    sides restricts which side(s) of the leg get narrowed - defaults to both
    (the usual case: a real two-lane road narrowed symmetrically). Pass a
    single side (e.g. ("left",)) when the OTHER side's edge is already owned
    by a different treatment - e.g. a marked-parking lane (add_marked_parking)
    already delineates its own side; this just adds the matching plain
    delineating line on the opposite (entering-traffic) side, matching real
    curb-to-curb width there but with no buffer painted for it."""
    if leg_name not in state.legs:
        raise KeyError(f"Leg {leg_name!r} not present in this state.")
    new_state = state.clone()
    new_state.lane_narrowing[leg_name] = stripe_width_ft
    new_state.lane_narrowing_sides[leg_name] = sides
    if line_only:
        new_state.lane_narrowing_line_only.add(leg_name)
    else:
        new_state.lane_narrowing_line_only.discard(leg_name)
    new_state.notes.append(
        f"add_lane_narrowing({leg_name}, stripe_width_ft={stripe_width_ft}, line_only={line_only}, sides={sides})")
    return new_state


PARKING_STALL_DEPTH_DEFAULT_FT = 8.0  # AASHTO/NACTO typical parallel-parking lane depth (curb to travel-lane edge)
PARKING_STALL_LENGTH_DEFAULT_FT = 22.0  # AASHTO/NACTO typical parallel-parking stall length
LEGAL_PARKING_SETBACK_FT = 25.0  # NJSA 39:4-138: no stopping/standing/parking within 25 ft of a marked crosswalk at
                                  # an intersection - a real legal minimum, not a rendering choice. Marked parking
                                  # (src/render/export.py/plan_view.py) starts at max(this distance past the real
                                  # crosswalk, leg_clearance_ft's physical past-the-corner-curve point) - whichever
                                  # is farther from the intersection - so it never starts somewhere a car legally
                                  # couldn't park even if the curb geometry alone would allow it.


def add_marked_parking(state: DesignState, leg_name: str, side: str,
                        depth_ft: float = PARKING_STALL_DEPTH_DEFAULT_FT,
                        stall_length_ft: float = PARKING_STALL_LENGTH_DEFAULT_FT,
                        curb_offset_ft: float = 0.0) -> DesignState:
    """Marked curbside parallel parking along one side of a leg: a lane-edge
    line depth_ft in from the curb, plus perpendicular divider ticks every
    stall_length_ft (src/geometry/model.py:parking_lane_edge_line_ft /
    parking_stall_lines_ft) - paint-only, zero curb/pavement change, same
    convention as add_lane_narrowing/add_corner_hatching in that regard.
    Independent of add_lane_narrowing - a leg can have marked parking with or
    without a separate travel-lane-narrowing buffer on the same or other
    side; nothing here assumes the two are combined, though a scenario is
    free to call both (e.g. narrow the near lane while marking parking in
    what the SLD calls the far side's shoulder zone).

    curb_offset_ft > 0 pulls the parking lane in from the curb by that much,
    leaving a striped no-parking buffer between the curb and the parking
    lane itself (so parking sits directly against the active travel lane
    instead of against the curb) - see build_striped_parking_buffer_polygons
    in src/render/export.py/plan_view.py, which paints that buffer with the
    same chevron treatment as add_lane_narrowing. 0 (the default) means the
    parking lane starts right at the curb, no buffer."""
    if leg_name not in state.legs:
        raise KeyError(f"Leg {leg_name!r} not present in this state.")
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    new_state = state.clone()
    new_state.parking_zones[(leg_name, side)] = {
        "depth_ft": depth_ft, "stall_length_ft": stall_length_ft, "curb_offset_ft": curb_offset_ft,
    }
    new_state.notes.append(
        f"add_marked_parking({leg_name}, side={side!r}, depth_ft={depth_ft}, stall_length_ft={stall_length_ft}, "
        f"curb_offset_ft={curb_offset_ft})")
    return new_state


BOLLARD_DEFAULT_SPACING_FT = 10.0  # typical flex-post delineator spacing for a channelized buffer


def add_parking_buffer_bollards(state: DesignState, leg_name: str, side: str,
                                 spacing_ft: float = BOLLARD_DEFAULT_SPACING_FT) -> DesignState:
    """Plastic bollards (flex-post delineators) centered in the striped
    no-parking buffer between a marked-parking lane and the curb - i.e. on
    the OUTSIDE of the parking lane (the curb side), protecting/delineating
    parked cars from that buffer, the mirror image of add_bollards (which
    centers bollards in a lane-narrowing buffer on the travel-lane side).
    Requires add_marked_parking to already be applied to this (leg_name,
    side) with curb_offset_ft > 0 - there's no buffer to put bollards in
    otherwise."""
    zone = state.parking_zones.get((leg_name, side))
    if zone is None:
        raise KeyError(f"({leg_name!r}, {side!r}) has no marked parking - call add_marked_parking first.")
    if not zone["curb_offset_ft"]:
        raise ValueError(
            f"({leg_name!r}, {side!r})'s marked parking has curb_offset_ft=0 - no curb buffer to put bollards in.")
    new_state = state.clone()
    new_state.parking_buffer_bollards[(leg_name, side)] = spacing_ft
    new_state.notes.append(f"add_parking_buffer_bollards({leg_name}, side={side!r}, spacing_ft={spacing_ft})")
    return new_state


def add_bollards(state: DesignState, leg_name: str, spacing_ft: float = BOLLARD_DEFAULT_SPACING_FT) -> DesignState:
    """Plastic bollards (flex-post delineators) down the center of a leg's
    painted lane-narrowing buffer (add_lane_narrowing) - a firmer, but still
    fully paint-plus-delineator (no curb/pavement change) escalation of that
    same treatment. Requires add_lane_narrowing to already be applied to this
    leg - a bollard line only makes sense inside a buffer that exists, and its
    lateral placement (centered in that buffer) is derived from the buffer's
    own stripe_width_ft, not a separately-specified position."""
    if leg_name not in state.lane_narrowing:
        raise KeyError(f"Leg {leg_name!r} has no lane_narrowing buffer - call add_lane_narrowing first.")
    new_state = state.clone()
    new_state.bollard_lines[leg_name] = spacing_ft
    new_state.notes.append(f"add_bollards({leg_name}, spacing_ft={spacing_ft})")
    return new_state


def add_corner_hatching(state: DesignState, corner: tuple[str, str],
                         depth_ft: float = CORNER_HATCHING_DEFAULT_DEPTH_FT) -> DesignState:
    """Paint-only diagonal hatching in a corner's gutter zone: a visual
    narrowing cue with zero curb/fillet geometry change - the paint-only
    alternative to bump_out() at the same corner. `corner` is a (leg_a, leg_b)
    key as produced by build_corner_fillets."""
    if corner not in state.corner_fillets:
        raise KeyError(f"Corner {corner} references a fillet not present in this state.")
    new_state = state.clone()
    new_state.corner_hatching[corner] = depth_ft
    new_state.notes.append(f"add_corner_hatching({corner}, depth_ft={depth_ft})")
    return new_state


def add_mountable_apron(state: DesignState, corner: tuple[str, str],
                         extent_ft: float = CORNER_APRON_DEFAULT_EXTENT_FT) -> DesignState:
    """Mountable apron: a textured (not painted-line) surface treatment at a
    corner, flush with the existing pavement grade - visually/optically
    narrows the corner for pedestrians while remaining fully drivable (e.g. by
    a fire apparatus's rear wheels during a wide turn) since no curb or
    elevation change is introduced. Same footprint as add_corner_hatching, a
    different real-world treatment for corners where a hard bump-out isn't an
    option (see fire_apparatus_constraint in a proposal's spec)."""
    if corner not in state.corner_fillets:
        raise KeyError(f"Corner {corner} references a fillet not present in this state.")
    new_state = state.clone()
    new_state.corner_aprons[corner] = extent_ft
    new_state.notes.append(f"add_mountable_apron({corner}, extent_ft={extent_ft})")
    return new_state


def shift_crosswalk_offset(state: DesignState, leg_name: str, delta_ft: float) -> DesignState:
    """Shift a leg's crosswalk further from (positive) or closer to (negative)
    the intersection, on top of whatever src/render/crosswalks.py:resolve_crosswalk_offsets
    would otherwise resolve (a real OSM-surveyed position or the geometric
    curve-clearance estimate) - e.g. to give a turning fire apparatus more room
    before it encounters the crosswalk mid-turn."""
    if leg_name not in state.legs:
        raise KeyError(f"Leg {leg_name!r} not present in this state.")
    new_state = state.clone()
    new_state.crosswalk_offset_overrides[leg_name] = (
        new_state.crosswalk_offset_overrides.get(leg_name, 0.0) + delta_ft
    )
    new_state.notes.append(f"shift_crosswalk_offset({leg_name}, delta_ft={delta_ft})")
    return new_state


def add_extra_prop(state: DesignState, leg_name: str, prop_type: str, offset_ft: float | None = None,
                    side: str = "right", note: str = "") -> DesignState:
    """Add one scenario-specific street-furniture prop (e.g. an RRFB, a
    relocated school-zone sign) along a leg - the treatment-level equivalent of
    a site config's `props.extra` (see sites/README.md), for props that only
    belong to this particular proposal, not every scenario at this site.

    offset_ft defaults to None, meaning "place it at this leg's real resolved
    crosswalk offset" (src/render/props.py:_extra_props_from_state falls back to it,
    same as _extra_props_from_config does for site-config props) - an RRFB or
    a relocated crossing sign belongs AT the crossing, and a real OSM-surveyed
    crosswalk can sit much farther from the corner than a small guessed
    number (e.g. ~42 ft on greenwood_ave_south here) - a hardcoded offset_ft
    can easily land inside the curb-return curve, in the roadway, instead of
    on the sidewalk. Only pass an explicit offset_ft when the prop genuinely
    belongs somewhere other than the crosswalk."""
    if leg_name not in state.legs:
        raise KeyError(f"Leg {leg_name!r} not present in this state.")
    new_state = state.clone()
    new_state.extra_props.append(
        {"leg": leg_name, "type": prop_type, "offset_ft": offset_ft, "side": side, "note": note}
    )
    new_state.notes.append(f"add_extra_prop({leg_name}, {prop_type!r}, offset_ft={offset_ft})")
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
