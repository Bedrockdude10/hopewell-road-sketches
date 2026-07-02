"""Street-furniture placement: streetlights, stop signs, traffic signals
(pole + pedestrian head), no-turn-on-red signs, and any site-specific extras
from config.yaml. Every function here only decides WHERE a prop goes and WHY
(a "source" string on every entry) - scripts/blender_props.py is what
actually draws it. See sites/README.md for the `signals`/`props.extra`
config schema this reads from."""
import numpy as np
from shapely.geometry import Point

from src.intersection import IntersectionModel
from src.treatments import DesignState

STREETLIGHT_SIDEWALK_SETBACK_FT = 4
SIGN_SIDEWALK_SETBACK_FT = 3
PED_HEAD_POLE_OFFSET_FT = 3  # lateral offset (tangent to the corner, along the sidewalk) for a pedestrian
                             # signal head confirmed to be on a separate pole from the vehicle signal -
                             # placement approximation, no surveyed separate-pole location available


def _corner_streetlight_props(corner_fillets: dict, center_ft: Point) -> list[dict]:
    """Streetlight at each corner's fillet-arc midpoint (real geometry), pushed a
    few feet further from the intersection center onto the sidewalk/corner -
    that offset distance is a placement approximation, flagged as such."""
    props = []
    for pieces in corner_fillets.values():
        if "error" in pieces:
            continue
        mid = pieces["arc"].interpolate(0.5, normalized=True)
        outward = np.array([mid.x - center_ft.x, mid.y - center_ft.y])
        norm = np.linalg.norm(outward)
        outward = outward / norm if norm > 1e-6 else np.array([1.0, 0.0])
        pos = (mid.x + outward[0] * STREETLIGHT_SIDEWALK_SETBACK_FT,
               mid.y + outward[1] * STREETLIGHT_SIDEWALK_SETBACK_FT)
        heading = np.degrees(np.arctan2(outward[1], outward[0]))
        props.append({
            "type": "streetlight", "position_ft": pos, "heading_deg": heading,
            "source": f"real: corner fillet arc midpoint; approximation: pushed {STREETLIGHT_SIDEWALK_SETBACK_FT} ft "
                      "outward onto the sidewalk (no surveyed pole location available)",
        })
    return props


def _leg_sign_position_ft(leg, offset_ft: float, side: str) -> tuple[tuple[float, float], float]:
    """A point offset_ft along a leg's centerline from the intersection, pushed
    laterally past the curb (left or right, per `side`) onto the sidewalk.
    Returns (position, heading_deg) with heading pointing back toward the road."""
    centerline = leg.centerline
    p = centerline.interpolate(min(offset_ft, centerline.length))
    p2 = centerline.interpolate(min(offset_ft + 1, centerline.length))
    u = np.array([p2.x - p.x, p2.y - p.y])
    u = u / np.linalg.norm(u)
    n = np.array([-u[1], u[0]]) if side == "left" else np.array([u[1], -u[0]])
    half_w = leg.curb_to_curb_ft / 2
    pos = (p.x + n[0] * (half_w + SIGN_SIDEWALK_SETBACK_FT), p.y + n[1] * (half_w + SIGN_SIDEWALK_SETBACK_FT))
    heading = np.degrees(np.arctan2(-n[1], -n[0]))  # face back toward the road
    return pos, heading


def _stop_sign_props(state: DesignState, offsets_ft: dict) -> list[dict]:
    """One stop sign per approach, placed on the leg's 'right' curb (our own
    left/right offset convention, not a real traffic-direction analysis) just
    past where the roadway straightens out. This is a placement approximation -
    real stop sign placement depends on engineering judgment not modeled here."""
    props = []
    for leg_name, leg in state.legs.items():
        offset_ft = offsets_ft[leg_name][0]
        pos, heading = _leg_sign_position_ft(leg, offset_ft, side="right")
        props.append({
            "type": "stop_sign", "position_ft": pos, "heading_deg": heading,
            "source": "approximation: placed on the leg's near-corner curb line, arbitrary side "
                      "(not a real traffic-direction/engineering placement study)",
        })
    return props


def _traffic_signal_props(model: IntersectionModel, state: DesignState, center_ft: Point) -> list[dict]:
    """
    Traffic signal pole + pedestrian signal head at each corner listed in the
    site config's `signals.corners` (see sites/README.md) - confirmed via
    direct street-view photo review, NOT a field survey, but real/observed
    rather than a geometric placeholder (distinct from e.g.
    existing_corner_radius_ft's estimate). Reuses the same real
    corner-fillet-arc-midpoint geometry as _corner_streetlight_props(); which
    way each signal head visually aims is a render-fidelity simplification,
    not a signal-timing-accurate model of actual head aiming.
    """
    signals_cfg = model.config.get("signals")
    if not signals_cfg:
        return []
    corner_cfg = {frozenset(c["legs"]): c for c in signals_cfg.get("corners", [])}
    confirmation = signals_cfg.get("source", "confirmed in site config.yaml (signals block)")

    props = []
    for (leg_a, leg_b), pieces in state.corner_fillets.items():
        if "error" in pieces:
            continue
        cfg = corner_cfg.get(frozenset((leg_a, leg_b)))
        if cfg is None:
            continue
        mid = pieces["arc"].interpolate(0.5, normalized=True)
        outward = np.array([mid.x - center_ft.x, mid.y - center_ft.y])
        norm = np.linalg.norm(outward)
        outward = outward / norm if norm > 1e-6 else np.array([1.0, 0.0])
        pole_pos = (mid.x + outward[0] * STREETLIGHT_SIDEWALK_SETBACK_FT,
                    mid.y + outward[1] * STREETLIGHT_SIDEWALK_SETBACK_FT)
        heading = np.degrees(np.arctan2(outward[1], outward[0]))
        props.append({
            "type": "traffic_signal_pole", "position_ft": pole_pos, "heading_deg": heading,
            "source": f"confirmed ({leg_a}/{leg_b} corner - {confirmation}): pole-mounted rigid/davit arm "
                      "signal, placed at the real corner-fillet arc midpoint (same position as the "
                      "streetlight prop here); exact arm/head aim is a render-fidelity simplification, "
                      "not a signal-timing-accurate model of which approach each head actually faces.",
        })

        same_pole = cfg.get("pedestrian_head") == "same_pole"
        if same_pole:
            ped_pos, ped_heading = pole_pos, heading
        else:
            tangent = np.array([-outward[1], outward[0]])
            ped_pos = (pole_pos[0] + tangent[0] * PED_HEAD_POLE_OFFSET_FT,
                       pole_pos[1] + tangent[1] * PED_HEAD_POLE_OFFSET_FT)
            ped_heading = heading
        props.append({
            "type": "pedestrian_signal_head", "position_ft": ped_pos, "heading_deg": ped_heading,
            "own_post": not same_pole,
            "source": f"confirmed ({leg_a}/{leg_b} corner - {confirmation}): " + (
                "pedestrian head mounted on the same pole as the vehicle signal."
                if same_pole else
                "pedestrian head is on a SEPARATE pole from the vehicle signal; approximation: offset "
                f"{PED_HEAD_POLE_OFFSET_FT} ft along the sidewalk from the vehicle signal pole (no "
                "surveyed separate-pole location available)."
            ),
        })
    return props


def _no_turn_on_red_props(model: IntersectionModel, state: DesignState, offsets_ft: dict) -> list[dict]:
    """NO TURN ON RED restriction signs for the legs listed in the site config's
    `signals.no_turn_on_red_legs` (confirmed via street-view photo review, not
    a signage-inventory survey). Positioned the same way as the automatic
    per-approach stop sign (_stop_sign_props) - same placement approximation."""
    signals_cfg = model.config.get("signals")
    if not signals_cfg:
        return []
    props = []
    for leg_name in signals_cfg.get("no_turn_on_red_legs", []):
        leg = state.legs.get(leg_name)
        if leg is None:
            continue
        offset_ft = offsets_ft[leg_name][0]
        pos, heading = _leg_sign_position_ft(leg, offset_ft, side="right")
        props.append({
            "type": "no_turn_on_red_sign", "position_ft": pos, "heading_deg": heading,
            "source": "confirmed (street-view photo review, site config.yaml signals.no_turn_on_red_legs) "
                      "that no-turn-on-red signage exists on this approach; placement approximation: same "
                      "near-corner curb-line pattern as _stop_sign_props (not a real traffic-engineering "
                      "placement study).",
        })
    return props


def _extra_props_from_config(model: IntersectionModel, state: DesignState, offsets_ft: dict) -> list[dict]:
    """User-specified extra signage (e.g. a school zone sign) from the site's
    config.yaml `props.extra` list - explicitly site-specific knowledge that
    doesn't belong in the general pipeline. See sites/README.md."""
    props = []
    for entry in model.config.get("props", {}).get("extra", []):
        leg = state.legs.get(entry["leg"])
        if leg is None:
            continue
        offset_ft = entry.get("offset_ft", offsets_ft.get(entry["leg"], (10, ""))[0])
        pos, heading = _leg_sign_position_ft(leg, offset_ft, side=entry.get("side", "right"))
        props.append({
            "type": entry["type"], "position_ft": pos, "heading_deg": heading,
            "source": f"user-specified in site config.yaml (props.extra): {entry.get('note', 'no note given')}",
        })
    return props


def build_props(model: IntersectionModel, state: DesignState, offsets_ft: dict, center_ft: Point) -> list[dict]:
    """All street-furniture props for one scenario export: a streetlight and a
    stop sign at every corner/approach (always), plus traffic signals and any
    site-specific extras where the site config asks for them."""
    return (
        _corner_streetlight_props(state.corner_fillets, center_ft)
        + _stop_sign_props(state, offsets_ft)
        + _traffic_signal_props(model, state, center_ft)
        + _no_turn_on_red_props(model, state, offsets_ft)
        + _extra_props_from_config(model, state, offsets_ft)
    )
