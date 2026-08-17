"""Treatments that belong to no family: an escape hatch for placing a prop, and the sidewalk
band every render draws behind the kerb."""
from dataclasses import dataclass

from shapely.geometry import Polygon

from src.geometry.targets import Side
from src.geometry.model import (build_pavement_polygon)
from src.geometry.treatments.base import Treatment
from src.geometry.treatments.state import DesignState



@dataclass(frozen=True)
class ExtraProp(Treatment):
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
    belongs somewhere other than the crosswalk.
    """
    prop_type: str = ""
    offset_ft: float | None = None
    side: Side = Side.RIGHT
    note: str = ""

    def __post_init__(self):
        if not self.prop_type:
            raise ValueError("An extra prop needs a type - it is what decides what is drawn.")
        object.__setattr__(self, "side", Side(self.side))

    def describe(self) -> str:
        return f"ExtraProp({self.target}, {self.prop_type!r}, offset_ft={self.offset_ft})"

    @property
    def entry(self) -> dict:
        """This prop in the shape src/render/props.py:_extra_prop places, which is also the
        shape a site config's `props.extra` list uses - the two go through one placer, so a
        scenario-level prop and a site-level one cannot end up positioned by different rules."""
        return {"leg": self.target.leg, "type": self.prop_type, "offset_ft": self.offset_ft,
                "side": str(self.side), "note": self.note}


def build_sidewalk_pieces(state: DesignState, sidewalk_width_ft: float = 6) -> list[Polygon]:
    """A sidewalk band hugging the real kerb, all the way round the junction.

    Built by widening each piece of the ACTUAL pavement boundary - the same
    (trimmed_a, arc, trimmed_b) that build_pavement_polygon walks - and cutting the roadway
    back out. Every piece therefore follows the surveyor's traced kerb by construction,
    including round the corner returns.

    It used to re-derive the outer edge instead: same centerlines, widened by
    2 * sidewalk_width_ft, re-filleted. That was correct only while the curbs were symmetric
    offsets of the NJDOT centerline, and they stopped being that when they became traced
    kerbs. Measured against the real pavement, 11-19% of the kerb had no sidewalk against it
    at all - gaps up to 27 ft where grass ran straight up to the roadway - and at W Broad
    658 sq ft of "sidewalk" lay inside the carriageway.

    Emitted as separate pieces rather than one ring on purpose: the renderer draws a
    polygon from its exterior only (scripts/blender/blender_scene.py:extrude_polygon), so a
    ring-with-a-hole would come out as a slab over the whole intersection.
    """
    try:
        pavement = build_pavement_polygon(state.corner_fillets)
    except ValueError:
        return []   # no closed roadway to lay a sidewalk against

    pieces = []
    for _corner, parts in state.corner_fillets.items():
        if "error" in parts:
            continue
        for key in ("trimmed_a", "arc", "trimmed_b"):
            edge = parts.get(key)
            if edge is None or edge.is_empty or edge.length <= 0:
                continue
            # Flat caps: a round cap would spill a half-disc past the end of each leg.
            band = edge.buffer(sidewalk_width_ft, cap_style=2).difference(pavement)
            if band.is_empty:
                continue
            pieces.extend(g for g in getattr(band, "geoms", [band])
                          if g.geom_type == "Polygon" and g.is_valid and not g.is_empty)
    return pieces
