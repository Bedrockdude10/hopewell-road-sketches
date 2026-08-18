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
    """One scenario-specific street-furniture prop (an RRFB, a relocated school-zone sign) along
    a leg - the treatment-level equivalent of a site config's `props.extra` (sites/README.md),
    for props belonging to this proposal rather than to every scenario at the site.

    offset_ft=None means "at this leg's real resolved crosswalk offset"
    (src/render/props.py:_extra_props_from_state). That is the default because an OSM-surveyed
    crosswalk can sit much farther from the corner than a guessed number (~42 ft on
    greenwood_ave_south), so a hardcoded offset lands inside the curb-return curve, in the
    roadway. Pass an explicit offset_ft only for a prop that belongs somewhere other than the
    crossing.
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
        """This prop in the shape src/render/props.py:_extra_prop places, which is also the shape
        a site config's `props.extra` uses - one placer, so a scenario-level prop and a
        site-level one cannot be positioned by different rules."""
        return {"leg": self.target.leg, "type": self.prop_type, "offset_ft": self.offset_ft,
                "side": str(self.side), "note": self.note}


def build_sidewalk_pieces(state: DesignState, sidewalk_width_ft: float = 6) -> list[Polygon]:
    """A sidewalk band hugging the real kerb, all the way round the junction.

    Built by widening each piece of the ACTUAL pavement boundary - the same (trimmed_a, arc,
    trimmed_b) build_pavement_polygon walks - and cutting the roadway back out, so every piece
    follows the traced kerb by construction. NOT by re-deriving the outer edge from the
    centerlines: that is only correct while the kerbs are symmetric offsets of the NJDOT
    centerline, which traced kerbs are not.

    Emitted as separate pieces rather than one ring: the renderer draws a polygon from its
    exterior only (scripts/blender/blender_scene.py:extrude_polygon), so a ring-with-a-hole would
    come out as a slab over the whole intersection.
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
