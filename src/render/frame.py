"""The one piece of ground both views are pointed at.

The plan view and the 3D render are the same reconstruction drawn twice, so a reader comparing
them is entitled to assume the two pictures cover the same street. They did not. The plan view
framed a hardcoded 110 ft square on the junction node; the 3D camera framed the pavement's own
extent, clipped to the modelled legs and grown 20%. Measured on the four sites, the 3D frame was
**1.15x to 1.57x** the 2D frame and centred 6.5-12.5 ft away from it - so the plan view cropped a
third of Broad St's modelled legs (and with them the far ends of the bike lanes the proposal
paints) while the render showed all of it, and by how much the two disagreed varied per site,
because one number was computed from the geometry and the other was a constant.

So the frame is computed once, here, and both views take it:

  * `plan_view.plot_design_state` sets its axis limits from it;
  * `export_scenario` writes it into the JSON as `frame`, and `blender_scene.py` frames its
    camera on that rather than recomputing an extent of its own.

WHAT THE FRAME IS. The extent of the pavement this project actually modelled, plus a small
margin. Two consequences worth stating:

  * Vertices past a leg's own far end are dropped. The pavement ring is stitched from traced OSM
    `barrier=kerb` ways, which do not stop where our leg does - at E Broad & Princeton one kerb
    runs 425 ft from the junction off a 130 ft leg, because the mapper drew the whole block in
    one way. That single vertex used to put the 3D camera at nearly twice the radius of the other
    three sites and not even pointed at the junction. A leg's far end is the edge of what was
    modelled; kerb beyond it is street, not junction.
  * It is measured from the MODEL, not from a DesignState. A curb extension moves the kerb, so
    framing on the resolved pavement would frame a proposal slightly differently from its own
    baseline - and these are published as before/after pairs, where a frame that shifts between
    the panels is exactly the thing that makes two pictures incomparable.

The 3D camera is a tilted perspective camera, so it necessarily sees ground beyond the frame:
what matches between the views is the subject and the radius, not the outline of the visible
region. The 2D axes are square; the render is 4:3.
"""
import math
import os
from dataclasses import dataclass

from shapely.geometry import Point

from src.geometry.model import build_pavement_polygon
from src.render.coords import FT_TO_M

# How much wider than the modelled pavement the frame is drawn. Tight on purpose: this is a
# drawing of one junction, and the paint and signage detail is the subject.
FRAME_MARGIN = 1.2

# A deliberate zoom-out, for a picture whose subject is longer than one junction - a corridor
# treatment down a street, say, where the point is that the bike lane runs the whole way rather
# than what colour the flex posts are. It scales the RADIUS only, so the frame stays centred on
# the same ground and both views widen together: the plan view's axes and the camera both come
# through junction_frame, and a knob that moved one of them would undo the whole reason this
# module exists.
#
# An environment variable because it has to reach two call sites several layers down (plot_design_
# state and export_scenario), which is the same problem HOPEWELL_RENDER_SCALE has and the same
# answer - see scripts/phase4_render_3d.py, which sets both from flags.
FRAME_SCALE_ENV = "HOPEWELL_FRAME_SCALE"


def frame_scale() -> float:
    """The zoom-out multiplier, or 1.0. Anything unparseable falls back with a warning rather than
    failing a batch of renders over an environment variable."""
    raw = os.environ.get(FRAME_SCALE_ENV, "1")
    try:
        scale = float(raw)
    except ValueError:
        print(f"  WARNING: {FRAME_SCALE_ENV}={raw!r} is not a number - framing at 1x.")
        return 1.0
    if scale <= 0:
        print(f"  WARNING: {FRAME_SCALE_ENV}={raw!r} is not positive - framing at 1x.")
        return 1.0
    return scale

# How far past a leg's far end a pavement vertex may still count as part of this junction. The
# corner fillets trim the curbs a little past the leg's own end, so a hard cut at the leg length
# would drop legitimate ring vertices.
LEG_REACH_TOLERANCE = 1.05


@dataclass(frozen=True)
class Frame:
    """Where both views point, and how much they take in. Feet, in the site's state plane."""
    center_ft: Point
    radius_ft: float

    def bounds_ft(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) - a square, which is what a plan view's axes want."""
        return (self.center_ft.x - self.radius_ft, self.center_ft.x + self.radius_ft,
                self.center_ft.y - self.radius_ft, self.center_ft.y + self.radius_ft)

    def as_local_m(self, center_ft: Point) -> dict:
        """The frame in the exported JSON's local-metre coordinates, for the 3D render."""
        return {"center_m": [(self.center_ft.x - center_ft.x) * FT_TO_M,
                            (self.center_ft.y - center_ft.y) * FT_TO_M],
                "radius_m": self.radius_ft * FT_TO_M}


def leg_reach_ft(model) -> float:
    """How far from the junction the modelled street extends, along its longest leg."""
    return max((model.center_ft.distance(Point(leg.centerline.coords[-1]))
                for leg in model.legs.values()), default=0.0)


def junction_frame(model) -> Frame:
    """The one frame both views draw, from the baseline pavement of `model`.

    Falls back to the leg reach itself where there is no pavement ring to measure - an
    unclosable ring is reported by check_pavement_ring rather than being fatal here, and a
    render with no frame at all would be worse than one framed on the legs.
    """
    reach_ft = leg_reach_ft(model)
    try:
        pavement = build_pavement_polygon(model.corner_fillets)
    except Exception:
        pavement = None
    xs, ys = [], []
    if pavement is not None and not pavement.is_empty:
        for x, y in pavement.exterior.coords:
            if not reach_ft or math.hypot(x - model.center_ft.x,
                                          y - model.center_ft.y) <= reach_ft * LEG_REACH_TOLERANCE:
                xs.append(x)
                ys.append(y)
    scale = frame_scale()
    if not xs:
        return Frame(model.center_ft, max(reach_ft, 1.0) * FRAME_MARGIN * scale)
    center = Point((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
    radius = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 * FRAME_MARGIN
    return Frame(center, radius * scale)
