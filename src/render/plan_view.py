"""Plan-view rendering: draws an IntersectionModel + DesignState to a matplotlib axis."""
import geopandas as gpd
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.geometry.model import (
    bollard_points_ft, build_pavement_polygon, corner_overlay_polygon, lane_narrowing_edge_lines_ft,
    lane_narrowing_polygons_ft, lane_narrowing_taper_ft, lane_narrowing_taper_polygons_ft, leg_clearance_ft,
    parking_lane_edge_line_ft, parking_stall_count_ft, parking_stall_lines_ft,
)
from src.geometry.intersection import IntersectionModel
from src.geometry.treatments import LEGAL_PARKING_SETBACK_FT, DesignState
from src.render.crosswalks import CROSSWALK_CLEARANCE_FT, resolve_crosswalk_offsets
from src.sources.osm_context import fetch_crossings

BUILDING_CONTEXT_RADIUS_M = 130  # matches src/render/export.py - same real-world radius crossings are searched
                                  # within, so a leg's crosswalk_offset here matches what the 3D export computes


def plot_design_state(ax, model: IntersectionModel, state: DesignState, title: str, dimension_labels: bool = True,
                       crossings: list[dict] | None = None):
    model.parcels.boundary.plot(ax=ax, color="tan", linewidth=0.6, zorder=1)
    model.corner_parcels.boundary.plot(ax=ax, color="saddlebrown", linewidth=1.5, zorder=1)

    try:
        pavement = build_pavement_polygon(state.corner_fillets)
        gpd.GeoSeries([pavement]).plot(ax=ax, color="#d9d9d9", zorder=2)
    except ValueError:
        pass

    for name, leg in state.legs.items():
        confirmed = model.config["legs"][name].get("confirmed")
        color = "black" if confirmed else "crimson"
        style = "-" if confirmed else "--"
        gpd.GeoSeries([leg.left_curb]).plot(ax=ax, color=color, linewidth=2, linestyle=style, zorder=3)
        gpd.GeoSeries([leg.right_curb]).plot(ax=ax, color=color, linewidth=2, linestyle=style, zorder=3)
        if dimension_labels:
            mid = leg.centerline.interpolate(min(leg.centerline.length * 0.85, leg.centerline.length - 5))
            ax.annotate(f"{leg.curb_to_curb_ft:.1f} ft", (mid.x, mid.y), fontsize=7, color=color,
                        ha="center", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    for corner, pieces in state.corner_fillets.items():
        if "error" in pieces:
            continue
        gpd.GeoSeries([pieces["arc"]]).plot(ax=ax, color="darkorange", linewidth=2.5, zorder=4)
        if dimension_labels and "radius_ft" in pieces:
            mid = pieces["arc"].interpolate(0.5, normalized=True)
            ax.annotate(f"R={pieces['radius_ft']:.0f} ft", (mid.x, mid.y), fontsize=7, color="darkorange",
                        fontweight="bold", ha="center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    for name, island in state.refuge_islands.items():
        poly = island["polygon"]
        gpd.GeoSeries([poly]).plot(ax=ax, color="seagreen", alpha=0.6, zorder=5)
        gpd.GeoSeries([poly]).boundary.plot(ax=ax, color="darkgreen", linewidth=1, zorder=5)
        if dimension_labels:
            c = poly.centroid
            ax.annotate(f"refuge\n{island['width_ft']:.0f} ft", (c.x, c.y), fontsize=6.5, color="darkgreen",
                        ha="center", va="center", fontweight="bold")

    for name, poly in state.raised_crossings.items():
        gpd.GeoSeries([poly]).plot(ax=ax, color="slateblue", alpha=0.35, hatch="//", zorder=2)
        gpd.GeoSeries([poly]).boundary.plot(ax=ax, color="slateblue", linewidth=1, zorder=2)
        if dimension_labels:
            c = poly.centroid
            ax.annotate("raised\ncrossing", (c.x, c.y), fontsize=6.5, color="indigo",
                        ha="center", va="center", fontweight="bold")

    needs_crosswalk_offsets = bool(state.lane_narrowing) or bool(state.parking_zones)
    if needs_crosswalk_offsets and crossings is None:
        # Only fetched when actually needed (lane-narrowing, marked parking - which needs the real
        # crosswalk to compute its own legal start point, see LEGAL_PARKING_SETBACK_FT below) - phase2's
        # bare baseline call never has either, so it never pays for this network round-trip.
        crossings = fetch_crossings(model.center_wgs84, radius_m=BUILDING_CONTEXT_RADIUS_M)
    crosswalk_offsets = resolve_crosswalk_offsets(state, crossings or []) if needs_crosswalk_offsets else {}

    for leg_name, stripe_width_ft in state.lane_narrowing.items():
        line_only = leg_name in state.lane_narrowing_line_only
        sides = state.lane_narrowing_sides.get(leg_name, ("left", "right"))
        anchor_ft = leg_clearance_ft(leg_name, state.legs, state.corner_fillets)
        target_ft = crosswalk_offsets[leg_name][0] + CROSSWALK_CLEARANCE_FT
        leg = state.legs[leg_name]

        if line_only:
            for line in lane_narrowing_edge_lines_ft(leg, stripe_width_ft,
                                                      start_left_ft=anchor_ft, start_right_ft=anchor_ft,
                                                      sides=sides):
                gpd.GeoSeries([line]).plot(ax=ax, color="goldenrod", linewidth=1.5, zorder=3)
        else:
            for poly in lane_narrowing_polygons_ft(leg, stripe_width_ft,
                                                    start_left_ft=anchor_ft, start_right_ft=anchor_ft, sides=sides):
                gpd.GeoSeries([poly]).plot(ax=ax, color="gold", alpha=0.5, hatch="//", zorder=3)
                gpd.GeoSeries([poly]).boundary.plot(ax=ax, color="goldenrod", linewidth=1, zorder=3)
            for poly in lane_narrowing_taper_polygons_ft(leg, stripe_width_ft, anchor_ft, target_ft, sides=sides):
                gpd.GeoSeries([poly]).plot(ax=ax, color="gold", alpha=0.5, hatch="//", zorder=3)

        # The taper's own curve (src/geometry/model.py:lane_narrowing_taper_ft) - drawn either way,
        # since it's the boundary line itself (chevron-filled or not, per line_only above), the same
        # curve src/render/export.py feeds the 3D render so the two views can be checked against
        # each other directly instead of trying to eyeball it off the 3D render alone.
        for taper in lane_narrowing_taper_ft(leg, stripe_width_ft, anchor_ft, target_ft, sides=sides):
            gpd.GeoSeries([taper]).plot(ax=ax, color="goldenrod", linewidth=1.5, zorder=3)

        if dimension_labels:
            # One label PER SIDE actually narrowed, offset into that lane itself - not a single label
            # sitting on the centerline, which reads as "this road is one 11 ft lane" instead of what's
            # actually there (see sides above - not always both).
            lane_ft = leg.curb_to_curb_ft / 2 - stripe_width_ft
            along_dist = min(leg.centerline.length * 0.6, leg.centerline.length - 5)
            for side, sign in (("left", 1), ("right", -1)):
                if side not in sides:
                    continue
                lane_mid = leg.centerline.offset_curve(sign * lane_ft / 2).interpolate(along_dist)
                ax.annotate(f"lane {lane_ft:.1f} ft", (lane_mid.x, lane_mid.y), fontsize=6.5, color="goldenrod",
                            ha="center", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

    for corner, depth_ft in state.corner_hatching.items():
        if "error" in state.corner_fillets[corner]:
            continue
        poly = corner_overlay_polygon(state.corner_fillets[corner], model.center_ft, depth_ft)
        gpd.GeoSeries([poly]).plot(ax=ax, color="gold", alpha=0.5, hatch="//", zorder=3)
        gpd.GeoSeries([poly]).boundary.plot(ax=ax, color="goldenrod", linewidth=1, zorder=3)

    for corner, extent_ft in state.corner_aprons.items():
        if "error" in state.corner_fillets[corner]:
            continue
        poly = corner_overlay_polygon(state.corner_fillets[corner], model.center_ft, extent_ft)
        gpd.GeoSeries([poly]).plot(ax=ax, color="peru", alpha=0.6, zorder=3)
        gpd.GeoSeries([poly]).boundary.plot(ax=ax, color="saddlebrown", linewidth=1, zorder=3)
        if dimension_labels:
            c = poly.centroid
            ax.annotate("mountable\napron", (c.x, c.y), fontsize=6, color="saddlebrown",
                        ha="center", va="center", fontweight="bold")

    for (leg_name, side), zone in state.parking_zones.items():
        leg = state.legs[leg_name]
        anchor_ft = leg_clearance_ft(leg_name, state.legs, state.corner_fillets)
        legal_start_ft = crosswalk_offsets[leg_name][0] + LEGAL_PARKING_SETBACK_FT
        parking_start_ft = max(anchor_ft, legal_start_ft)
        depth_ft, stall_length_ft, curb_offset_ft = zone["depth_ft"], zone["stall_length_ft"], zone["curb_offset_ft"]
        edge = parking_lane_edge_line_ft(leg, side, depth_ft, parking_start_ft, curb_offset_ft=curb_offset_ft)
        gpd.GeoSeries([edge]).plot(ax=ax, color="steelblue", linewidth=1.5, zorder=3)
        for divider in parking_stall_lines_ft(leg, side, depth_ft, stall_length_ft, parking_start_ft,
                                               curb_offset_ft=curb_offset_ft):
            gpd.GeoSeries([divider]).plot(ax=ax, color="steelblue", linewidth=1, zorder=3)
        if dimension_labels:
            n_stalls = parking_stall_count_ft(leg, stall_length_ft, parking_start_ft)
            mid = edge.interpolate(0.5, normalized=True)
            ax.annotate(f"parking\n{n_stalls} stalls ({depth_ft:.0f} ft)", (mid.x, mid.y), fontsize=6, color="steelblue",
                        ha="center", va="center", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))

        # The striped no-parking buffer between the parking lane and the curb itself
        # (add_marked_parking's curb_offset_ft) - same chevron treatment as a lane-narrowing buffer,
        # just built with `sides=` restricted to this one side (see export.py's mirroring block).
        if curb_offset_ft:
            target_ft = crosswalk_offsets[leg_name][0] + CROSSWALK_CLEARANCE_FT
            for poly in lane_narrowing_polygons_ft(leg, curb_offset_ft, start_left_ft=anchor_ft,
                                                    start_right_ft=anchor_ft, sides=(side,)):
                gpd.GeoSeries([poly]).plot(ax=ax, color="gold", alpha=0.5, hatch="//", zorder=3)
                gpd.GeoSeries([poly]).boundary.plot(ax=ax, color="goldenrod", linewidth=1, zorder=3)
            for poly in lane_narrowing_taper_polygons_ft(leg, curb_offset_ft, anchor_ft, target_ft, sides=(side,)):
                gpd.GeoSeries([poly]).plot(ax=ax, color="gold", alpha=0.5, hatch="//", zorder=3)
            for taper in lane_narrowing_taper_ft(leg, curb_offset_ft, anchor_ft, target_ft, sides=(side,)):
                gpd.GeoSeries([taper]).plot(ax=ax, color="goldenrod", linewidth=1.5, zorder=3)
            if (leg_name, side) in state.parking_buffer_bollards:
                spacing_ft = state.parking_buffer_bollards[(leg_name, side)]
                points = bollard_points_ft(leg, curb_offset_ft, anchor_ft, spacing_ft, sides=(side,))
                if points:
                    xs, ys = zip(*points)
                    ax.scatter(xs, ys, color="darkorange", marker="o", s=10, zorder=6)

    for leg_name, spacing_ft in state.bollard_lines.items():
        stripe_width_ft = state.lane_narrowing[leg_name]
        start_ft = leg_clearance_ft(leg_name, state.legs, state.corner_fillets)
        points = bollard_points_ft(state.legs[leg_name], stripe_width_ft, start_ft, spacing_ft)
        if points:
            xs, ys = zip(*points)
            ax.scatter(xs, ys, color="darkorange", marker="o", s=10, zorder=6)

    ax.scatter([model.center_ft.x], [model.center_ft.y], color="blue", zorder=6, s=40)
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    zoom_ft = 110
    ax.set_xlim(model.center_ft.x - zoom_ft, model.center_ft.x + zoom_ft)
    ax.set_ylim(model.center_ft.y - zoom_ft, model.center_ft.y + zoom_ft)
    ax.set_xlabel("Feet (EPSG:3424)")


def legend_handles():
    return [
        Line2D([0], [0], color="black", lw=2, label="Curb line - confirmed width"),
        Line2D([0], [0], color="crimson", lw=2, ls="--", label="Curb line - estimate/placeholder width"),
        Line2D([0], [0], color="darkorange", lw=2.5, label="Corner fillet (radius labeled)"),
        Line2D([0], [0], color="seagreen", lw=6, alpha=0.6, label="Pedestrian refuge island"),
        Line2D([0], [0], color="slateblue", lw=6, alpha=0.35, label="Raised crossing"),
        Patch(facecolor="gold", alpha=0.5, hatch="//", edgecolor="goldenrod", label="Lane narrowing / corner hatching"),
        Line2D([0], [0], color="goldenrod", lw=1.5, label="Lane narrowing - line only (no chevron fill)"),
        Patch(facecolor="peru", alpha=0.6, edgecolor="saddlebrown", label="Mountable apron"),
        Line2D([0], [0], marker="o", color="darkorange", lw=0, label="Bollard"),
        Line2D([0], [0], color="steelblue", lw=1.5, label="Marked parking lane + stalls"),
        Line2D([0], [0], color="saddlebrown", lw=1.5, label="Corner parcel"),
        Line2D([0], [0], marker="o", color="blue", lw=0, label="Intersection"),
    ]
