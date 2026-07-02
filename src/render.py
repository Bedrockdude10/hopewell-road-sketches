"""Plan-view rendering: draws an IntersectionModel + DesignState to a matplotlib axis."""
import geopandas as gpd
import numpy as np
from matplotlib.lines import Line2D

from src.geometry_model import build_pavement_polygon
from src.intersection import IntersectionModel
from src.treatments import DesignState


def plot_design_state(ax, model: IntersectionModel, state: DesignState, title: str, dimension_labels: bool = True):
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
        Line2D([0], [0], color="saddlebrown", lw=1.5, label="Corner parcel"),
        Line2D([0], [0], marker="o", color="blue", lw=0, label="Intersection"),
    ]
