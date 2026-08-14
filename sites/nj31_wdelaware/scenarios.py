"""Treatment proposals for NJ 31 & W Delaware Ave, Pennington.

PLACEHOLDER - replaced once phase2 has settled this site's traced-kerb widths.
"""
from src.geometry.treatments import DesignState, all_crosswalks_continental, complete_centerlines


def build_demo_scenario(baseline: DesignState, model=None) -> DesignState:
    if model is None:
        return baseline
    return all_crosswalks_continental(complete_centerlines(baseline))
