"""How well-sourced a number is, and which source wins when two disagree.

The project's core principle (README) is never to trust a generic guess when real
sourced data exists. That was originally expressed as a single boolean per leg -
`confirmed: true|false` - which conflated two very different things: a value nobody
measured, and a value derived from real third-party survey data like OpenStreetMap.
Those deserve different treatment and different rendering, so there are three tiers:

    FIELD_MEASURED  Someone put a tape measure (or a survey instrument) on the actual
                    roadway. This is reality. It supersedes everything else, including
                    OSM, and is never overridden automatically by anything in this
                    repo - if OSM disagrees with a field measurement, OSM is wrong.

    OSM_DERIVED     Computed from real third-party surveyed geometry (crossing ways,
                    sidewalk centerlines). Better than a guess, because someone
                    actually went and mapped it - but it is not a measurement of the
                    thing we want, so it carries its own error (a sidewalk centerline
                    is not a curb line).

    ESTIMATED       Inferred from something that only correlates with the answer -
                    platted ROW minus an assumed verge, a typical corner radius for
                    this kind of street. Fine as a placeholder, not fine to cost work
                    off, and must say so.

`confirmed: true|false` in an existing config.yaml still works and maps onto
FIELD_MEASURED / ESTIMATED, so nothing has to be rewritten at once. A leg can state
its tier explicitly with `width_provenance:` instead.
"""

FIELD_MEASURED = "field_measured"
OSM_DERIVED = "osm_derived"
ESTIMATED = "estimated"

VALID_PROVENANCE = (FIELD_MEASURED, OSM_DERIVED, ESTIMATED)

# Higher wins. Used to decide whether a newly-available source may replace a value.
_RANK = {FIELD_MEASURED: 3, OSM_DERIVED: 2, ESTIMATED: 1}

# How each tier is described in phase output, and drawn in the plan view. Kept here so
# the CLI summary, the 2D plot and its legend can't drift into describing the same leg
# three different ways.
LABEL = {
    FIELD_MEASURED: "FIELD-MEASURED",
    OSM_DERIVED: "OSM-DERIVED",
    ESTIMATED: "ESTIMATE / PLACEHOLDER",
}
PLOT_STYLE = {
    FIELD_MEASURED: {"color": "black", "linestyle": "-"},
    OSM_DERIVED: {"color": "darkviolet", "linestyle": "-."},
    ESTIMATED: {"color": "crimson", "linestyle": "--"},
}


def leg_width_provenance(leg_cfg: dict) -> str:
    """The tier a leg's `curb_to_curb_ft` belongs to.

    Explicit `width_provenance:` wins. Otherwise fall back to the older boolean:
    `confirmed: true` meant a field measurement, anything else meant a placeholder.
    """
    declared = leg_cfg.get("width_provenance")
    if declared in VALID_PROVENANCE:
        return declared
    if declared is not None:
        raise ValueError(
            f"Unknown width_provenance {declared!r} - expected one of {VALID_PROVENANCE}."
        )
    return FIELD_MEASURED if leg_cfg.get("confirmed") else ESTIMATED


def supersedes(candidate: str, existing: str) -> bool:
    """True if a `candidate`-tier value may replace an `existing`-tier one.

    Strictly greater, never equal: a second estimate is not an improvement on the
    first, and an OSM-derived width must never quietly displace a field measurement
    of the same thing.
    """
    return _RANK[candidate] > _RANK[existing]
