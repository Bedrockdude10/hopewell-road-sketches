"""How well-sourced a number is, and which source wins when two disagree.

The project's core principle (README) is never to trust a generic guess when real sourced
data exists. A single boolean cannot say that, because it conflates a value nobody measured
with one derived from real third-party survey data, so there are three tiers:

    FIELD_MEASURED  A tape measure or survey instrument on the actual roadway. This is
                    reality. It supersedes everything else, including OSM, and is never
                    overridden automatically - if OSM disagrees, OSM is wrong.

    OSM_DERIVED     Computed from real third-party surveyed geometry (crossing ways,
                    sidewalk centerlines). Better than a guess, but not a measurement of
                    the thing we want, so it carries its own error: a sidewalk centerline
                    is not a curb line.

    ESTIMATED       Inferred from something that only correlates with the answer - platted
                    ROW minus an assumed verge, a typical corner radius for this kind of
                    street. Fine as a placeholder, not fine to cost work off, and must
                    say so.

The older `confirmed: true|false` still works and maps onto FIELD_MEASURED / ESTIMATED;
`width_provenance:` states a tier explicitly.
"""

FIELD_MEASURED = "field_measured"
OSM_DERIVED = "osm_derived"
ESTIMATED = "estimated"

VALID_PROVENANCE = (FIELD_MEASURED, OSM_DERIVED, ESTIMATED)

# Higher wins. Used to decide whether a newly-available source may replace a value.
_RANK = {FIELD_MEASURED: 3, OSM_DERIVED: 2, ESTIMATED: 1}

# How each tier is described in phase output and drawn in the plan view. The single home, so
# the CLI summary, the 2D plot and its legend cannot describe one leg three different ways.
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


def built_width_provenance(leg, leg_cfg: dict) -> str:
    """The tier of the width a Leg was actually BUILT with, not the one its config declares.

    They differ wherever the traced kerbs governed: a width measured between two mapped kerbs
    is osm_derived, so reporting the config's own "ESTIMATE / PLACEHOLDER" beside a curb line
    drawn from that trace describes the wrong number. The better of the two always wins, so a
    config that upgrades to a field measurement still shows through.
    """
    declared = leg_width_provenance(leg_cfg)
    built = getattr(leg, "width_provenance", None)
    return built if built in VALID_PROVENANCE and supersedes(built, declared) else declared


# WHERE a width was measured, which matters as much as how well-sourced it is. A tape
# measure is authoritative for the cross-section it was laid across, and these are old
# streets with uneven layouts - a road can genuinely be 68 ft mid-block and narrower at
# the corner. So a field measurement taken BESIDE a junction does not automatically
# describe the junction, and should not override corner-specific evidence there.
AT_INTERSECTION = "intersection"
NEAR_INTERSECTION = "near_intersection"
LOCATION_UNKNOWN = "unknown"
VALID_WIDTH_LOCATIONS = (AT_INTERSECTION, NEAR_INTERSECTION, LOCATION_UNKNOWN)


def leg_width_location(leg_cfg: dict) -> str:
    """Where a leg's width was measured. Defaults to LOCATION_UNKNOWN: claiming a
    measurement was taken at the corner is a positive assertion, so it has to be stated."""
    declared = leg_cfg.get("width_measured_at", LOCATION_UNKNOWN)
    if declared not in VALID_WIDTH_LOCATIONS:
        raise ValueError(
            f"Unknown width_measured_at {declared!r} - expected one of {VALID_WIDTH_LOCATIONS}."
        )
    return declared


def field_measurement_governs_corner(leg_cfg: dict) -> bool:
    """Whether this leg's width should be treated as authoritative AT THE JUNCTION.

    True only for a field measurement explicitly recorded as taken at the intersection.
    A field measurement from nearby, or of unrecorded location, still outranks estimates
    everywhere - but it does not outrank a kerb traced at the corner itself, because the
    two are describing different cross-sections of the same street.
    """
    return (leg_width_provenance(leg_cfg) == FIELD_MEASURED
            and leg_width_location(leg_cfg) == AT_INTERSECTION)


def supersedes(candidate: str, existing: str) -> bool:
    """True if a `candidate`-tier value may replace an `existing`-tier one.

    Strictly greater, never equal: a second estimate is not an improvement on the
    first, and an OSM-derived width must never quietly displace a field measurement
    of the same thing.
    """
    return _RANK[candidate] > _RANK[existing]
