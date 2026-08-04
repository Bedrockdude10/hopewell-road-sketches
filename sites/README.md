# Sites

Each subdirectory is one intersection/corridor study: a `config.yaml` and a
`scenarios.py`. Nothing in `src/` or `scripts/` hardcodes anything about a
specific intersection - see the main README's "Adding a new site" section for
the step-by-step process. This file documents the `config.yaml` schema.

Use `sites/broad_st_greenwood/config.yaml` as a working example of every field below.

## `config.yaml` schema

```yaml
data_sources:
  road_network: data/<file>.geojson   # paths relative to repo root - doesn't have to be NJDOT/statewide
  parcels: data/<file>.shp            # doesn't have to be Mercer County

intersection:
  name: "..."                         # human-readable, used in plot titles
  center_wgs84: [lon, lat]            # resolved once via `phase1_audit.py --street1/--street2/--anchor`
  street1: "..."                      # what phase1_audit.py used to resolve center_wgs84 -
  street2: "..."                      # kept so re-resolving later (e.g. after OSM edits) is one command
  anchor_query: "..."                 # a Nominatim-geocodable address/place to anchor the OSM search bbox
  resolution_method: >                # free text - document how you cross-checked the resolved point
    ...
  clip_radius_m: 150                  # how far out to load/clip the road network around the center
  leg_working_length_ft: 130          # DEFAULT for how far each leg's centerline extends from the
                                       # intersection - a leg may override it with working_length_ft
  existing_marked_crosswalks: [...]   # leg names that currently have ANY marked crosswalk (checked against
                                       # real imagery/knowledge, not assumed)

corridor:                             # free-form - corridor-level facts from an SLD or similar, for reference
  ...

legs:
  <leg_name>:                         # e.g. "main_st_north" - your own naming convention
    sri: "..."                        # road network's own ID field for this road (SRI for NJDOT)
    bearing_deg: 0-360                # REQUIRED, and the only thing that has to be geometrically accurate -
                                       # compass bearing (0=N, 90=E, clockwise) from the intersection center
                                       # OUTWARD along this leg. Used to tell apart multiple legs sharing the
                                       # same road (a through road produces 2, a dead-end stub produces 1) -
                                       # get this from the resolved centerline's own geometry, not a guess.
    street_name: "..."                # human-readable
    working_length_ft: <number>       # OPTIONAL - overrides intersection.leg_working_length_ft for this
                                       # leg alone. Legs are not equally honest at the same length: a curb
                                       # is drawn from tracing as far as the tracing goes and extrapolated
                                       # from a bearing past that, so how far a leg can be carried without
                                       # inventing kerb is a per-leg fact. Set this to show more of an
                                       # arterial than of a cross street, and keep it at or under the
                                       # furthest traced kerb vertex on the leg's SHORTER side.
    curb_to_curb_ft: <number>         # the actual width used for curb-line construction
    width_provenance: field_measured|osm_derived|estimated   # how well-sourced that width is.
                                       # See src/provenance.py. Optional - if omitted it's inferred from
                                       # `confirmed` below (true -> field_measured, else -> estimated), so
                                       # older configs keep working. Prefer stating it explicitly.
                                       #   field_measured - someone measured the real roadway. This is
                                       #     reality: it supersedes OSM and every estimate, and nothing in
                                       #     this repo will override it. If OSM disagrees, OSM is wrong, and
                                       #     phase2 reports it as a CONFLICT against OSM rather than the leg.
                                       #   osm_derived - computed from real third-party surveyed geometry
                                       #     (sidewalk centerlines, crossing ways). Better than a guess, but
                                       #     it measures something adjacent to what we want, so it carries
                                       #     its own error - say what the derivation was.
                                       #   estimated - inferred from something that only correlates (platted
                                       #     ROW minus an assumed verge, a typical radius). Placeholder only.
    confirmed: true|false             # legacy boolean, kept for back-compat. true == field_measured.
    source: >                         # REQUIRED always - cite where curb_to_curb_ft came from, and if it
      ...                             # isn't field_measured, the methodology AND its uncertainty range.

    # Phase 2 cross-checks every leg against OSM's surveyed sidewalk centerlines
    # (src/geometry/model.py:sidewalk_span_ft): the curb line must sit inside them. A
    # width leaving under ~3 ft between the curb and the sidewalk CENTERLINE is
    # physically impossible and is reported as IMPLAUSIBLE WIDTH. This is a bound and a
    # sanity check, not a measurement - the curb-to-sidewalk setback measured 11.8 ft/side
    # on one field-measured leg and 4.0 ft/side on another 100 ft away.
    centerline_style: single_yellow_dashed|double_yellow|none  # optional, defaults to single_yellow_dashed
                                       # (src/geometry/treatments.py:DEFAULT_CENTERLINE_STYLE) - what's actually
                                       # painted down the middle of this leg TODAY. No OSM tag for this; state
                                       # your source in a comment (street-view review, field survey, etc.) the
                                       # same way the `signals` block below does.

signals:                              # optional - only for signalized intersections. Presence of this block
                                       # IS what "signalized" means now (replaces the old `signalized: true`
                                       # flag, which nothing downstream ever read).
                                       #
                                       # TRAFFIC CONTROL PRECEDENCE (src/render/props.py:build_props):
                                       #   1. this block - direct observation, and the ONLY source that says
                                       #      where each pole and pedestrian head actually is. Supersedes OSM.
                                       #   2. OSM highway=stop / give_way nodes - surveyed, and they say which
                                       #      approaches are controlled. Used wherever we have no observation.
                                       #   3. a guess (one stop sign per approach) - only when neither exists,
                                       #      and labelled GUESS in the prop's `source`.
                                       # Phase 2 prints a NOTE if this block and OSM's traffic_signals node
                                       # disagree either way. The observation wins; the note is advisory.
  pole_type: >                        # free text - what kind of signal hardware (informs blender_scene.py's
    ...                               # procedural geometry, e.g. "full-width mast arm" or "pole-mounted
                                       # rigid/davit arm" - a mast arm's reach is derived from real adjacent
                                       # leg widths (src/render/props.py), not read from this string
  source: >                           # REQUIRED - how this was confirmed (street-view photo review, field
    ...                               # survey, etc.) - doesn't have to be a survey, but say what it actually is
  corners:
    - legs: [<leg_name>, <leg_name>]  # the two legs whose curbs meet at this corner - matched as a set
                                       # the same way build_corner_fillets() identifies corners internally
                                       # (order doesn't matter)
      pedestrian_head: same_pole|separate_pole   # is the ped signal head on the vehicle signal's pole,
                                                  # or its own separate pole?
  no_turn_on_red_legs: [<leg_name>, ...]  # legs where turning onto the cross street is restricted

props:                                # optional - fidelity-pass signage/props with no general derivation
  extra:
    - type: school_zone_sign          # or any type blender_scene.py knows how to draw
      leg: <leg_name>
      offset_ft: <number>             # distance from the intersection along that leg's centerline
      side: left|right                # which side of the leg (our own offset convention, not traffic direction)
      note: >                         # REQUIRED - why this prop exists / what it's based on (or isn't)
        ...

treatments:
  existing_corner_radius_ft: <number>
  existing_corner_radius_source: >    # REQUIRED - survey/measured, or estimated-and-say-so
    ...
```

## `scenarios.py`

Must expose:

```python
def build_demo_scenario(baseline: DesignState) -> DesignState:
    ...
```

Compose treatments from `src/geometry/treatments.py` (`bump_out`, `refuge_island`, `raise_crossing`,
`upgrade_crosswalk_markings`) - see `sites/broad_st_greenwood/scenarios.py` for a worked example.
Nothing stops you from adding more functions to a site's `scenarios.py` for
alternative scenarios (e.g. `build_minimal_scenario`); phase3/phase4 scripts
only call `build_demo_scenario` by convention, not by hard requirement.
