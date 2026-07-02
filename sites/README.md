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
  leg_working_length_ft: 130          # how far each leg's centerline extends from the intersection
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
    curb_to_curb_ft: <number>         # the actual width used for curb-line construction
    confirmed: true|false             # true = field-measured/surveyed; false = geometric/estimated placeholder
    source: >                         # REQUIRED if confirmed: false - explain the estimation methodology
      ...                             # and REQUIRED either way - cite where curb_to_curb_ft came from

signals:                              # optional - only for signalized intersections. Presence of this block
                                       # IS what "signalized" means now (replaces the old `signalized: true`
                                       # flag, which nothing downstream ever read).
  pole_type: >                        # free text - what kind of signal hardware (informs blender_scene.py's
    ...                               # procedural geometry, e.g. "pole-mounted rigid/davit arm")
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

Compose treatments from `src/treatments.py` (`bump_out`, `refuge_island`, `raise_crossing`,
`upgrade_crosswalk_markings`) - see `sites/broad_st_greenwood/scenarios.py` for a worked example.
Nothing stops you from adding more functions to a site's `scenarios.py` for
alternative scenarios (e.g. `build_minimal_scenario`); phase3/phase4 scripts
only call `build_demo_scenario` by convention, not by hard requirement.
