"""The schema every site's config.yaml must satisfy, as code rather than prose.

A config.yaml is the entire factual basis for what gets drawn, so its failures are silent:
the render confidently describes a street that isn't there. The three shapes each rule here
exists for:

  * A TYPO IS A MISSING FACT. `bearing_dg:` is not an error anywhere - the key is simply
    absent, and the leg fails several hundred lines later inside leg matching, naming
    neither the file nor the key. Hence `extra="forbid"` on every section.
  * A NAME THAT REFERS TO NOTHING. A crosswalk, signal corner or no-turn-on-red entry
    naming a leg that isn't there matches nothing and draws nothing, with no warning.
  * AN UNSOURCED NUMBER. Per README/STANDARDS.md, a width or radius nobody can trace to a
    source is a plausible-looking number, not a measurement; `source:` fields were required
    by documentation only.

Every site is validated on load (src/site.py:load_site_config), raising before any geometry
is built. Vocabularies are imported from their one home (provenance tiers from
src/provenance.py); the centerline styles are the sole copy, because importing them would
drag shapely into every phase script that only wanted to read a config.

This validates; it does not replace dict access. load_site_config still returns the plain
dict every call site reads - a gate at the boundary, not a migration. See load_site_schema()
for the typed view.
"""
import itertools
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, model_validator

from src.provenance import VALID_PROVENANCE, VALID_WIDTH_LOCATIONS

# Mirrors src.geometry.treatments.VALID_CENTERLINE_STYLES, which cannot be imported here
# without dragging shapely and the geometry stack into every config read. Kept honest by
# tests/test_site_schema.py:test_centerline_styles_match_treatments.
VALID_CENTERLINE_STYLES = ("single_yellow_dashed", "double_yellow", "none")

# Two legs of one junction cannot leave the centre on the same heading. The leg matcher
# (src/geometry/intersection/osm_roads.py:_assign_leg_pieces) tells legs sharing an SRI apart by
# picking the nearest bearing, so a duplicate does not produce a warning - it produces a
# coin flip about which half of a road is which, and a render that may be mirrored.
MIN_BEARING_SEPARATION_DEG = 1.0

# Non-empty AFTER STRIPPING: a `source: ""` - or the `source: >` block someone started and
# left blank, which YAML hands over as whitespace - satisfies "the key is present" while
# asserting nothing, and the whole point of these fields is that a number is traceable.
Sourced = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Strict(BaseModel):
    """Base for every section: an unrecognised key is an error, not something to ignore.
    Every other rule catches a fact stated wrongly; this catches a fact never stated at all
    because its key was misspelled."""
    model_config = ConfigDict(extra="forbid")


class DataSources(Strict):
    """Paths are relative to the repo root, and are NOT checked for existence here: the NJDOT
    network and the county parcels are large licensed downloads kept out of git, and the whole
    suite is designed to skip rather than fail when they are absent (tests/conftest.py)."""
    road_network: str
    parcels: str
    tax_list: str | None = None


class Intersection(Strict):
    name: str
    center_wgs84: tuple[float, float]
    street1: str
    street2: str
    anchor_query: str
    resolution_method: Sourced
    clip_radius_m: float = Field(gt=0)
    leg_working_length_ft: float = Field(gt=0)
    existing_marked_crosswalks: list[str] = []

    @model_validator(mode="after")
    def _on_the_globe(self):
        """center_wgs84 is [lon, lat], the reverse of spoken order — writing it backwards is
        the obvious mistake. This catches less than it looks: NJ coordinates are legal
        latitudes, so a swap lands in the ocean only by luck. resolution_method and
        phase1_audit are the real guards."""
        lon, lat = self.center_wgs84
        if not -180 <= lon <= 180 or not -90 <= lat <= 90:
            raise ValueError(
                f"center_wgs84 is [lon, lat] and got [{lon}, {lat}], which is off the globe - "
                "the usual cause is latitude and longitude the wrong way round.")
        return self


class Leg(Strict):
    sri: str
    # The only value in this file that has to be geometrically accurate (sites/README.md).
    bearing_deg: float = Field(ge=0, lt=360)
    street_name: str
    curb_to_curb_ft: float = Field(gt=0)
    source: Sourced
    working_length_ft: float | None = Field(default=None, gt=0)
    width_provenance: Literal[VALID_PROVENANCE] | None = None      # type: ignore[valid-type]
    width_measured_at: Literal[VALID_WIDTH_LOCATIONS] | None = None  # type: ignore[valid-type]
    confirmed: bool = False
    centerline_style: Literal[VALID_CENTERLINE_STYLES] | None = None  # type: ignore[valid-type]


class SignalCorner(Strict):
    """The two legs whose curbs meet at this corner - matched as a SET, the same way
    build_corner_fillets() identifies corners internally, so order does not matter."""
    legs: tuple[str, str]
    pedestrian_head: Literal["same_pole", "separate_pole"]

    @model_validator(mode="after")
    def _two_distinct_legs(self):
        if self.legs[0] == self.legs[1]:
            raise ValueError(f"a corner is where two DIFFERENT legs meet, got {self.legs[0]!r} twice")
        return self


class Signals(Strict):
    """Presence of this block IS what "signalized" means."""
    source: Sourced
    pole_type: str | None = None
    corners: list[SignalCorner] = []
    no_turn_on_red_legs: list[str] = []


class ExtraProp(Strict):
    type: str
    leg: str
    offset_ft: float
    side: Literal["left", "right"]
    # REQUIRED by sites/README.md: a prop with no stated basis is exactly the "plausible
    # looking" content this project's provenance rules exist to keep out of a render.
    note: Sourced


class Props(Strict):
    extra: list[ExtraProp] = []


class Treatments(Strict):
    existing_corner_radius_ft: float = Field(gt=0)
    existing_corner_radius_source: Sourced


class SiteConfig(Strict):
    data_sources: DataSources
    intersection: Intersection
    legs: dict[str, Leg] = Field(min_length=2)
    treatments: Treatments
    # Free-form by design (sites/README.md): corridor-level reference facts off an SLD, for a
    # human reader. Nothing derives geometry from it, so it is the one section that may carry
    # whatever a particular source happens to publish.
    corridor: dict = {}
    signals: Signals | None = None
    props: Props | None = None

    @model_validator(mode="after")
    def _leg_references_resolve(self):
        """Every leg name mentioned anywhere else in the file must be a leg. Collected into
        ONE error rather than raised at the first miss, as src/checks.py reports violations:
        three renamed legs should take one edit to fix, not three runs.
        """
        known = set(self.legs)
        problems = []

        def check(names, where):
            for name in names:
                if name not in known:
                    problems.append(f"  {where}: {name!r} is not a leg in this file")

        check(self.intersection.existing_marked_crosswalks, "intersection.existing_marked_crosswalks")
        if self.signals:
            for i, corner in enumerate(self.signals.corners):
                check(corner.legs, f"signals.corners[{i}].legs")
            check(self.signals.no_turn_on_red_legs, "signals.no_turn_on_red_legs")
        if self.props:
            for i, prop in enumerate(self.props.extra):
                check([prop.leg], f"props.extra[{i}].leg")

        if problems:
            raise ValueError(
                "leg name(s) that refer to nothing - these fail SILENTLY at render time, by "
                "matching no leg and drawing nothing:\n" + "\n".join(problems)
                + f"\n  legs defined here: {', '.join(sorted(known))}")
        return self

    @model_validator(mode="after")
    def _bearings_are_distinguishable(self):
        """No two legs may leave the junction on (almost) the same heading."""
        items = sorted((leg.bearing_deg, name) for name, leg in self.legs.items())
        for (bearing_a, name_a), (bearing_b, name_b) in itertools.pairwise(items):
            if bearing_b - bearing_a < MIN_BEARING_SEPARATION_DEG:
                raise ValueError(
                    f"legs {name_a!r} and {name_b!r} have indistinguishable bearings "
                    f"({bearing_a} and {bearing_b} deg). Legs sharing a road are told apart by "
                    "nearest bearing, so this makes which-half-is-which a coin flip.")
        return self

    @model_validator(mode="after")
    def _measured_at_needs_a_measurement(self):
        """`width_measured_at` says where a tape measure was laid. It is only meaningful on a
        leg that claims a field measurement, and on an estimate it reads as one
        (src/provenance.py:field_measurement_governs_corner turns the pair into authority at
        the corner)."""
        for name, leg in self.legs.items():
            measured = leg.width_provenance == "field_measured" or (
                leg.width_provenance is None and leg.confirmed)
            if leg.width_measured_at and leg.width_measured_at != "unknown" and not measured:
                raise ValueError(
                    f"leg {name!r} says width_measured_at={leg.width_measured_at!r} but its width "
                    f"is not a field measurement (width_provenance="
                    f"{leg.width_provenance!r}, confirmed={leg.confirmed}). Stating where a "
                    "measurement was taken asserts that one was.")
        return self


class SiteConfigError(ValueError):
    """A config.yaml that does not describe a buildable site."""


def validate_site_config(raw: dict, path: Path | str | None = None) -> SiteConfig:
    """Validate a parsed config.yaml, or raise SiteConfigError naming the file and every
    problem in it — one line per problem, because "field required" with no filename means
    re-running phase scripts one at a time to find out which.
    """
    try:
        return SiteConfig.model_validate(raw)
    except ValidationError as e:
        lines = []
        for error in e.errors():
            where = ".".join(str(part) for part in error["loc"]) or "(top level)"
            lines.append(f"  {where}: {error['msg']}")
        where_file = f" in {path}" if path else ""
        raise SiteConfigError(
            f"{len(lines)} problem(s){where_file}:\n" + "\n".join(lines)
            + "\n\nThe schema is src/site_schema.py; sites/README.md explains each field."
        ) from e
