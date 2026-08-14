"""The config.yaml schema (src/site_schema.py).

Every test below is a way a real config.yaml can be wrong while still parsing as YAML. The
ones that matter most are not the type errors - those already failed loudly somewhere - but
the two silent classes: a misspelled key, which reads downstream as "this fact was omitted",
and a leg name that matches nothing, which reads as "there is no crosswalk here".
"""
import pytest

from src.site import list_sites, load_site_schema
from src.site_schema import SiteConfigError, validate_site_config

MINIMAL = {
    "data_sources": {"road_network": "data/net.geojson", "parcels": "data/parcels.shp"},
    "intersection": {
        "name": "Test St & Other St",
        "center_wgs84": [-74.76, 40.39],
        "street1": "Test St",
        "street2": "Other St",
        "anchor_query": "Test Street, Nowhere, NJ",
        "resolution_method": "made up for a test",
        "clip_radius_m": 150,
        "leg_working_length_ft": 130,
        "existing_marked_crosswalks": ["test_st_west"],
    },
    "legs": {
        "test_st_west": {"sri": "0001", "bearing_deg": 270.0, "street_name": "Test St W",
                          "curb_to_curb_ft": 34, "source": "estimate, for a test"},
        "test_st_east": {"sri": "0001", "bearing_deg": 90.0, "street_name": "Test St E",
                          "curb_to_curb_ft": 34, "source": "estimate, for a test"},
    },
    "treatments": {"existing_corner_radius_ft": 20,
                    "existing_corner_radius_source": "placeholder, for a test"},
}


def config(**overrides) -> dict:
    """A deep-enough copy of MINIMAL with `overrides` merged one level into each section."""
    import copy

    out = copy.deepcopy(MINIMAL)
    for section, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(section), dict):
            out[section] = {**out[section], **value}
        else:
            out[section] = value
    return out


def test_the_baseline_is_valid():
    """Guards every other test in this file: they all assert that ONE change makes a config
    invalid, which says nothing unless the unchanged config is valid."""
    assert validate_site_config(config()).legs.keys() == {"test_st_west", "test_st_east"}


@pytest.mark.parametrize("site", list_sites())
def test_every_real_site_validates(site):
    load_site_schema(site)


def test_centerline_styles_match_treatments():
    """src/site_schema.py copies this vocabulary rather than importing it (to keep shapely off
    the config-read path), so the copy has to be pinned to the original."""
    from src.geometry.treatments import VALID_CENTERLINE_STYLES as CANONICAL
    from src.site_schema import VALID_CENTERLINE_STYLES as MIRRORED

    assert set(MIRRORED) == set(CANONICAL)


# ---------------------------------------------------------------------------
# The silent failures - what this schema exists for
# ---------------------------------------------------------------------------

def test_a_misspelled_key_is_rejected():
    """`bearing_dg` used to mean "this leg has no bearing", indistinguishable from omitting it
    on purpose, and surfaced hundreds of lines away inside leg matching."""
    broken = config()
    broken["legs"]["test_st_west"]["bearing_dg"] = broken["legs"]["test_st_west"].pop("bearing_deg")
    with pytest.raises(SiteConfigError, match="bearing_dg"):
        validate_site_config(broken)


def test_a_misspelled_section_is_rejected():
    broken = config()
    broken["treatment"] = broken.pop("treatments")
    with pytest.raises(SiteConfigError) as excinfo:
        validate_site_config(broken)
    assert "treatment" in str(excinfo.value)


def test_a_crosswalk_naming_no_leg_is_rejected():
    """This is the quietest one: the name is looked up in a set, misses, and a crossing that
    exists on the real street is simply not drawn."""
    with pytest.raises(SiteConfigError, match="test_st_wst"):
        validate_site_config(config(intersection={"existing_marked_crosswalks": ["test_st_wst"]}))


def test_a_signal_corner_naming_no_leg_is_rejected():
    signals = {"source": "for a test",
               "corners": [{"legs": ["test_st_west", "nonexistent_leg"],
                             "pedestrian_head": "same_pole"}]}
    with pytest.raises(SiteConfigError, match="nonexistent_leg"):
        validate_site_config(config(signals=signals))


def test_every_dangling_leg_name_is_reported_at_once():
    """Collected, not raised at the first miss - renaming a leg breaks several references and
    should take one edit to fix, not one run per reference."""
    broken = config(
        intersection={"existing_marked_crosswalks": ["gone_a"]},
        signals={"source": "for a test", "no_turn_on_red_legs": ["gone_b", "gone_c"]},
    )
    with pytest.raises(SiteConfigError) as excinfo:
        validate_site_config(broken)
    message = str(excinfo.value)
    assert all(name in message for name in ("gone_a", "gone_b", "gone_c")), message


def test_an_unsourced_width_is_rejected():
    """The project's rule is that a number nobody can trace to a source is not a measurement.
    It was enforced by documentation only."""
    broken = config()
    del broken["legs"]["test_st_west"]["source"]
    with pytest.raises(SiteConfigError, match="source"):
        validate_site_config(broken)


def test_an_empty_source_is_rejected():
    """A present-but-blank source satisfies "the key exists" while asserting nothing."""
    broken = config()
    broken["legs"]["test_st_west"]["source"] = "   "
    with pytest.raises(SiteConfigError):
        validate_site_config(broken)


def test_an_off_globe_center_is_rejected():
    """center_wgs84 is [lon, lat], and a swap is the obvious mistake - but only a detectable
    one when a longitude lands in the latitude slot and blows the +/-90 bound."""
    with pytest.raises(SiteConfigError, match="off the globe"):
        validate_site_config(config(intersection={"center_wgs84": [-74.76, 140.39]}))


def test_a_swap_within_both_ranges_is_NOT_caught():
    """Pinning the limit of the check above, so nobody reads it as more than it is: Hopewell's
    own coordinate swaps to a legal lon/lat pair in the South Atlantic. Nothing in a schema can
    see that; `resolution_method` and phase1_audit are what actually guard it."""
    validate_site_config(config(intersection={"center_wgs84": [40.389179, -74.7619598]}))


def test_indistinguishable_bearings_are_rejected():
    """Legs sharing an SRI are told apart by nearest bearing, so two the same is not a warning,
    it is a coin flip about which half of the road is which."""
    broken = config()
    broken["legs"]["test_st_east"]["bearing_deg"] = 270.4
    with pytest.raises(SiteConfigError, match="indistinguishable bearings"):
        validate_site_config(broken)


def test_measured_at_without_a_measurement_is_rejected():
    """`width_measured_at: intersection` on an estimate makes it authoritative at the corner
    (src/provenance.py:field_measurement_governs_corner) over a kerb someone actually traced."""
    broken = config()
    broken["legs"]["test_st_west"]["width_measured_at"] = "intersection"
    with pytest.raises(SiteConfigError, match="asserts that one was"):
        validate_site_config(broken)


def test_measured_at_is_accepted_alongside_a_field_measurement():
    ok = config()
    ok["legs"]["test_st_west"]["width_provenance"] = "field_measured"
    ok["legs"]["test_st_west"]["width_measured_at"] = "intersection"
    assert validate_site_config(ok).legs["test_st_west"].width_measured_at == "intersection"


def test_an_unknown_centerline_style_is_rejected():
    broken = config()
    broken["legs"]["test_st_west"]["centerline_style"] = "dotted_purple"
    with pytest.raises(SiteConfigError, match="centerline_style"):
        validate_site_config(broken)


def test_a_negative_width_is_rejected():
    broken = config()
    broken["legs"]["test_st_west"]["curb_to_curb_ft"] = -1
    with pytest.raises(SiteConfigError, match="curb_to_curb_ft"):
        validate_site_config(broken)


def test_a_prop_without_a_note_is_rejected():
    props = {"extra": [{"type": "school_zone_sign", "leg": "test_st_west",
                         "offset_ft": 40, "side": "left"}]}
    with pytest.raises(SiteConfigError, match="note"):
        validate_site_config(config(props=props))


def test_the_error_names_the_file():
    """A build does four sites; "field required" with no filename means re-running them one at
    a time to find out which one it was."""
    broken = config()
    del broken["legs"]["test_st_west"]["source"]
    with pytest.raises(SiteConfigError, match=r"sites/somewhere/config\.yaml"):
        validate_site_config(broken, "sites/somewhere/config.yaml")
