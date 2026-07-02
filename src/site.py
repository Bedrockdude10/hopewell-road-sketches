"""Site discovery and loading. A "site" is one intersection/corridor study:
a directory under sites/<name>/ with a config.yaml (schema documented in
sites/README.md) and a scenarios.py exposing build_demo_scenario(baseline).

src/ itself is a general-purpose library with no data specific to any one
intersection - everything that varies per-site (widths, bearings, which roads,
which treatments to demo) lives under sites/."""
import importlib.util
from pathlib import Path
from types import ModuleType

from src.config import load_config

SITES_DIR = Path(__file__).resolve().parent.parent / "sites"
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "output"
DEFAULT_SITE = "broad_st_greenwood"


def list_sites() -> list[str]:
    if not SITES_DIR.exists():
        return []
    return sorted(p.name for p in SITES_DIR.iterdir() if (p / "config.yaml").exists())


def site_dir(site: str) -> Path:
    path = SITES_DIR / site
    if not (path / "config.yaml").exists():
        available = ", ".join(list_sites()) or "(none found)"
        raise FileNotFoundError(f"No site {site!r} (expected {path / 'config.yaml'}). Available sites: {available}")
    return path


def load_site_config(site: str) -> dict:
    config = load_config(site_dir(site) / "config.yaml")
    config["_site"] = site  # stashed for scripts that want to name output dirs etc.
    return config


def site_output_dir(site: str) -> Path:
    """Every phase script writes to output/<site>/ rather than a flat shared
    output/ - keeps multiple sites' results (and Overpass caches) from colliding."""
    path = OUTPUT_ROOT / site
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_site_arg(parser):
    """Shared --site CLI flag for phase scripts. Returns the parser for chaining."""
    parser.add_argument("--site", default=DEFAULT_SITE,
                         help=f"Site name under sites/ (default: {DEFAULT_SITE}). Available: {', '.join(list_sites())}")
    return parser


DEFAULT_SCENARIO = "build_demo_scenario"


def add_scenario_arg(parser):
    """Shared --scenario CLI flag: the name of a function in the site's
    scenarios.py to build (e.g. build_proposal_a_paint_only) - lets a site
    define any number of named proposals beyond the one build_demo_scenario
    every phase script uses by default. Returns the parser for chaining."""
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO,
                         help=f"Function name in the site's scenarios.py to build (default: {DEFAULT_SCENARIO})")
    return parser


def scenario_label(scenario_name: str) -> str:
    """Filename-safe label for a scenario's output files - 'proposed' for the
    default demo scenario (preserves the original geometry_proposed.json /
    phase4_render_proposed.png filenames), else the function name with its
    'build_' prefix stripped (e.g. build_proposal_a_paint_only -> proposal_a_paint_only)."""
    if scenario_name == DEFAULT_SCENARIO:
        return "proposed"
    return scenario_name[len("build_"):] if scenario_name.startswith("build_") else scenario_name


def load_site_scenarios(site: str) -> ModuleType:
    """Dynamically import sites/<site>/scenarios.py and return the module."""
    path = site_dir(site) / "scenarios.py"
    if not path.exists():
        raise FileNotFoundError(f"Site {site!r} has no scenarios.py at {path}")
    spec = importlib.util.spec_from_file_location(f"sites.{site}.scenarios", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
