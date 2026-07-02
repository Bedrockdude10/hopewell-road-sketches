"""Generic YAML config loader. See src/site.py for resolving a site's config.yaml
by name - this module has no knowledge of sites, paths are just files."""
from pathlib import Path

import yaml


def load_config(path: Path | str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
