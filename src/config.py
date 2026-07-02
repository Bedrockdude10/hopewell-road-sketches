"""Load intersection_config.yaml - the authoritative SLD/ground-truth values that
supersede whatever (if anything) the road network file records for width."""
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "intersection_config.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
