from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path) as fh:
        return yaml.safe_load(fh)


def resolve_path(base: Path, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    return (base / p).resolve()
